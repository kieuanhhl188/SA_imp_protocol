import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
import cupy as cp
from cuml.cluster import KMeans
from torch.utils.dlpack import to_dlpack
from cupy import fromDlpack
import math
import time

def run_clustering(tdict, num_clusters, observation_window=100, print_log=False, device=None):
    if device is None:
        device = "cuda:0"

    # initialize dicts to return
    centroids_tensor_dict = {}
    centroids_labels_dict = {}

    # compute num heads
    num_heads = tdict[0].shape[-3]
    num_lyrs = len(tdict)

    # arg list
    args_list = []

    # lists
    shm_list = []
    centroid_list = []
    label_list = []

    # compute shared prefix length
    shared_prefix_length = tdict[0].shape[-2]
    promptlen = shared_prefix_length - observation_window

    # loop over layers
    t1 = time.time()
    for layer_num in range(num_lyrs):
        if print_log:
            print('layer: ', layer_num)

        keys = tdict[layer_num].squeeze(0).float().to(device)

        K = num_clusters

        assert(len(keys.shape) == 3)

        if observation_window > 0:
            keys = keys[:,:-observation_window,:]
        num_heads = keys.shape[0]
        kdim = keys.shape[2]

        cluster_labels_list = []
        cluster_centers_list = []

        # iterate over heads
        for H in range(num_heads):
            head_data = keys[H]
            data_normalized = F.normalize(head_data, p=2, dim=-1)

            dlpack_tensor = to_dlpack(data_normalized)
            data_cp = fromDlpack(dlpack_tensor)

            kmeans = KMeans(
                n_clusters=K,
                max_iter=300,
                init='k-means++',  # Initialization method
                verbose=0,
                random_state=0
            )
            kmeans.fit(data_cp)
            cluster_labels = kmeans.labels_

            # convert labels to pytorch tensor
            dlpack_labels = cluster_labels.toDlpack()
            labels = torch.utils.dlpack.from_dlpack(dlpack_labels)

            # Compute cluster centers (centroids)
            cluster_centers = []
            for i in range(K):
                mask = labels == i
                cluster_keys = head_data[mask]
                if len(cluster_keys) > 0:
                    centroid = torch.mean(cluster_keys, dim=0)
                else:
                    centroid = torch.zeros(head_data.shape[1], dtype=head_data.dtype, device=head_data.device)
                cluster_centers.append(centroid)
            cluster_centers = torch.stack(cluster_centers, dim=0)

            cluster_labels_list.append(labels)
            cluster_centers_list.append(cluster_centers)

        a = torch.stack(cluster_centers_list, dim=0).unsqueeze(0)#.cpu()
        b = torch.stack(cluster_labels_list, dim=0).unsqueeze(0).to(torch.int64)#.cpu()

        centroids_tensor_dict[layer_num] = a
        centroids_labels_dict[layer_num] = b

    return centroids_tensor_dict, centroids_labels_dict

def run_global_threshold(
        key_dict, query_dict, centroids_tensor_dict, centroids_labels_dict, num_clusters,
        observation_window=100, print_log=False, device=None
    ):

    # copy to GPU 0 if not specified
    if device is None:
        device = "cuda:0"

    # get shared prefix length here
    shared_prefix_length = query_dict[0].shape[-2]
    num_heads = query_dict[0].shape[-3]
    num_lyrs = len(query_dict)

    # global dict for centroids
    K = num_clusters

    # loop over layers
    attn_score_centroid_list = []
    for layer_num in range(num_lyrs):
        if print_log:
            print('layer: ', layer_num)

        # load centroids
        centroids_tensor = centroids_tensor_dict[layer_num].squeeze(0).to(device)
        centroids_labels = centroids_labels_dict[layer_num].squeeze(0).to(device)

        keys = key_dict[layer_num].squeeze(0).to(device)
        queries = query_dict[layer_num].squeeze(0).to(device)
        keys_shared_prefix = keys[:, :-observation_window, :]

        # === GQA ===
        # Với model GQA (Qwen2.5-Coder: 28 head Q / 4 head KV), key và centroid có ít head
        # hơn query. Nhân bản centroid/label/key lên đủ số head Q để mỗi query head tự tính
        # ngưỡng riêng — đúng quy ước Appendix G của bài ("each head independently selects").
        # Với MHA thì num_key_value_groups == 1, khối này là no-op hoàn toàn, nên đường chạy
        # LLaMA/LongChat của gate Phase 0 không đổi một phép tính nào.
        num_q_heads = queries.shape[0]
        num_kv_heads = keys.shape[0]
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0, (
                f"num_attention_heads ({num_q_heads}) phải chia hết cho "
                f"num_key_value_heads ({num_kv_heads})"
            )
            groups = num_q_heads // num_kv_heads
            centroids_tensor = torch.repeat_interleave(centroids_tensor, groups, dim=0)
            centroids_labels = torch.repeat_interleave(centroids_labels, groups, dim=0)
            keys_shared_prefix = torch.repeat_interleave(keys_shared_prefix, groups, dim=0)

        # compute attention to centroids
        queries_obs_window = queries[:, -observation_window:, :].float() # only obs window queries
        attn_scores_centroids = torch.matmul(queries_obs_window, centroids_tensor.transpose(1, 2)) / math.sqrt(keys.shape[-1])

        # (khong con khoi tao `scores` bang zeros: ban vector hoa ben duoi sinh thang ra no)

        # === Vector hoa hai vong `for k in range(K)` ===
        # Ban goc lap Python qua K x num_layers = 789-1141 x 32 ~ 25.000-36.000 vong moi
        # mau tren RepoBench-P, moi vong thao tac tren tensor [H, S, obs]. Do la nut that
        # lam clustering RepoBench-P ton ~5 phut/mau (uoc ~37 gio cho 500 mau).
        #
        # Ca hai vong deu la phep co san, va thay the CHINH XAC TUNG BIT — khong phai xap xi:
        #
        #  1. scores[h,s,w] = attn_scores_centroids[h, w, labels[h,s]]
        #     Voi moi (h,s,w) chi dung MOT k khop label, nen tong qua K co duy nhat mot so
        #     hang khac 0, cong vao `scores` von khoi tao bang 0.0 -> ket qua y het gather.
        #
        #  2. num_keys_per_cluster[h,k] = so token cua head h mang nhan k -> scatter_add.
        #     Tong cac so 1.0 trong float32 la chinh xac tuyet doi den 2^24 = 16.7M, con S
        #     lon nhat o day ~23K.
        H_ = centroids_labels.shape[0]
        # [H, obs, K] -> [H, K, obs] roi lay theo nhan: out[h,s,w] = A[h, labels[h,s], w]
        _A = attn_scores_centroids.transpose(1, 2)
        scores = _A[torch.arange(H_, device=centroids_labels.device)[:, None], centroids_labels]

        # compute number of keys per cluster
        num_keys_per_cluster = torch.zeros(
            (keys_shared_prefix.shape[0], K), device=keys_shared_prefix.device
        )
        num_keys_per_cluster.scatter_add_(
            1, centroids_labels,
            torch.ones_like(centroids_labels, dtype=num_keys_per_cluster.dtype),
        )

        # === On dinh so hoc: TRU MAX TRUOC KHI exp ===
        # Ban goc goi torch.exp thang tren logit tho. Do la softmax chua chuan hoa:
        # float32 tran thanh inf khi logit vuot ~88.7, roi inf/inf = nan, va np.quantile
        # tren mang co nan tra ve nan. Nguong nan thi `score > threshold` LUON False ->
        # khong cluster nao duoc chon -> model mat toan bo fixed context, khong crash,
        # khong assert nao no. Do dung la ca Qwen2.5-Coder ngay 18/8: tau = nan o ca 4
        # quantile, Sq-70% tut tu 65.35 xuong 23.05.
        #
        # LLaMA/LongChat thoat vi logit nam trong dai an toan; Qwen2 co massive
        # activations nen logit lon hon han. Loi nay o code DUNG CHUNG, khong phai o
        # ban port GQA.
        #
        # Tru cung mot hang so M cho ca tu va mau thi ti so KHONG DOI:
        #     exp(s - M) / sum_k n_k*exp(a_k - M)  ==  exp(s) / sum_k n_k*exp(a_k)
        # M lay theo chieu cluster cho tung (head, token quan sat), nen moi so hang deu
        # <= 0 va exp bi chan boi 1.
        # Lay max CHI TREN CLUSTER CO KEY. Cluster rong bi run_clustering gan centroid =
        # VECTOR 0, nen diem cua no la q.0 = 0 — thuong cao hon moi cluster that khi diem
        # that deu am. De no lam max thi:
        #   - no gop 0 vao mau so (num_keys = 0)
        #   - moi cluster that thanh exp(rat am) -> underflow ve 0
        #   - mau so = 0, tu so = 0  ->  0/0 = NaN
        # Do la 0.64% diem nan con lai sau khi da tru max (do 18/8, khop ti le cluster rong
        # 0.8-1% ma inspect_centroids.py bao).
        #
        # Loai chung ra la dung ca ve so hoc lan ve ngu nghia: cluster rong dai dien cho 0
        # key, khong co ly do gi tham gia chuan hoa. Sau khi loai, cluster dat max chac chan
        # co num_keys >= 1 nen mau so >= 1 > 0 — khong the chia cho 0 nua.
        #
        # Ket qua KHONG DOI voi moi truong hop huu han: cluster rong von da gop 0 vao mau so
        # du co bi loai hay khong, con M thi trie tieu giua tu va mau.
        # Mask dung MOT LAN roi dung cho ca max lan tong. Dat -inf o cluster rong thi
        # exp(-inf - M) = 0 chinh xac, khong phu thuoc vao viec num_keys = 0 nhan voi cai gi.
        #
        # (Chi loai khoi max thoi thi KHONG du: M tut xuong muc cua cluster that, roi diem 0
        #  cua cluster rong thanh exp(0 - M) = exp(+150) = inf, va 0 * inf = NaN. Doi cho tran
        #  chu khong khu duoc no.)
        empty = (num_keys_per_cluster == 0).unsqueeze(-2)            # [H, 1, K]
        attn_scores_masked = attn_scores_centroids.masked_fill(empty, float("-inf"))
        amax = attn_scores_masked.amax(dim=-1, keepdim=True)         # [H, obs, 1]

        # estimate denominator here
        attn_scores_centroids_est_exp = torch.exp(attn_scores_masked - amax)
        num_keys_per_cluster = num_keys_per_cluster.unsqueeze(-2)
        denom_est_tmp = num_keys_per_cluster * attn_scores_centroids_est_exp
        denom_est = torch.sum(denom_est_tmp, dim=-1) # per-head estimate

        # divide centroid scores (copied per-token) by the denominator estimate
        # scores: [H, S_prefix, obs] | amax -> [H, 1, obs] de broadcast dung truc
        scores_scaled_sm = torch.exp(scores - amax.squeeze(-1).unsqueeze(-2)) / denom_est.unsqueeze(-2)

        # compute average across tokens
        scored_scaled_sm_sum = torch.mean(scores_scaled_sm, dim=-1, dtype=torch.float32)
        attn_score_centroid_list.append(scored_scaled_sm_sum)

        del keys
        del queries
        del keys_shared_prefix

    # stack all scores
    full_centroid_scores = torch.stack(attn_score_centroid_list, dim=0) # shape should be 32, 32, 5642

    # compute global thresholds here
    qlist = [0.5, 0.7, 0.8, 0.9]
    q = torch.tensor(qlist, device="cpu")

    # for long sequence lengths, we need to move to CPU
    full_centroid_scores_cpu = full_centroid_scores.cpu().numpy()

    # Chan nguong hong TAI CHO, thay vi de no xuong dia roi phat tac o pred.py sau nhieu gio.
    # np.quantile tra ve nan neu dau vao co nan, va nguong nan lam `score > threshold` luon
    # False -> khong cluster nao duoc chon. Trieu chung la accuracy tut sau, khong crash.
    n_bad = int((~np.isfinite(full_centroid_scores_cpu)).sum())
    if n_bad:
        total = full_centroid_scores_cpu.size
        raise RuntimeError(
            f"run_global_threshold: {n_bad}/{total} diem centroid khong huu han (nan/inf).\n"
            "Nguyen nhan thuong gap: tran so hoc o exp() khi logit attention qua lon "
            "(vd Qwen2 co massive activations).\n"
            "Khoi tinh exp da tru max VA loai cluster rong khoi phep lay max. Con bao "
            "loi nay nghia la key/query dau vao da chua nan/inf san — kiem hook thu q/k "
            "trong offline_clustering.py, hoac centroid co nan (chay inspect_centroids.py)."
        )

    quantile_result = np.quantile(full_centroid_scores_cpu, q)
    thresholds = torch.tensor(quantile_result)

    if not np.all(np.isfinite(quantile_result)):
        raise RuntimeError(
            f"run_global_threshold: quantile ra gia tri khong huu han {quantile_result}. "
            f"Khong luu file de tranh sinh du lieu hong."
        )

    tdict = {}
    i = 0
    for q_idx in qlist:
        tdict[q_idx] = thresholds[i].item()
        i += 1

    # save shared prefix length here
    tdict['shared_prefix_length'] = shared_prefix_length
    tdict['observation_window'] = observation_window

    return tdict
