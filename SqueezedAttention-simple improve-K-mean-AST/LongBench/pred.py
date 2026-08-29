import os
import warnings

# use this for now to filter out torch warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
import pickle
import textwrap
import sys
from squeezedattention.utils import build_chat, truncate_fn, apply_rope_scaling

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def _known_models():
    """Danh sách model hợp lệ lấy thẳng từ model2path.json.

    Trước đây danh sách này hard-code trong argparse, nên thêm model mới vào
    model2path.json/model2maxlen.json là chưa đủ — `--model` vẫn bị argparse từ chối
    (chính là ca qwen2.5-coder-7b-instruct của Phase 1.5). Đọc từ file để hai chỗ
    không bao giờ lệch nhau nữa.
    """
    with open(os.path.join(_CONFIG_DIR, "model2path.json"), "r", encoding="utf-8") as f:
        return sorted(json.load(f).keys())


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=_known_models())
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--path_to_clusters", type=str, default="/tmp")
    parser.add_argument("--use_centroids", action="store_true")
    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--percentile_lower", type=float, default=0.7)
    parser.add_argument("--obs_window", type=int, default=100)
    parser.add_argument("--rope_scaling", default=None,
                        help="dang 'dynamic:4' de dat 128K. PHAI trung voi cau hinh "
                             "da dung khi sinh centroid")
    parser.add_argument("--force_chat", action="store_true",
                        help="AP chat template cho ca lcc/repobench-p. LongBench co y BO "
                             "template o hai task nay; bat len la de KIEM CHUNG gia thuyet "
                             "'Instruct hong vi thieu template', khong phai cau hinh mac dinh")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed; protocol yêu cầu mean±std qua >=3 seed cho số accuracy chính")
    parser.add_argument("--run_tag", type=str, default="",
                        help="hau to thu muc ket qua: pred/<config>_run<TAG>. Bat buoc khi lap "
                             "lai CUNG mot cau hinh nhieu lan (mean+-std), vi ten thu muc mac "
                             "dinh chi gom model + percent_clusters + percentile -> lan sau ghi "
                             "de lan truoc va khong con gi de lay trung binh.")
    parser.add_argument("--resume", action="store_true",
                        help="neu file .jsonl da co, bo qua nhung dataidx da xong va chay "
                             "tiep phan con lai (append). Khong dung chung voi --overwrite. "
                             "eval.py key per_sample theo dataidx nen file khong theo thu tu "
                             "van cham dung.")
    parser.add_argument("--overwrite", action="store_true",
                        help="xoá file .jsonl cũ trước khi chạy. pred.py ghi ở chế độ append, "
                             "chạy lại mà không có cờ này sẽ nhân đôi prediction -> eval.py ra số sai")
    parser.add_argument("--limit", type=int, default=-1,
                        help="chỉ chạy N sample ĐẦU (smoke test); -1 = cả dataset. "
                             "Kết quả ghi vào thư mục có hậu tố _lim<N> để không lẫn với "
                             "lượt chạy đầy đủ. eval.py phải truyền cùng --limit để đọc được")
    return parser.parse_args(args)

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    # mp.spawn tạo process mới -> phải seed lại trong con, seed ở parent không lan sang
    seed_everything(config_params['seed'] + rank)
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, config_params)

    # iterate over longbench dataset
    for json_obj in tqdm(data):
        different_prefix_index = json_obj.pop('different_prefix_index')
        prompt = prompt_format.format(**json_obj)
        prompt_noquery = prompt_only_format.format(**json_obj)

        # perform truncation
        prompt, truncated_shared_prefix_length = truncate_fn(
            prompt, prompt_noquery, tokenizer, max_length, dataset, device,
            model_name=model_name, force_chat=config_params.get('force_chat', False))
        model.model.shared_prefix_length = truncated_shared_prefix_length
        model.model.different_prefix_index = different_prefix_index

        # encode input
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                use_cache=True
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            # dataidx = different_prefix_index (0..N-1, trung voi ten file centroid).
            # BAT BUOC cho paired test (Phase 5.5) va mean+-std qua nhieu seed: voi
            # world_size > 1 cac process cung append vao mot file nen thu tu dong bi
            # tron, khong the dung thu tu de ghep mau giua hai config.
            json.dump({"dataidx": different_prefix_index, "pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    if dist.is_initialized():
        dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device, config_params):
    is_llama_family = ("LLaMA-2-7B-32K" in model_name or "LWM" in model_name
                       or "longchat" in model_name)
    is_qwen2 = "qwen" in model_name.lower()

    if not (is_llama_family or is_qwen2):
        # Squeezed Attention mới chỉ patch cho models/llama và models/qwen2 của
        # transformers fork trong repo. Thêm model khác thì phải port trước.
        raise NotImplementedError(
            f"Chưa port Squeezed Attention cho model '{model_name}'. "
            f"Hiện hỗ trợ: LLaMA-2-7B-32K, LWM, longchat, qwen2*."
        )

    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(path)
    config = apply_rope_scaling(config, config_params.get('rope_scaling'))

    # set attn implementation
    config._flash_attn_2_enabled = True
    config._attn_implementation = "flash_attention_2"
    dtype = torch.bfloat16

    # clustering config parameters
    config.path_to_clusters_cosine = config_params['path_to_clusters_cosine']
    config.use_centroids = config_params['use_centroids']
    config.hierarchical_lookup = config_params['hierarchical_lookup']
    config.percent_clusters = config_params['percent_clusters']
    config.percent_clusters_l2 = config_params['percent_clusters_l2']
    config.percentile = config_params['percentile']
    config.percentile_lower = config_params['percentile_lower']
    config.obs_window = config_params['obs_window']

    if is_qwen2:
        # sliding window cắt bớt key theo cửa sổ -> mâu thuẫn với SA (cần toàn bộ
        # fixed context). Qwen2.5-Coder mặc định đã tắt; ép tắt cho chắc.
        if getattr(config, "use_sliding_window", False):
            print("[WARN] use_sliding_window=True -> ép về False cho Squeezed Attention")
            config.use_sliding_window = False

    model = AutoModelForCausalLM.from_pretrained(path, config=config, torch_dtype=dtype)
    model = model.to(device)

    if is_qwen2:
        # use_fast=False BẮT BUỘC: offline_clustering.py cũng load use_fast=False.
        # truncate_fn tính shared_prefix_length bằng tokenizer, nên hai bên dùng
        # tokenizer khác nhau mà lệch dù chỉ 1 token thì centroid_labels không còn
        # khớp vị trí key -> assert `shared_prefix_length` ở modeling_qwen2 (~L1347)
        # nổ sau khi đã nạp xong model 15 GB, mất trắng cả lượt chạy.
        # (Đường LLaMA vốn đã đúng: LlamaTokenizer là bản chậm.)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(path)

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    max_length = model2maxlen[model_name]

    if args.task is not None:
        datasets = [args.task] # only run single task
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "lcc", "repobench-p"]

    # config params
    config_params = {}
    config_params['use_centroids'] = args.use_centroids
    config_params['hierarchical_lookup'] = args.hierarchical_lookup
    config_params['percent_clusters'] = args.percent_clusters
    config_params['percent_clusters_l2'] = args.percent_clusters_l2
    config_params['percentile'] = args.percentile
    config_params['percentile_lower'] = args.percentile_lower
    config_params['obs_window'] = args.obs_window
    config_params['seed'] = args.seed
    config_params['force_chat'] = args.force_chat
    config_params['rope_scaling'] = args.rope_scaling

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print('dataset: ', dataset)

        # update path to clusters here for each dataset
        config_params['path_to_clusters_cosine'] = args.path_to_clusters + f'{dataset}/'
        data = load_dataset('THUDM/LongBench', dataset, split='test')

        # construct savepath here
        if not args.use_centroids:
            savepath = f"pred/{model_name}_baseline"
        else:
            if args.hierarchical_lookup:
                savepath = f"pred/{model_name}_PC1_{args.percent_clusters}_PERC1_{args.percentile}_PC2_{args.percent_clusters_l2}_PERC2_{args.percentile_lower}_lookup"
            else:
                savepath = f"pred/{model_name}_PC{args.percent_clusters}_PERC{args.percentile}"

        # Hậu tố _lim<N>: điểm của N sample đầu KHÔNG so được với điểm của cả 500 sample.
        # Không tách thư mục thì một lượt smoke test sẽ lặng lẽ ghi đè kết quả đầy đủ —
        # cùng loại lỗi với bug append #3 của Phase 0.
        if args.limit > 0:
            savepath = f"{savepath}_lim{args.limit}"

        # _run<TAG> tach cac lan lap cua cung mot cau hinh. eval.py phai nhan cung --run_tag.
        if args.run_tag:
            savepath = f"{savepath}_run{args.run_tag}"

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        out_path = savepath + f"/{dataset}.jsonl"
        done_idx = set()
        if os.path.exists(out_path):
            if args.overwrite and args.resume:
                raise SystemExit('[ERROR] --overwrite va --resume loai tru nhau.')
            if args.overwrite:
                print(f'[overwrite] xoa ket qua cu: {out_path}')
                os.remove(out_path)
            elif args.resume:
                with open(out_path, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            done_idx.add(json.loads(line)['dataidx'])
                        except (ValueError, KeyError):
                            pass  # dong cuoi co the bi cat cut khi bi kill giua chung
                print(f'[resume] {out_path}: da co {len(done_idx)} dataidx -> bo qua, chay tiep')
            else:
                print(f'[CANH BAO] {out_path} da ton tai va pred.py ghi o che do append.')
                print(f'           Ket qua se bi nhan doi va eval.py se ra so sai.')
                print(f'           Dung --overwrite / --resume hoac xoa file truoc khi chay lai.')

        prompt_format = dataset2prompt[dataset]
        prompt_only_format = dataset2prompt[dataset + '_prompt']
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        for i in range(len(data_all)):
            data_all[i]['different_prefix_index'] = i

        # Cắt SAU khi gán different_prefix_index -> index vẫn là 0..N-1, khớp đúng tên
        # file centroid mà offline_clustering.py --limit N đã sinh ra.
        if args.limit > 0:
            data_all = data_all[:args.limit]
            print(f'[limit] chi chay {len(data_all)}/{len(data)} sample dau -> {savepath}')

        # --resume: bo qua nhung mau da co trong file. Loc SAU khi cat --limit va SAU khi
        # gan different_prefix_index -> anh xa index<->ten file centroid van dung.
        if done_idx:
            before = len(data_all)
            data_all = [d for d in data_all if d['different_prefix_index'] not in done_idx]
            print(f'[resume] con {len(data_all)}/{before} mau chua chay')
            if not data_all:
                print('[resume] tat ca mau da xong, khong con gi de chay.')
                continue

        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, prompt_only_format, dataset, device, model_name, model2path, out_path, config_params))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Worker chet vi exception thi p.join() van tra ve binh thuong va tien trinh cha
        # thoat voi ma 0. Hau qua day du: `set -e` cua script khong bat duoc, eval.py cham
        # diem tren so mau thieu, va result.json ghi ra mot con so trong nhu that.
        # Xay ra that ngay 27/8: mot file centroid hong lam worker chet o mau 226/500,
        # ca pipeline van chay het va bao Sq-70% = 57.28 (thuc te la diem cua 226 mau).
        failed = [(rank, p.exitcode) for rank, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            raise SystemExit(
                "[ERROR] %d/%d worker that bai (rank, exitcode): %s\n"
                "        Prediction trong %s KHONG day du -> KHONG duoc cham diem.\n"
                "        Xem traceback cua worker o phia tren de biet nguyen nhan."
                % (len(failed), len(processes), failed, out_path))
