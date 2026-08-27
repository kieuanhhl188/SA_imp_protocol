import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--use_centroids", action="store_true")
    parser.add_argument("--hierarchical_lookup", action="store_true")
    parser.add_argument("--percent_clusters", type=int, default=-1)
    parser.add_argument("--percent_clusters_l2", type=int, default=-1)
    parser.add_argument("--percentile", type=float, default=0.5)
    parser.add_argument("--percentile_lower", type=float, default=0.7)
    parser.add_argument("--run_tag", type=str, default="",
                        help="phai trung --run_tag da dung khi chay pred.py")
    parser.add_argument("--expect", type=int, default=-1,
                        help="so mau BAT BUOC phai co. Lech la dung, KHONG ghi result.json. "
                             "Khong co co nay thi eval.py cham bat cu thu gi no thay: "
                             "ngay 27/8 mot luot pred.py chet o mau 226/500 va eval.py van "
                             "ghi ra {'lcc': 57.28} — mot con so cua 226 mau, dat canh 54.83 "
                             "cua 500 mau trong cung mot bang.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="đọc thư mục có hậu tố _lim<N> do `pred.py --limit N` sinh ra. "
                             "Phải trùng đúng N đã dùng ở pred.py")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    """Tra ve (diem trung binh, list diem tung mau).

    Diem trung binh tinh y HET ban cu -- round(100*sum/len(predictions), 2) -- nen moi
    so da cong bo van tai lap bit-for-bit. Phan them la `per_sample`, can cho:
      - sai so chuan cua trung binh (quyet dinh tolerance +-0.3 co kha thi hay khong)
      - paired test giua cac method (Phase 5.5)
    Ca hai tinh duoc tu day, KHONG phai chay lai model.
    """
    per_sample = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        per_sample.append(score)
    if not predictions:
        raise SystemExit("[ERROR] %s: 0 prediction -- jsonl rong, khong tinh duoc diem." % dataset)
    return round(100 * sum(per_sample) / len(predictions), 2), per_sample

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    # Tên thư mục config dựng MỘT lần rồi dùng lại cho cả chỗ đọc và chỗ ghi.
    # Trước đây dựng hai lần bằng hai khối if giống hệt nhau -> sửa một bên quên bên
    # kia là kết quả rơi vào thư mục khác chỗ đọc.
    if not args.use_centroids:
        config_dir = f"{args.model}_baseline"
    elif args.hierarchical_lookup:
        config_dir = (f"{args.model}_PC1_{args.percent_clusters}_PERC1_{args.percentile}"
                      f"_PC2_{args.percent_clusters_l2}_PERC2_{args.percentile_lower}_lookup")
    else:
        config_dir = f"{args.model}_PC{args.percent_clusters}_PERC{args.percentile}"

    if args.limit > 0:
        config_dir = f"{config_dir}_lim{args.limit}"

    if args.run_tag:
        config_dir = f"{config_dir}_run{args.run_tag}"

    path = f"pred/{config_dir}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    detail = {}
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths, dataidxs = [], [], [], []
        dataset = filename.split('.')[0]
        # Phai khoi tao: `all_classes` chi duoc gan BEN TRONG vong doc file, nen jsonl
        # rong se lam dong scorer(...) nem NameError thay vi bao loi ro rang. Bug co san
        # tu ban goc -- cung ho loi "output rong" ma commit 3b84e71 xu ly o cho khac.
        all_classes = None

        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
                if "dataidx" in data:
                    dataidxs.append(data["dataidx"])
        score, per_sample = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score

        n = len(predictions)
        if args.expect > 0 and n != args.expect:
            raise SystemExit(
                "[ERROR] %s: co %d prediction nhung can %d.\n"
                "        Luot pred.py chua chay xong (hoac chay du roi bi mat file).\n"
                "        KHONG ghi result.json — diem cua tap con khong so duoc voi diem\n"
                "        cua ca tap, va de bi doc nham thanh ket qua that.\n"
                "        Chay lai: pred.py ... --overwrite" % (dataset, n, args.expect))
        n_empty = sum(1 for p in predictions if not p.strip())
        dup = (len(dataidxs) - len(set(dataidxs))) if dataidxs else None
        detail[dataset] = {
            "n_samples": n,
            "n_empty_pred": n_empty,
            "score": score,
            "dataidx_available": bool(dataidxs),
            "n_duplicate_dataidx": dup,
            # diem tung mau, KEY THEO dataidx neu co -> ghep duoc giua cac config
            "per_sample": (dict(zip([str(i) for i in dataidxs], per_sample))
                           if dataidxs else per_sample),
            "lengths": lengths,
        }
        msg = "  %s: %s   n=%d   pred rong=%d" % (dataset, score, n, n_empty)
        if dup:
            msg += "   dataidx trung lap=%d" % dup
        print(msg)
        if dup:
            print("  [CANH BAO] %s: %d dataidx trung lap -> jsonl da bi append nhieu "
                  "luot. Chay lai pred.py voi --overwrite." % (dataset, dup))
        if n_empty:
            print("  [CANH BAO] %s: %d/%d prediction rong." % (dataset, n_empty, n))

    out_path = f"{path}result.json"

    # result.json giu NGUYEN schema {dataset: score} -- check_gate.py doc truc tiep got[task].
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Chi tiet ra file RIENG. result.json chi co trung binh, nen 3 mau va 500 mau trong y
    # het nhau -- do la ly do file nay ton tai.
    with open(f"{path}result_detail.json", "w") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    print("")
    print("  result.json        -> %s" % out_path)
    print("  result_detail.json -> %sresult_detail.json" % path)
