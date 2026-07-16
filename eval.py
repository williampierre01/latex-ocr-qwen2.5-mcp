
"""
eval.py — Avaliação do Space Qwen2-VL LaTeX OCR contra um dataset público (im2latex).
 
Uso básico:
    python eval.py --space seu-usuario/nome-do-space
 
Exemplos:
    # 30 amostras impressas (default, rápido)
    python eval.py --space luminnon/qwen-latex-ocr
 
    # 20 amostras manuscritas (mais desafiador)
    python eval.py --space luminnon/qwen-latex-ocr --dataset-config human_handwrite --n 20
 
    # Space privado
    python eval.py --space luminnon/qwen-latex-ocr --hf-token hf_xxx
"""
import argparse
import csv
import json
import os
import tempfile
import time
from difflib import SequenceMatcher
from typing import Optional
 
from datasets import load_dataset
from gradio_client import Client
 
 
def normalize_latex(s: str) -> str:
    """Normaliza espaços e remove \\left/\\right para uma comparação mais justa.
 
    \\left e \\right são puramente cosméticos (ajuste de tamanho de delimitador) e
    modelos costumam ser inconsistentes em usá-los ou não, mesmo quando o LaTeX
    resultante é matematicamente idêntico. Removê-los evita penalizar o modelo
    por uma diferença que não afeta a equação renderizada.
    """
    s = s.strip()
    s = " ".join(s.split())
    s = s.replace("\\left", "").replace("\\right", "")
    return s
 
 
def edit_distance(a: str, b: str) -> int:
    """Distância de Levenshtein (DP clássico), sem dependência externa."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]
 
 
def cer(pred: str, gold: str) -> float:
    """Character Error Rate: edit_distance / len(gold). 0 = perfeito, >1 possível se pred for muito maior."""
    if len(gold) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gold) / len(gold)
 
 
def similarity(pred: str, gold: str) -> float:
    """Similaridade sequencial 0-1, mais tolerante a pequenas reordenações que o CER puro."""
    return SequenceMatcher(None, pred, gold).ratio()
 
 
def run_eval(
    space: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    n: int,
    hf_token: Optional[str],
    out_dir: str,
):
    print(f"Carregando dataset {dataset_name} ({dataset_config}) [{split}]...")
    ds = load_dataset(dataset_name, name=dataset_config, split=split)
    if n and n < len(ds):
        ds = ds.select(range(n))
 
    print(f"Conectando ao Space {space}...")
    client = Client(space, hf_token=hf_token)
 
    rows = []
    os.makedirs(out_dir, exist_ok=True)
 
    for i, sample in enumerate(ds):
        image = sample["image"]
        gold = normalize_latex(sample["text"])
 
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.convert("RGB").save(tmp.name)
                tmp_path = tmp.name
 
            t0 = time.time()
            error = None
            try:
                pred_raw = client.predict(image=tmp_path, api_name="/predict")
                pred = normalize_latex(pred_raw)
            except Exception as e:
                pred_raw = ""
                pred = ""
                error = str(e)
            latency = time.time() - t0
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
 
        row = {
            "idx": i,
            "gold": gold,
            "pred": pred,
            "pred_raw": pred_raw,
            "exact_match": int(pred == gold),
            "cer": round(cer(pred, gold), 4),
            "similarity": round(similarity(pred, gold), 4),
            "latency_s": round(latency, 2),
            "error": error,
        }
        rows.append(row)
        status = "OK " if row["exact_match"] else ("ERR" if error else "DIF")
        print(f"[{i + 1}/{len(ds)}] {status} cer={row['cer']:.3f} sim={row['similarity']:.3f} ({latency:.1f}s)")
 
    n_ok = sum(r["exact_match"] for r in rows)
    n_err = sum(1 for r in rows if r["error"])
    avg_cer = sum(r["cer"] for r in rows) / len(rows)
    avg_sim = sum(r["similarity"] for r in rows) / len(rows)
    avg_latency = sum(r["latency_s"] for r in rows) / len(rows)
 
    summary = {
        "total": len(rows),
        "exact_match": n_ok,
        "exact_match_rate": round(n_ok / len(rows), 4),
        "errors": n_err,
        "avg_cer": round(avg_cer, 4),
        "avg_similarity": round(avg_sim, 4),
        "avg_latency_s": round(avg_latency, 2),
    }
 
    csv_path = os.path.join(out_dir, "eval_results.csv")
    json_path = os.path.join(out_dir, "eval_summary.json")
 
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
 
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
 
    print("\n" + "=" * 40)
    print("RESUMO")
    print("=" * 40)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nDetalhes por amostra: {csv_path}")
    print(f"Resumo agregado:      {json_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o Space Qwen2-VL LaTeX OCR contra um dataset público.")
    parser.add_argument("--space", required=True, help="ID do Space no HF, ex: usuario/nome-do-space")
    parser.add_argument("--dataset", default="linxy/LaTeX_OCR", help="Dataset no HF Hub")
    parser.add_argument(
        "--dataset-config",
        default="small",
        help="Config do dataset: 'small' (impresso, 30 amostras teste), 'human_handwrite' (manuscrito), 'full' (~100k)",
    )
    parser.add_argument("--split", default="test", help="Split do dataset")
    parser.add_argument("--n", type=int, default=30, help="Número máximo de amostras a avaliar")
    parser.add_argument("--hf-token", default=None, help="Token HF, necessário se o Space for privado")
    parser.add_argument("--out-dir", default="./eval_output", help="Diretório de saída dos resultados")
    args = parser.parse_args()
 
    run_eval(
        space=args.space,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        n=args.n,
        hf_token=args.hf_token,
        out_dir=args.out_dir,
    )
 