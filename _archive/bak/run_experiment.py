"""
实验运行器: 中间层表征对齐的同义句生成
用法:
  python run_experiment.py --layer 8 --n-steps 25 --diversity 0.5 --sentences "s1" "s2"
"""
from __future__ import annotations
import argparse, time, json
import torch
from transformers import AutoTokenizer
from align import Aligner

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"


def build_shared_vocab(aligner: Aligner) -> set[str]:
    """小模型 tokenizer 与目标模型 tokenizer 的共享子词表。"""
    small = set(aligner.tok.get_vocab().keys())
    print(f"small model vocab: {len(small)}")
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True)
    tgt = set(tgt_tok.get_vocab().keys())
    print(f"target model vocab: {len(tgt)}")
    shared = small & tgt
    print(f"shared vocab: {len(shared)} ({len(shared)/len(small)*100:.1f}% of small)")
    return shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8, help="主中间层(1-based)")
    ap.add_argument("--layers", type=str, default=None,
                    help="多层对齐: 逗号分隔层列表,如 '4,8,12'(优先于 --layer)")
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--diversity", type=float, default=0.5)
    ap.add_argument("--diversity-grad", type=float, default=1.0)
    ap.add_argument("--min-changes", type=int, default=2)
    ap.add_argument("--method", choices=["greedy", "synonym"], default="greedy")
    ap.add_argument("--k-neighbors", type=int, default=300)
    ap.add_argument("--pool", type=str, default="mean")
    ap.add_argument("--no-shared-vocab", action="store_true", help="不限制共享词表(对照)")
    ap.add_argument("--sentences", nargs="+", default=[
        "The quick brown fox jumps over the lazy dog.",
        "The committee approved the new budget proposal this morning.",
        "She walked slowly through the old town under the rain.",
    ])
    args = ap.parse_args()

    layer_arg = [int(x) for x in args.layers.split(",")] if args.layers else args.layer
    aligner = Aligner(MODEL_NAME, layer=layer_arg, pool=args.pool)
    print(f"model layers: {aligner.model.config.num_hidden_layers}, hidden: {aligner.model.config.hidden_size}")

    if not args.no_shared_vocab:
        shared = build_shared_vocab(aligner)
        aligner.restrict_vocab(shared)
    else:
        aligner.restrict_vocab(None)

    results = []
    for s in args.sentences:
        print(f"\n=== 参考句: {s!r} ===")
        ids = aligner.tok(s, return_tensors="pt")["input_ids"].to("cuda")
        t0 = time.time()
        if args.method == "synonym":
            seq, meta = aligner.synonym_search(
                ids, n_steps=args.n_steps, top_k=args.top_k,
                k_neighbors=args.k_neighbors, min_changes=args.min_changes,
            )
        else:
            seq, meta = aligner.greedy_search(
                ids, n_steps=args.n_steps, top_k=args.top_k,
                diversity=args.diversity, diversity_grad=args.diversity_grad,
                min_changes=args.min_changes,
            )
        dt = time.time() - t0
        out = aligner.decode(seq)[0]
        src = aligner.decode(ids)[0]
        print(f"原句:   {src}")
        print(f"生成:   {out}")
        print(f"mse={meta['mse']:.4f} cos={meta['cos']:.3f} "
              f"changed={meta['changes']}/{ids.shape[1]} tokens | {dt:.1f}s")
        results.append({"src": src, "out": out, "mse": meta["mse"],
                        "cos": meta["cos"], "changes": meta["changes"],
                        "tokens": ids.shape[1]})

    print("\n===== SUMMARY =====")
    for r in results:
        print(f"{r['src']!r} -> {r['out']!r}  (mse={r['mse']:.4f} cos={r['cos']:.3f} "
              f"changed={r['changes']}/{r['tokens']})")


if __name__ == "__main__":
    main()
