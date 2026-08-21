"""
判别力检验: 小模型中间层 hidden state 是否区分"同义 / 随机"?
对每个参考句, 对比其在各层(1/4/8/12/16/23)的 pooled 表征与:
  (a) 人工同义改写   (期望 cos 高)
  (b) 随机无关句     (期望 cos 低)
  (c) 贪心搜索输出   (期望如何? 检验前提)
若 (a)>>(b) 则 hidden 距离是含义的弱信号(相关但不充分); 若 (a)≈(b) 则完全无判别力。
"""
from __future__ import annotations
import argparse, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUT = "../data/disc.txt"   # 相对本脚本 (scripts/); 论文图 fig_discriminative 读取该文件

PAIRS = [
    ("The quick brown fox jumps over the lazy dog.",
     "The fast orange fox leaps over the sleepy hound."),
    ("The committee approved the new budget proposal this morning.",
     "The board ratified the fresh spending plan earlier today."),
    ("She walked slowly through the old town under the rain.",
     "She strolled gently across the ancient city in the drizzle."),
]
# 每句配 3 条随机无关句
RANDOM = [
    "Quantum computing offers new possibilities for cryptography research.",
    "The recipe requires three eggs and a cup of fresh milk.",
    "Electric vehicles are changing the landscape of urban transportation.",
    "The orchestra performed a symphony by a famous composer.",
    "Recent studies link regular exercise to better sleep quality.",
    "Solar panels convert sunlight into usable electrical energy.",
    "The museum displays artifacts from the ancient civilization.",
    "Autonomous drones deliver packages in remote mountainous regions.",
    "A balanced diet is essential for maintaining good health.",
]


def pool(h, mask, mode="mean"):
    if mode == "mean":
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return h[:, 0]


@torch.no_grad()
def cos_at_layers(model, tok, text, layers):
    ids = tok(text, return_tensors="pt")["input_ids"].to("cuda")
    out = model(ids, output_hidden_states=True)
    mask = (ids != tok.pad_token_id).to(out.hidden_states[0].dtype)
    return [pool(out.hidden_states[l], mask) for l in layers], ids


def main():
    ap = argparse.ArgumentParser(description="判别力检验: 小模型中间层 hidden state 是否区分 同义/随机")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="结果输出文件 (格式: layer syn-vs-syn syn-vs-rnd gap); 默认 %(default)s")
    args = ap.parse_args()

    layers = [1, 4, 8, 12, 16, 23]
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results = {l: [] for l in layers}
    for s1, s2 in PAIRS:
        h1, _ = cos_at_layers(model, tok, s1, layers)   # list per layer [1,D]
        h2, _ = cos_at_layers(model, tok, s2, layers)
        for i, l in enumerate(layers):
            results[l].append(F.cosine_similarity(h1[i], h2[i]).item())

    rows = [f"{'layer':>5} {'syn-vs-syn':>11} {'syn-vs-rnd':>11} {'gap':>7}", "-" * 40]
    for l in layers:
        syn = sum(results[l]) / len(results[l])
        rnd = []
        for s1, _ in PAIRS:
            h1, _ = cos_at_layers(model, tok, s1, layers)
            l_ = layers.index(l)
            for r in RANDOM:
                hr, _ = cos_at_layers(model, tok, r, layers)
                rnd.append(F.cosine_similarity(h1[l_], hr[l_]).item())
        rnd_avg = sum(rnd) / len(rnd)
        rows.append(f"{l:>5} {syn:>11.3f} {rnd_avg:>11.3f} {syn-rnd_avg:>7.3f}")

    text = "\n".join(rows) + "\n"
    sys.stdout.write(text)
    out = args.out
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
