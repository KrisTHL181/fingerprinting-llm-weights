"""
跨模型迁移验证: 小模型生成的对齐序列, 在目标模型上是否仍保持表征对齐?
逻辑: 若 h_target(S') ≈ h_target(S), 则同义句迁移成立(共享子词表保证可被目标 tokenizer 切分)。
"""
from __future__ import annotations
import argparse, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"


def pool_hidden(h: torch.Tensor, mask: torch.Tensor, mode="mean"):
    if mode == "mean":
        return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return h[:, 0]


@torch.no_grad()
def hidden_at_layer(model, tok, text_ids, layer, pool="mean"):
    out = model(text_ids, output_hidden_states=True)
    h = out.hidden_states[layer]
    mask = (text_ids != tok.pad_token_id).to(h.dtype)
    return pool_hidden(h, mask, pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-frac", type=float, default=0.35,
                    help="中间层位置比例(小模型和目标模型各自取其第 N 层)")
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--pairs", nargs="+", default=[
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox leaps across the idle canine."),
        ("The committee approved the new budget proposal this morning.",
         "The committee passed the fresh spending plan early today."),
    ], action="append")
    args = ap.parse_args()
    # flatten pairs
    flat = [p for grp in args.pairs for p in grp]
    pairs = list(zip(flat[::2], flat[1::2]))

    print("loading small model...")
    small = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    stok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if stok.pad_token_id is None:
        stok.pad_token = stok.eos_token
    L_small = small.config.num_hidden_layers
    layer_small = max(1, min(L_small - 1, int(L_small * args.layer_frac)))
    print(f"small layers={L_small} -> layer {layer_small}")

    print("loading target model...")
    tgt = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    ttok = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True)
    if ttok.pad_token_id is None:
        ttok.pad_token = ttok.eos_token
    L_tgt = tgt.config.num_hidden_layers
    layer_tgt = max(1, min(L_tgt - 1, int(L_tgt * args.layer_frac)))
    print(f"target layers={L_tgt} -> layer {layer_tgt}")

    print(f"\n{'pair':<4} {'small_cos':>10} {'target_cos':>11}  check")
    print("-" * 50)
    for i, (s1, s2) in enumerate(pairs):
        i1 = stok(s1, return_tensors="pt")["input_ids"].to("cuda")
        i2 = stok(s2, return_tensors="pt")["input_ids"].to("cuda")
        # small model alignment (mask to min len)
        h1s = hidden_at_layer(small, stok, i1, layer_small)
        h2s = hidden_at_layer(small, stok, i2, layer_small)
        small_cos = F.cosine_similarity(h1s, h2s, dim=-1).item()

        # target model: tokenize with TARGET tokenizer (cross-model, len may differ)
        j1 = ttok(s1, return_tensors="pt")["input_ids"].to("cuda")
        j2 = ttok(s2, return_tensors="pt")["input_ids"].to("cuda")
        h1t = hidden_at_layer(tgt, ttok, j1, layer_tgt)
        h2t = hidden_at_layer(tgt, ttok, j2, layer_tgt)
        target_cos = F.cosine_similarity(h1t, h2t, dim=-1).item()

        ok = target_cos > small_cos
        print(f"{i:<4} {small_cos:>10.3f} {target_cos:>11.3f}  {'TRANSFER' if ok else 'NO-TRANSFER'}")


if __name__ == "__main__":
    main()
