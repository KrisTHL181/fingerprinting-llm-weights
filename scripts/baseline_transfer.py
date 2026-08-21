#!/usr/bin/env python3
"""
baseline_transfer.py — transferability / low-rank check contrasting the baseline's RANDOM
probes vs the paper's representation-aligned probes, on the source model and a target model.

Mirrors paper Table 5 (ablate_conc): read each probe condition on the source (Qwen, layer 8)
and re-encode on a target (Llama-3.2-1B), and measure the spectral shape of the per-probe
next-token log-probability covariance:
  effrank : participation ratio (sum lam)^2 / sum lam^2
  top3    : fraction of trace in top-3 eigenvalues
  var90   : number of eigenvalues to reach 90% of the trace
The baseline (random probes) should be diffuse on the target (high effrank), while aligned
probes stay low-rank, which is the paper's load-bearing transferability claim.

Usage:
  python baseline_transfer.py --source Qwen/Qwen3.5-0.8B --target unsloth/Llama-3.2-1B \
      --layer 8 --out baseline_transfer.txt
"""
from __future__ import annotations
import argparse
import torch
from transformers import AutoTokenizer
from align import Aligner

TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]
RND_SEED = 0


@torch.no_grad()
def next_token_logp(model, input_ids):
    return torch.log_softmax(model(input_ids).logits[0, -1].float(), dim=-1)


@torch.no_grad()
def random_probes(aligner, ref_ids, n_per_ref, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = aligner.shared_ids
    out = []
    for ref in ref_ids:
        T = ref.shape[1]
        for _ in range(n_per_ref):
            idx = torch.randint(0, ids.numel(), (T,), generator=g, device=ids.device)
            out.append(ids[idx].unsqueeze(0))
    return out


def spectral(X):
    Xc = X - X.mean(0, keepdim=True)
    if Xc.shape[0] < 2 or Xc.shape[1] < 2:
        return None
    lam = torch.linalg.svdvals(Xc) ** 2
    lam = lam[lam > 1e-12]
    if lam.numel() == 0:
        return None
    tr = lam.sum().item()
    eff = (lam.sum() ** 2 / (lam ** 2).sum()).item()
    top3 = lam[:3].sum().item() / tr
    cum = torch.cumsum(lam, 0)
    var90 = int((cum < 0.90 * lam.sum()).sum().item()) + 1
    return eff, top3, var90, tr


def read_condition(model, tok, probes_text, vsh_ids):
    """Encode probe texts on a model's tokenizer, read next-token logp restricted to the
    intersection of vsh_ids with that tokenizer's vocab (surface-shared tokens)."""
    common = [i for i, t in enumerate(tok.get_vocab().keys()) if t in vsh_ids]
    # map probe tokens: keep only probes fully tokenizable in target vocab
    rows = []
    for s in probes_text:
        ids = tok(s, return_tensors="pt")["input_ids"].to(model.device)
        lp = next_token_logp(model, ids)
        rows.append(lp[common])
    if not rows:
        return None
    return torch.stack(rows)   # [n, |common|]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--target", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--m-rnd", type=int, default=20)
    ap.add_argument("--out", default="baseline_transfer.txt")
    args = ap.parse_args()

    # source: build rnd + alg probe TEXT
    src = Aligner(args.source, layer=args.layer)
    src.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True).get_vocab().keys())
    shared = set(src.tok.get_vocab().keys()) & dt
    src.restrict_vocab(shared)
    ref_ids = [src.tok(r, return_tensors="pt")["input_ids"].to(src.vocab.device) for r in REFERENCES]

    alg_seqs = []
    for ids in ref_ids:
        alg_seqs.extend(src.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200))
    rnd_seqs = random_probes(src, ref_ids, args.m_rnd, seed=RND_SEED)
    alg_text = [src.tok.decode(s[0], skip_special_tokens=True) for s in alg_seqs]
    rnd_text = [src.tok.decode(s[0], skip_special_tokens=True) for s in rnd_seqs]

    # source covariance (on source shared vocab)
    X_alg_src = torch.stack([next_token_logp(src.model, s) for s in alg_seqs])[:, src.shared_ids]
    X_rnd_src = torch.stack([next_token_logp(src.model, s) for s in rnd_seqs])[:, src.shared_ids]

    # target
    tgt = Aligner(args.target, layer=args.layer)
    tgt.model.eval()
    tgt_common = set(tgt.tok.get_vocab().keys()) & dt
    X_alg_tgt = read_condition(tgt.model, tgt.tok, alg_text, tgt_common)
    X_rnd_tgt = read_condition(tgt.model, tgt.tok, rnd_text, tgt_common)

    def fmt(tag, X):
        s = spectral(X)
        if s is None:
            print(f"{tag}: n/a (too small)")
            return
        eff, top3, var90, tr = s
        print(f"{tag}: effrank={eff:.2f} top3={top3:.3f} var90={var90} trace={tr:.1f}")

    print("SOURCE", flush=True)
    fmt("  alg", X_alg_src); fmt("  rnd", X_rnd_src)
    print("TARGET", flush=True)
    fmt("  alg", X_alg_tgt); fmt("  rnd", X_rnd_tgt)

    with open(args.out, "w") as f:
        f.write("model condition effrank top3 var90 trace\n")
        for tag, X in [("source", X_alg_src), ("source", X_rnd_src), ("target", X_alg_tgt), ("target", X_rnd_tgt)]:
            s = spectral(X)
            if s is None:
                continue
            f.write(f"{tag} {s[0]:.3f} {s[1]:.3f} {s[2]} {s[3]:.1f}\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
