#!/usr/bin/env python3
"""
baseline_swap.py — Cross-family (swap) detection reliability: aligned vs random probes.

For a wholesale model swap (Qwen -> Llama) the paper detects the change by the mean
per-probe next-token KL over surface-shared tokens. The reliability of that verdict
with a finite probe budget is set by the per-probe signal-to-noise (mean/std) of the
cross-model KL. Representation-aligned probes are low-rank on the target model, so
they should induce a concentrated, low-variance cross-model signal; random probes are
diffuse, so their cross-model KL should be noisier. We report, per probe condition,
the per-probe cross-model KL mean, std, and snr.

Protocol mirrors the paper's swap (S 5.4): probe TEXTS are re-tokenized on the target,
and the KL is computed over the surface-shared subword tokens.

Usage:
  python baseline_swap.py --source Qwen/Qwen3.5-0.8B --target unsloth/Llama-3.2-1B \
      --layer 8 --out baseline_swap_qwen2llama.txt
"""
from __future__ import annotations
import argparse
import torch
from transformers import AutoTokenizer
from align import Aligner

TARGET_API = "deepseek-ai/DeepSeek-V4-Flash-0731"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]
FLOOR = -30.0
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


def shared_token_indices(tok_src, tok_tgt):
    """Map surface-shared subword tokens to (src_ids, tgt_ids) tensors."""
    sv, tv = tok_src.get_vocab(), tok_tgt.get_vocab()
    common = set(sv.keys()) & set(tv.keys())
    s_ids, t_ids = [], []
    for k in common:
        s_ids.append(sv[k]); t_ids.append(tv[k])
    return torch.tensor(s_ids, dtype=torch.long), torch.tensor(t_ids, dtype=torch.long)


def truncated_kl_over(logp_src_vec, logp_tgt_vec, k=20):
    """KL over the shared-token vectors using union-of-top-k + renormalized tail."""
    top = torch.cat([torch.topk(logp_src_vec, k).indices, torch.topk(logp_tgt_vec, k).indices]).unique()
    q0 = logp_src_vec[top].exp().clamp_min(1e-30)
    qr = logp_tgt_vec[top].exp().clamp_min(1e-30)
    tail0 = (1.0 - q0.sum()).clamp_min(0.0)
    tailr = (1.0 - qr.sum()).clamp_min(0.0)
    kl = (q0 * (logp_src_vec[top] - logp_tgt_vec[top])).sum()
    if tail0 > 0:
        kl = kl + tail0 * (torch.log(tail0 + 1e-30) - torch.log(tailr + 1e-30))
    return kl


def report(tag, kls):
    kls = torch.tensor(kls)
    m = kls.mean().item(); s = kls.std().item()
    snr = (m / s) if s > 1e-12 else float("inf")
    print(f"{tag:>12}: mean={m:8.4f} std={s:8.4f} snr={snr:7.2f} n*(3sig)={((3*s/m)**2 if m>1e-12 else float('inf')):8.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--target", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--m-rnd", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="baseline_swap.txt")
    args = ap.parse_args()

    src = Aligner(args.source, layer=args.layer)
    src.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_API, trust_remote_code=True).get_vocab().keys())
    shared_src = set(src.tok.get_vocab().keys()) & dt
    src.restrict_vocab(shared_src)
    ref_ids = [src.tok(r, return_tensors="pt")["input_ids"].to(src.vocab.device) for r in REFERENCES]

    alg_seqs = []
    for ids in ref_ids:
        alg_seqs.extend(src.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200))
    rnd_seqs = random_probes(src, ref_ids, args.m_rnd, seed=RND_SEED)
    alg_text = [src.tok.decode(s[0], skip_special_tokens=True) for s in alg_seqs]
    rnd_text = [src.tok.decode(s[0], skip_special_tokens=True) for s in rnd_seqs]
    print(f"alg_probes={len(alg_text)} rnd_probes={len(rnd_text)}", flush=True)

    tgt = Aligner(args.target, layer=args.layer)
    tgt.model.eval()

    # source-side shared indices (relative to the API-shared Vsh already restricted)
    s_ids_shared, t_ids_shared = shared_token_indices(src.tok, tgt.tok)
    s_ids_shared = s_ids_shared.to(src.vocab.device)
    t_ids_shared = t_ids_shared.to(tgt.vocab.device)
    print(f"surface-shared tokens with target = {s_ids_shared.numel()}", flush=True)

    def per_probe_kl(texts):
        kls = []
        for s in texts:
            s_ids = src.tok(s, return_tensors="pt")["input_ids"].to(src.vocab.device)
            t_ids = tgt.tok(s, return_tensors="pt")["input_ids"].to(tgt.vocab.device)
            lp_src = next_token_logp(src.model, s_ids)[s_ids_shared]
            lp_tgt = next_token_logp(tgt.model, t_ids)[t_ids_shared]
            kls.append(truncated_kl_over(lp_src, lp_tgt, args.top_k).item())
        return kls

    kl_alg = per_probe_kl(alg_text)
    kl_rnd = per_probe_kl(rnd_text)
    report("alg", kl_alg)
    report("rnd", kl_rnd)

    with open(args.out, "w") as f:
        f.write("condition per_probe_kl\n")
        for v in kl_alg:
            f.write(f"alg {v:.6f}\n")
        for v in kl_rnd:
            f.write(f"rnd {v:.6f}\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
