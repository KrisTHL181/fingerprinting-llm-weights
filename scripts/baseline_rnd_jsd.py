#!/usr/bin/env python3
"""
baseline_rnd_jsd.py — "One Token Is Enough"-style distributional baseline vs the paper's
representation-aligned fingerprint, on the SAME r-sweep protocol as positive_control.py.

Baseline method (A): use RANDOM probes (matched-length token sequences drawn from the
shared subword vocab, NOT representation-aligned), and measure the mean per-probe
next-token divergence (KL and Jensen-Shannon) between clean and perturbed weights.
This is the published random-probe distributional fingerprint ("One Token Is Enough",
arXiv 2607.10252) instantiated on next-token distributions.

For direct contrast we also compute the paper's representation-aligned probe KL on the
same protocol. Protocol matches Table 3 / positive_control.py: single fixed Gaussian
direction (seed 0), logprobs clamped at -30.

Usage:
  python baseline_rnd_jsd.py --model Qwen/Qwen3.5-0.8B --layer 8 \
      --out baseline_rnd_qwen_l8.txt
"""
from __future__ import annotations
import argparse, copy
import torch
from transformers import AutoTokenizer
from align import Aligner

TARGET_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]
FLOOR = -30.0


@torch.no_grad()
def next_token_logp(model, input_ids):
    logits = model(input_ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)   # [Vfull]


@torch.no_grad()
def kl_div(lp0, lpr):
    p0 = lp0.exp().clamp_min(1e-30)
    return (p0 * (lp0.clamp(min=FLOOR) - lpr.clamp(min=FLOOR))).sum()


@torch.no_grad()
def js_div(lp0, lpr):
    p0 = lp0.exp().clamp_min(1e-30)
    pr = lpr.exp().clamp_min(1e-30)
    m = 0.5 * (p0 + pr)
    lm = torch.log(m).clamp(min=FLOOR)
    kl0m = (p0 * (lp0.clamp(min=FLOOR) - lm)).sum()
    klrm = (pr * (lpr.clamp(min=FLOOR) - lm)).sum()
    return 0.5 * (kl0m + klrm)


@torch.no_grad()
def perturb_weights(model, r, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m2.parameters():
            if p.requires_grad:
                std = p.data.std().clamp_min(1e-8)
                noise = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
                p.data.add_(noise * (r * std))
    return m2


@torch.no_grad()
def random_probes(aligner, ref_ids, n_per_ref, seed=0):
    """Matched-length random token sequences from the shared subword vocab Vsh.
    (The 'rnd' condition: uniform draws over the shared vocabulary, not aligned.)"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = aligner.shared_ids
    out = []
    for ref in ref_ids:
        T = ref.shape[1]
        for _ in range(n_per_ref):
            idx = torch.randint(0, ids.numel(), (T,), generator=g, device=ids.device)
            out.append(ids[idx].unsqueeze(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)           # probes per reference (alg)
    ap.add_argument("--m-rnd", type=int, default=20)       # probes per reference (rnd)
    ap.add_argument("--noise", default="0,0.02,0.05,0.10,0.20,0.40,0.60,1.00")
    ap.add_argument("--out", default="baseline_rnd.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--do-alg", action="store_true", help="also compute the paper's aligned-probe KL for contrast")
    args = ap.parse_args()

    aligner = Aligner(args.model, layer=args.layer)
    aligner.model.eval()
    dt = set(AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, trust_remote_code=True).get_vocab().keys())
    shared = set(aligner.tok.get_vocab().keys()) & dt
    aligner.restrict_vocab(shared)
    vsh_ids = aligner.shared_ids

    ref_ids = [aligner.tok(r, return_tensors="pt")["input_ids"].to(vsh_ids.device)
               for r in REFERENCES]

    # probe sets
    rnd_probes = random_probes(aligner, ref_ids, args.m_rnd, seed=args.seed)
    print(f"rnd_probes={len(rnd_probes)}  shared_vocab={vsh_ids.numel()}", flush=True)

    alg_probes = []
    if args.do_alg:
        for ids in ref_ids:
            alg_probes.extend(aligner.sample_near_synonyms(ids, M=args.m, top_k=6, k_neighbors=200))
        print(f"alg_probes={len(alg_probes)}", flush=True)

    # reference readouts
    def stats(probes):
        LPs = torch.stack([next_token_logp(aligner.model, p) for p in probes])  # [n,V]
        return LPs

    LP0_rnd = stats(rnd_probes)
    LP0_alg = stats(alg_probes) if args.do_alg else None

    noise_list = [float(x) for x in args.noise.split(",")]
    hdr = ["r", "JSD_rnd", "KL_rnd"]
    if args.do_alg:
        hdr += ["JSD_alg", "KL_alg"]
    print(" ".join(f"{h:>10}" for h in hdr), flush=True)
    rows = []
    for r in noise_list:
        m_r = perturb_weights(aligner.model, r, args.seed)
        # compute per-probe means over the whole probe set
        LP1_rnd = torch.stack([next_token_logp(m_r, p) for p in rnd_probes])
        js_rnd = sum(js_div(LP0_rnd[i], LP1_rnd[i]).item() for i in range(len(rnd_probes))) / len(rnd_probes)
        kl_rnd = sum(kl_div(LP0_rnd[i], LP1_rnd[i]).item() for i in range(len(rnd_probes))) / len(rnd_probes)
        row = [r, js_rnd, kl_rnd]
        if args.do_alg:
            LP1_alg = torch.stack([next_token_logp(m_r, p) for p in alg_probes])
            js_alg = sum(js_div(LP0_alg[i], LP1_alg[i]).item() for i in range(len(alg_probes))) / len(alg_probes)
            kl_alg = sum(kl_div(LP0_alg[i], LP1_alg[i]).item() for i in range(len(alg_probes))) / len(alg_probes)
            row += [js_alg, kl_alg]
        rows.append(row)
        print(" ".join(f"{x:>10.4f}" for x in row), flush=True)
        del m_r
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        f.write(" ".join(hdr) + "\n")
        for row in rows:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
