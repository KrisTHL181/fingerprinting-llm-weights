#!/usr/bin/env python3
"""
ablate_concentration.py — What representation-equivalence actually buys.

If representation-equivalence is NOT about raw sensitivity (the ablate_repeq.py
result), its residual, testable value should be that it produces a fingerprint
that is (a) concentrated / low-rank and (b) transferable across models. Random
sentences, by contrast, should give a diffuse (high-rank) spectrum that does not
cohere across models.

This script synthesizes three probe conditions on the SOURCE model (Qwen L8)
within the Qwen <-> Llama shared subword vocab, then reads each condition on BOTH
Qwen and a target open-weight model (Llama-3.2-1B), decoding each probe to text
and re-encoding on the target tokenizer (matching the paper's cross-model
re-tokenization of Sec 5.3).

Metrics per (model x condition), on the per-probe next-token log-prob vectors
restricted to the union top-k token set:
  effrank : participation ratio (sum lambda)^2 / sum(lambda^2)  -- low = concentrated
  top3    : fraction of trace in the top-3 eigenvalues
  var90   : # eigenvalues to reach 90% of the trace
  entropy : mean per-probe next-token entropy (full vocab)
"""
from __future__ import annotations
import argparse, gc, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from align import Aligner
SOURCE = "Qwen/Qwen3.5-0.8B"
TARGET_MAP = {
    "llama": "unsloth/Llama-3.2-1B",
    "gemma": "unsloth/gemma-3-1b-pt",
}
REFERENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The committee approved the new budget proposal this morning.",
    "She walked slowly through the old town under the rain.",
]

def next_token_logp(model, input_ids):
    logits = model(input_ids).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)      # [Vfull]

def covariance_metrics(X):
    """X: [m, |T|] per-probe log-prob vectors. Return (effrank, top3, var90, trace)."""
    m = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    gram = (Xc @ Xc.T) / max(m - 1, 1)
    eig = torch.linalg.eigvalsh(gram)
    eig = torch.flip(torch.clamp(eig, min=0.0), dims=(0,))
    trace = eig.sum().item()
    if trace <= 0:
        return (0.0, 0.0, 0, trace)
    effrank = (trace ** 2) / (eig.pow(2).sum()).clamp_min(1e-12).item()
    top3 = eig[:3].sum().item() / trace
    cs = torch.cumsum(eig, 0)
    var90 = int((cs >= 0.90 * trace).nonzero(as_tuple=False).flatten()[0].item()) + 1
    return (effrank, top3, var90, trace)

def mean_entropy(model, probes):
    H = 0.0
    for p in probes:
        logp = next_token_logp(model, p)
        H += -(logp.exp() * logp).sum().item()
    return H / len(probes)

def build_probe_sets(aligner, m_per_ref, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    shared_ids = aligner.shared_ids
    sets = {"eq": [], "rnd": [], "ref": []}
    for ref in REFERENCES:
        ids = aligner.tok(ref, return_tensors="pt")["input_ids"].to(aligner.vocab.device)
        T = ids.shape[1]
        syn = aligner.sample_near_synonyms(ids, M=m_per_ref, top_k=6, k_neighbors=200)
        syn = [s for s in syn if not torch.equal(s, ids)][:m_per_ref]
        sets["eq"].extend(syn)
        for _ in range(m_per_ref):
            sets["rnd"].append(shared_ids[torch.randint(0, len(shared_ids), (1, T), generator=rng, device="cuda")])
        sets["ref"].extend([ids.clone()] * m_per_ref)
    return sets

def read_on(model, src_tok, tgt_tok, probes):
    """Probes are token-id sequences on the SOURCE tokenizer. Re-encode for the
    target by decoding text on src_tok and encoding on tgt_tok (paper's Sec 5.3
    cross-model re-tokenization). When src_tok is tgt_tok this round-trips."""
    out = []
    for p in probes:
        text = src_tok.decode(p[0], skip_special_tokens=True)
        e = tgt_tok(text, return_tensors="pt")["input_ids"].to(model.device)
        out.append(e)
    return out

def measure(model, probes, top_k, mname, cond, rows):
    # single stacked pass over all probes; derive the union top-k token set and
    # the covariance from LP0 directly (no second forward). H is computed from a
    # chunked mean to avoid materializing [m, Vfull] float copies that fragment
    # the hybrid Qwen kernel's memory on the 12 GB card.
    # explicit loop + sync per forward: a list comprehension stack lets the
    # hybrid Qwen linear-attention kernel's forwards overlap and blows the 12 GB
    # card (verified: the loop stays at ~1.6 GB / 4.2 GB peak, the stack OOMs).
    LP = []
    for p in probes:
        LP.append(next_token_logp(model, p))
        torch.cuda.synchronize()
    LP0 = torch.stack(LP)                                                   # [m, Vfull]
    Tset = set()
    for row in LP0.topk(top_k, dim=-1).indices:
        Tset.update(row.tolist())
    T_ids = torch.tensor(sorted(Tset), device=model.device)
    X = LP0[:, T_ids]                                                       # [m,|T|]
    eff, top3, var90, tr = covariance_metrics(X)
    hsum = 0.0
    for row in LP0:
        lp = row.to(torch.float32)
        hsum -= (lp.exp() * lp).sum().item()
    H = hsum / LP0.shape[0]
    line = f"{mname:>5} {cond:>3} {X.shape[1]:>4} {eff:>8.2f} {top3:>6.3f} {var90:>5} {H:>8.3f}"
    print(line, flush=True)
    rows.append(line)
    del LP0, X
    gc.collect(); torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "peer"], required=True,
                    help="qwen = synthesize + measure on source; peer = measure on the --target model (reads probe texts)")
    ap.add_argument("--target", choices=["llama", "gemma"], default="llama",
                    help="cross-model peer: restricts the shared vocab at synthesis and is read at peer time")
    ap.add_argument("--source-layer", type=int, default=8)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../data/ablate_concentration.txt")
    args = ap.parse_args()

    peer = TARGET_MAP[args.target]
    rows = []
    header = "model cond ndist  effrank  top3  var90  entropy"
    print("\n" + header, flush=True)
    rows.append(header)

    if args.model == "qwen":
        # synthesize probes and measure on the source model, dumping probe texts
        aligner = Aligner(SOURCE, layer=args.source_layer)
        aligner.model.eval()
        tgt_tok = AutoTokenizer.from_pretrained(peer, trust_remote_code=True)
        shared = set(aligner.tok.get_vocab().keys()) & set(tgt_tok.get_vocab().keys())
        aligner.restrict_vocab(shared)
        print(f"Qwen<->{args.target} shared vocab: {len(aligner.shared_ids)}", flush=True)
        sets = build_probe_sets(aligner, args.m, args.seed)
        src_tok = aligner.tok
        texts = {cond: [src_tok.decode(p[0], skip_special_tokens=True) for p in sets[cond]]
                 for cond in ["eq", "rnd", "ref"]}
        with open(args.out + ".probes.json", "w") as f:
            json.dump(texts, f)
        print(f"dumped probe texts -> {args.out}.probes.json", flush=True)
        for cond in ["eq", "rnd", "ref"]:
            measure(aligner.model, read_on(aligner.model, src_tok, src_tok, sets[cond]),
                    args.top_k, "qwen", cond, rows)

    else:  # peer: read dumped probe texts on the target model
        tgt_tok = AutoTokenizer.from_pretrained(peer, trust_remote_code=True)
        tgt_model = AutoModelForCausalLM.from_pretrained(peer, dtype=torch.bfloat16, device_map="cuda")
        tgt_model.eval()
        with open(args.out + ".probes.json") as f:
            texts = json.load(f)
        for cond in ["eq", "rnd", "ref"]:
            probe_ids = [tgt_tok(t, return_tensors="pt")["input_ids"].to("cuda") for t in texts[cond]]
            measure(tgt_model, probe_ids, args.top_k, args.target, cond, rows)

    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nwrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
