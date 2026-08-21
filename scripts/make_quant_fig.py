#!/usr/bin/env python3
"""
make_quant_fig.py — 附录图: 指纹对 weight-only 量化的灵敏度 (Qwen-0.8B, L8)。

面板 (a): 位宽 bits → D_KL(clean || quantized), 全词表与 top-20 截断, log-y。
面板 (b): 量化层比例 frac @ int4 → D_KL, 连续单调曲线, 标出检测下限(r=0.02)。
右侧标注跨模型族 int4/int8 单点(Llama / Gemma), 展示跨模型族一致性。

读 quant_test_results.json。输出 figures/fig_quant_sensitivity.pdf。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
# 结果 json: 统一在根 data/ 下
RES = HERE.parent / "data" / "quant_test_results.json"

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "stixsans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
})

C_FULL = "#1f77b4"     # 蓝  全词表 KL
C_TRUNC = "#ff7f0e"    # 橙  top-20 截断 KL
C_FLR = "#2ca02c"      # 绿  检测下限
C_LL = "#9467bd"       # 紫  Llama
C_GM = "#8c564b"       # 棕  Gemma

# Qwen-L8 高斯噪声标定 (论文 Table 3): r -> D_KL_full
CAL_R = np.array([0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 1.00])
CAL_D = np.array([0.0000, 0.0164, 0.0992, 0.5246, 3.2754, 9.2107, 12.9029, 14.7039])
FLOOR_D = 0.0164       # r=0.02 处 D_KL (标定最小档)


def map_to_r(D):
    return np.interp(D, CAL_D, CAL_R)


def main():
    r = json.load(open(RES))
    q = r["qwen"]

    bits = np.array([row["bits"] for row in q["bit_sweep"]])
    d_full = np.array([row["KL_full"] for row in q["bit_sweep"]])
    d_trunc = np.array([row["KL_trunc20"] for row in q["bit_sweep"]])

    frac = np.array([row["frac"] for row in q["frac_sweep"]])
    f_full = np.array([row["KL_full"] for row in q["frac_sweep"]])
    f_trunc = np.array([row["KL_trunc20"] for row in q["frac_sweep"]])

    # 跨模型族 int4/int8 点
    cross = {c["tag"]: c["cross_points"] for c in r.get("cross_models", [])}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=False)

    # 面板 (a): 位宽扫描
    ax = axes[0]
    ax.plot(bits, d_full, "-o", ms=4, lw=1.3, color=C_FULL, label=r"$D_{\mathrm{KL}}$ (full)", zorder=3)
    ax.plot(bits, d_trunc, "-s", ms=4, lw=1.3, color=C_TRUNC, label=r"$D_{\mathrm{KL}}$ (top-20)", zorder=3)
    ax.axhline(FLOOR_D, color=C_FLR, ls=":", lw=1.0)
    ax.text(7.1, FLOOR_D * 1.4, r"detection floor ($r$=0.02)", color=C_FLR, fontsize=7, ha="right")
    # 跨模型族 int8/int4
    for tag, c, off in [("llama", C_LL, 0.0), ("gemma", C_GM, 0.0)]:
        for row in cross.get(tag, []):
            b, df = row["bits"], row["KL_full"]
            ax.plot(b, df, "D", ms=5, color=c, zorder=4)
    ax.annotate("Llama/Gemma int8/int4", xy=(0.97, 0.05), xycoords="axes fraction",
                fontsize=7, color="#555", ha="right")
    ax.set_xlabel(r"weight-only quantization bit-width $k$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}$")
    ax.set_yscale("log")
    ax.set_xlim(2, 8)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5, which="both")
    ax.legend(loc="lower left")
    ax.set_title(r"(a) bit-width sensitivity (Qwen, L8)")

    # 面板 (b): 层比例扫描 @ int4
    ax = axes[1]
    ax.plot(frac, f_full, "-o", ms=4, lw=1.3, color=C_FULL, label=r"$D_{\mathrm{KL}}$ (full)", zorder=3)
    ax.plot(frac, f_trunc, "-s", ms=4, lw=1.3, color=C_TRUNC, label=r"$D_{\mathrm{KL}}$ (top-20)", zorder=3)
    ax.axhline(FLOOR_D, color=C_FLR, ls=":", lw=1.0)
    ax.text(0.05, FLOOR_D * 1.5, r"detection floor ($r$=0.02)", color=C_FLR, fontsize=7)
    # 首次超过下限的 frac
    idx = np.where(f_full > FLOOR_D)[0]
    if len(idx):
        fr_thr = float(frac[idx[0]])
        ax.axvline(fr_thr, color=C_FLR, ls=":", lw=0.9)
        ax.text(fr_thr + 0.02, f_full.max() * 0.3, rf"$f\!\approx\!{fr_thr:.2f}$",
                color=C_FLR, fontsize=7)
    ax.set_xlabel(r"fraction of layers quantized $f$ (int4)")
    ax.set_ylabel(r"$D_{\mathrm{KL}}$")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5, which="both")
    ax.legend(loc="lower right")
    ax.set_title(r"(b) layer-fraction sensitivity (Qwen, L8, int4)")

    fig.tight_layout()
    fig.savefig(OUT / "fig_quant_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_quant_sensitivity.pdf'}")
    print(f"bits D_KL(full): " + ", ".join(f"{b}:{d:.3f}" for b, d in zip(bits, d_full)))
    print(f"frac D_KL(full): " + ", ".join(f"{f:.2f}:{d:.3f}" for f, d in zip(frac, f_full)))
    print("r_eq(bits): " + ", ".join(f"{b}:{map_to_r(d):.2f}" for b, d in zip(bits, d_full)))


if __name__ == "__main__":
    main()
