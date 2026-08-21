#!/usr/bin/env python3
"""
make_lora_fig.py — 附录图: LoRA 微调 / 模型偷换 的指纹灵敏度 (Qwen3.5-0.8B, L8)。

面板 (a):  LoRA adapter 尺度 s → D_KL(clean || Qwen+LoRA), 全词表与 top-20 截断,
           二次拟合, 检测下限(r=0.02), 检出阈值 s≈0.29, 以及跨族偷换
           Qwen→Llama 的 D_KL=4.41 (log-y, 展示 ~25× 裕度)。
面板 (b):  LoRA 的 D_KL 经 Qwen-L8 高斯噪声标定映射到"等价相对噪声尺度 r_eq",
           标出 r=0.02 下限与 s=1 处 r_eq≈0.059。

用法: python3 make_lora_fig.py    (输出 figures/fig_lora_sensitivity.pdf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

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
C_SWAP = "#d62728"     # 红  跨族偷换
C_FLR = "#2ca02c"      # 绿  检测下限

# ------------------------------------------------------------- 数据
s = np.array([0.00, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00])
d_full = np.array([0.0000, 0.0010, 0.0022, 0.0075, 0.0284, 0.0872, 0.1734])
d_trunc = np.array([0.0000, 0.0008, 0.0018, 0.0061, 0.0234, 0.0735, 0.1477])

# Qwen-L8 高斯噪声标定 (论文 Table 3): r -> D_KL_full
CAL_R = np.array([0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 1.00])
CAL_D = np.array([0.0000, 0.0164, 0.0992, 0.5246, 3.2754, 9.2107, 12.9029, 14.7039])

FLOOR_D = 0.0164       # r=0.02 处的 D_KL(标定最小档)
FLOOR_R = 0.02
CROSS = 4.4059         # Qwen→Llama 跨族 D_KL(全词表/共享面)
CROSS_TRUNC = 4.1072
CROSS_TAG = 4.41

c_full = float(np.dot(s[1:]**2, d_full[1:]) / np.dot(s[1:]**2, s[1:]**2))  # ~0.174
ss = np.linspace(0, 1, 200)
fit = c_full * ss ** 2
s_thr = float(np.sqrt(FLOOR_D / c_full))   # 拟合交点 (≈0.307); 用插值≈0.29 标注


def map_to_r(D):
    return np.interp(D, CAL_D, CAL_R)


r_eq = map_to_r(d_full)

# ------------------------------------------------------------- 图
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=False)

# 面板 (a): D_KL vs s (log-y), 展示 LoRA 灵敏度 + 跨族裕度
ax = axes[0]
ax.plot(s, d_full, "-o", ms=4, lw=1.3, color=C_FULL, label=r"$D_{\mathrm{KL}}$ (full)", zorder=3)
ax.plot(s, d_trunc, "-s", ms=4, lw=1.3, color=C_TRUNC, label=r"$D_{\mathrm{KL}}$ (top-20)", zorder=3)
ax.plot(ss, fit, "--", lw=0.9, color=C_FULL, alpha=0.55, zorder=2,
        label=r"fit $\approx 0.17\,s^2$")
ax.axhline(FLOOR_D, color=C_FLR, ls=":", lw=1.0)
ax.text(0.72, FLOOR_D * 1.35, r"detection floor ($r$=0.02)", color=C_FLR, fontsize=7)
ax.axvline(s_thr, color=C_FLR, ls=":", lw=0.9)
ax.text(s_thr + 0.02, 1.6e-2, r"$s\!\approx\!0.3$", color=C_FLR, fontsize=7, rotation=90)
# 跨族偷换线 (数量级更高)
ax.axhline(CROSS, color=C_SWAP, ls="--", lw=1.1)
ax.text(0.03, CROSS * 1.02, rf"Qwen$\to$Llama swap: $D_{{\mathrm{{KL}}}}$={CROSS_TAG}"
        rf" ($\times${CROSS/0.1734:.0f})", color=C_SWAP, fontsize=7, va="bottom")
ax.annotate("", xy=(0.95, CROSS), xytext=(0.95, 0.1734),
            arrowprops=dict(arrowstyle="<->", color=C_SWAP, lw=0.8))
ax.set_xlabel(r"adapter scale $s$  ($W_s = W_0 + s\,\Delta W$)")
ax.set_ylabel(r"$D_{\mathrm{KL}}$")
ax.set_yscale("log")
ax.set_xlim(0, 1.0)
ax.set_ylim(0.5e-3, 30)
ax.grid(True, ls=":", lw=0.4, alpha=0.5, which="both")
ax.legend(loc="lower right")
ax.set_title(r"(a) LoRA sensitivity & swap margin")

# 面板 (b): 等价噪声尺度 r_eq(s)
ax = axes[1]
ax.plot(s, r_eq, "-o", ms=4, lw=1.3, color=C_FULL, label=r"$r_{\mathrm{eq}}$ (full)", zorder=3)
ax.plot(s, map_to_r(d_trunc), "-s", ms=4, lw=1.3, color=C_TRUNC, label=r"$r_{\mathrm{eq}}$ (top-20)", zorder=3)
ax.axhline(FLOOR_R, color=C_FLR, ls=":", lw=1.0)
ax.text(0.05, FLOOR_R + 0.0012, r"$r$=0.02 (smallest calibrated scale)", color=C_FLR, fontsize=7)
ax.axvline(s_thr, color=C_FLR, ls=":", lw=0.9)
ax.text(s_thr + 0.01, 0.028, r"$s\!\approx\!0.3$", color=C_FLR, fontsize=7)
ax.annotate(r"full adapter: $r_{\mathrm{eq}}\!\approx\!0.06$",
            xy=(1.0, r_eq[-1]), xytext=(0.5, 0.072),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7.5)
ax.set_xlabel(r"adapter scale $s$")
ax.set_ylabel(r"equivalent noise scale $r_{\mathrm{eq}}$")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.085)
ax.grid(True, ls=":", lw=0.4, alpha=0.5)
ax.legend(loc="upper left")
ax.set_title(r"(b) Fisher-equivalent perturbation")

fig.tight_layout()
fig.savefig(OUT / "fig_lora_sensitivity.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_lora_sensitivity.pdf'}")
print(f"c_full={c_full:.4f}  s_thr(fit)={s_thr:.3f}  r_eq(s=1)={r_eq[-1]:.3f}")
