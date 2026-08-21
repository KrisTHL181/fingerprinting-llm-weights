#!/usr/bin/env python3
"""
make_figures.py — 从 data/ 原始数据生成论文图 (Arial 字体, 矢量 PDF)。

输出到 figures/:
  fig_calibration.pdf   D_KL 随权重噪声尺度 r 的标定曲线 (Qwen L8/L16, Llama, Gemma)
  fig_discriminative.pdf  逐层 hidden-state 余弦判别 gap
  fig_spectrum.pdf      DeepSeek 指纹协方差谱(特征值衰减, 低秩)

用法: python3 make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Arial 设置
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

# 色盲友好配色
C_L8 = "#1f77b4"       # 蓝  Qwen L8
C_L16 = "#d62728"      # 红  Qwen L16
C_LL = "#ff7f0e"       # 橙  Llama-3.2-1B
C_GM = "#9467bd"       # 紫  Gemma-3-1b


# ---------------------------------------------------------------- 解析数据
def parse_sweep(path):
    """解析 sweep_*_v2.txt (列: r D_mean D_cov D_KL D_TV D_rank), 返回 (r, D_KL)。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    r, dkl = [], []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 6:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        r.append(vals[0]); dkl.append(vals[3])
    return np.array(r), np.array(dkl)


def parse_disc(path):
    """解析 disc.txt 里的 layer / eq-eq / eq-rnd / gap。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    layer, ss, sr = [], [], []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if vals[0] < 1 or vals[0] > 30:  # 层号应在 1..24
            continue
        layer.append(vals[0]); ss.append(vals[1]); sr.append(vals[2])
    return np.array(layer), np.array(ss), np.array(sr)


# ---------------------------------------------------------------- 数据
SERIES = [
    (parse_sweep(DATA / "sweep_qwen_l8_v2.txt"),  C_L8,  "Qwen-0.8B L8"),
    (parse_sweep(DATA / "sweep_qwen_l16_v2.txt"), C_L16, "Qwen-0.8B L16"),
    (parse_sweep(DATA / "sweep_llama_l8_v2.txt"), C_LL,  "Llama-3.2-1B L8"),
    (parse_sweep(DATA / "sweep_gemma_l8_v2.txt"), C_GM,  "Gemma-3-1B L8"),
]

layers, ss, sr = parse_disc(DATA / "disc.txt")
eig = np.array(json.loads((DATA / "ds_cov.json").read_text())["cov_eigvals"])


def quad_coef(r, d, rmax=0.1):
    """在小 r 区间拟合 D_KL ≈ c·r², 返回 c (最小二乘)。"""
    m = (r <= rmax) & (r > 0)
    return float(np.sum(d[m] * r[m] ** 2) / np.sum(r[m] ** 4))


# ---------------------------------------------------------------- 图 1: D_KL 标定
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

# (a) log-log: 小 r 段斜率 2 (费舍尔信息二次响应)
ax = axes[0]
rr = np.linspace(0.01, 1.0, 200)
for (r, d), c, lab in SERIES:
    m = r > 0
    ax.plot(r[m], d[m], "-o", ms=4, lw=1.3, color=c, label=lab, zorder=3)
    cc = quad_coef(r, d)
    ax.plot(rr, cc * rr ** 2, "--", lw=0.9, color=c, alpha=0.45, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"noise scale $r$")
ax.set_ylabel(r"$D_{\mathrm{KL}}$")
ax.set_xlim(0.01, 1.0)
ax.set_ylim(0.005, 60)
ax.grid(True, ls=":", lw=0.4, alpha=0.5, which="both")
ax.set_title(r"(a) log--log (small-$r$ slope $=2$)")

# (b) 线性轴: 单调增长 + 大 r 饱和
ax = axes[1]
for (r, d), c, lab in SERIES:
    ax.plot(r, d, "-o", ms=4, lw=1.3, color=c, label=lab, zorder=3)
ax.set_xlabel(r"noise scale $r$")
ax.set_ylabel(r"$D_{\mathrm{KL}}$")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, None)
ax.grid(True, ls=":", lw=0.4, alpha=0.5)
ax.set_title(r"(b) monotonic growth + saturation")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT / "fig_calibration.pdf", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- 图 2: 判别 gap
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.plot(layers, ss, "-o", ms=4, lw=1.3, color=C_L8, label="eq--eq")
ax.plot(layers, sr, "-s", ms=4, lw=1.3, color=C_L16, label="eq--rnd")
ax.fill_between(layers, sr, ss, color="#888888", alpha=0.18)
for x, hi, lo in zip(layers, ss, sr):
    ax.annotate(f"{hi-lo:.2f}", (x, (hi+lo)/2), ha="center", va="center",
                fontsize=6.5, color="#333333")
ax.set_xlabel("layer")
ax.set_ylabel("mean cosine similarity")
ax.set_xlim(0, 24)
ax.set_ylim(0.55, 1.0)
ax.grid(True, ls=":", lw=0.4, alpha=0.5)
ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(OUT / "fig_discriminative.pdf", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- 图 3: 谱
fig, ax = plt.subplots(figsize=(3.3, 2.6))
rank = np.arange(1, len(eig) + 1)
ax.plot(rank, eig, "-o", ms=4, lw=1.3, color=C_L8)
ax.axvline(3.5, color=C_L16, ls="--", lw=0.9)
ax.text(4.0, eig[0] * 0.55, "top-3 = 73%", color=C_L16, fontsize=7.5)
ax.set_yscale("log")
ax.set_xlabel("eigenvalue index")
ax.set_ylabel(r"eigenvalue $\lambda_i$")
ax.set_xlim(0.5, len(eig) + 0.5)
ax.grid(True, ls=":", lw=0.4, alpha=0.5, which="both")
fig.tight_layout()
fig.savefig(OUT / "fig_spectrum.pdf", bbox_inches="tight")
plt.close(fig)


print("已生成:")
for f in sorted(OUT.glob("*.pdf")):
    print(f"  {f.name}")
