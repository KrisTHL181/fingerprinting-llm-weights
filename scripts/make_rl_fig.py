#!/usr/bin/env python3
"""
make_rl_fig.py — 附录图: RL(DPO) 对齐微调 vs SFT(LoRA) 的指纹灵敏度对比 (Qwen3.5-0.8B, L8)。

面板 (a): adapter 尺度 s → D_KL(clean || tuned), 全词表与 top-20 截断, log-y;
           对比 RL(DPO) 与论文附录 SFT(LoRA) 两条曲线 + 检测下限。
面板 (b): 两条曲线经 Qwen-L8 高斯噪声标定映射到"等价相对噪声尺度 r_eq"。

读 rl_test_results.json。输出 figures/fig_rl_sensitivity.pdf。
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
RES = HERE.parent / "data" / "rl_test_results.json"

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

C_RL = "#ff7f0e"     # 橙  RL(DPO)
C_RLT = "#ffbb78"    # 浅橙 top-20
C_SFT = "#1f77b4"    # 蓝  SFT(LoRA)
C_SFTT = "#9ecae1"   # 浅蓝 top-20
C_FLR = "#2ca02c"    # 绿  检测下限

# Qwen-L8 高斯噪声标定 (论文 Table 3)
CAL_R = np.array([0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 1.00])
CAL_D = np.array([0.0000, 0.0164, 0.0992, 0.5246, 3.2754, 9.2107, 12.9029, 14.7039])
FLOOR_D = 0.0164
FLOOR_R = 0.02


def map_to_r(D):
    return np.interp(D, CAL_D, CAL_R)


def main():
    r = json.load(open(RES))
    rows = r["dpo_sweep"]
    s = np.array([x["s"] for x in rows])
    d_rl = np.array([x["KL_full"] for x in rows])
    d_rlt = np.array([x["KL_trunc20"] for x in rows])

    # SFT-LoRA (论文附录 tab:lora)
    s_sft = np.array([0.00, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00])
    d_sft = np.array([0.0000, 0.0010, 0.0022, 0.0075, 0.0284, 0.0872, 0.1734])
    d_sftt = np.array([0.0000, 0.0008, 0.0018, 0.0061, 0.0234, 0.0735, 0.1477])

    # 检测下限交点
    s_thr_rl = float(np.interp(FLOOR_D, d_rl, s))
    s_thr_sft = float(np.interp(FLOOR_D, d_sft, s_sft))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=False)

    # 面板 (a): D_KL vs s (log-y)
    ax = axes[0]
    ax.plot(s, d_rl, "-o", ms=4, lw=1.3, color=C_RL, label=r"RL (DPO)", zorder=3)
    ax.plot(s, d_rlt, "-o", ms=3, lw=1.0, color=C_RLT, label=r"RL (DPO, top-20)", zorder=2)
    ax.plot(s_sft, d_sft, "-s", ms=4, lw=1.3, color=C_SFT, label=r"SFT (LoRA)", zorder=3)
    ax.plot(s_sft, d_sftt, "-s", ms=3, lw=1.0, color=C_SFTT, label=r"SFT (LoRA, top-20)", zorder=2)
    ax.axhline(FLOOR_D, color=C_FLR, ls=":", lw=1.0)
    ax.text(0.03, FLOOR_D * 1.5, r"detection floor ($r$=0.02)", color=C_FLR, fontsize=7)
    ax.axvline(s_thr_rl, color=C_RL, ls=":", lw=0.8)
    ax.text(s_thr_rl + 0.02, 6e-2, rf"$s\!\approx\!{s_thr_rl:.2f}$", color=C_RL, fontsize=7)
    ax.set_xlabel(r"adapter scale $s$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}$")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(5e-4, 2e-1)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5, which="both")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(r"(a) DPO vs.\ SFT sensitivity")

    # 面板 (b): r_eq vs s
    ax = axes[1]
    ax.plot(s, map_to_r(d_rl), "-o", ms=4, lw=1.3, color=C_RL, label=r"$r_{\mathrm{eq}}$ RL (DPO)", zorder=3)
    ax.plot(s_sft, map_to_r(d_sft), "-s", ms=4, lw=1.3, color=C_SFT, label=r"$r_{\mathrm{eq}}$ SFT (LoRA)", zorder=3)
    ax.axhline(FLOOR_R, color=C_FLR, ls=":", lw=1.0)
    ax.text(0.05, FLOOR_R + 0.0015, r"$r$=0.02 (smallest calibrated scale)", color=C_FLR, fontsize=7)
    ax.axvline(s_thr_rl, color=C_RL, ls=":", lw=0.8)
    ax.set_xlabel(r"adapter scale $s$")
    ax.set_ylabel(r"equivalent noise scale $r_{\mathrm{eq}}$")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.07)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(r"(b) Fisher-equivalent perturbation")

    fig.tight_layout()
    fig.savefig(OUT / "fig_rl_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_rl_sensitivity.pdf'}")
    print(f"RL full adapter D_KL={d_rl[-1]:.4f} r_eq={map_to_r(d_rl[-1]):.3f}  thr_s={s_thr_rl:.3f}")
    print(f"SFT full adapter D_KL={d_sft[-1]:.4f} r_eq={map_to_r(d_sft[-1]):.3f}  thr_s={s_thr_sft:.3f}")


if __name__ == "__main__":
    main()
