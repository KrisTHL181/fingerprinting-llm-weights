#!/usr/bin/env python3
"""
analyze_lora_results.py — 解释远程实验的 D_KL 结果, 回答"能否识别 / 多灵敏"。

读 lora_test_results.json, 并用论文 Qwen L8 高斯噪声标定曲线
(Table 3: r -> D_KL_full, 全词表, clamp -30) 把 LoRA 的 D_KL 映射回
"等价相对噪声尺度 r_eq" (r_eq = 使噪声标定 D_KL 与该 LoRA 尺度 D_KL 相同的 r)。

同时给出跨族 (Qwen->Llama) 的 D_KL, 与噪声标定饱和上限(≈熵上限)对比。
"""
import json, os, sys
import numpy as np

# Qwen L8 全词表 KL 标定 (论文 Table 3, r -> D_KL_full)
CALIB_R = np.array([0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 1.00])
CALIB_D = np.array([0.0000, 0.0164, 0.0992, 0.5246, 3.2754, 9.2107, 12.9029, 14.7039])

# 最小可检测: 取 r=0.02 处 D_KL (标定最小非零档) 为"检测下限"
FLOOR_D = CALIB_D[1]          # 0.0164 (r=0.02)


def map_to_r(D):
    """给定 D_KL, 在标定曲线上反插值出等价相对噪声尺度 r_eq (0..1)。"""
    if D <= 0:
        return 0.0
    if D >= CALIB_D[-1]:
        return 1.0
    return float(np.interp(D, CALIB_D, CALIB_R))


def main(path):
    r = json.load(open(path))
    n = r["n_probes"]
    print(f"model {r['model']} layer 8 | probes n={n}")
    zn = r["zero_noise"]
    print(f"zero-noise sanity: D_KL(full)={zn['KL_full']:.6f}  D_KL(trunc20)={zn['KL_trunc20']:.6f}")

    print("\n== LoRA adapter-scale sensitivity (Qwen clean || Qwen+LoRA) ==")
    rows = r["lora_sweep"]
    print(f"{'s':>6} {'D_KL_full':>10} {'D_KL_trunc20':>12} {'r_eq(full)':>10} {'r_eq(trunc)':>10}")
    s_arr, d_arr = [], []
    for row in rows:
        s, d, dt = row["s"], row["KL_full"], row["KL_trunc20"]
        s_arr.append(s); d_arr.append(d)
        print(f"{s:>6.2f} {d:>10.4f} {dt:>12.4f} {map_to_r(d):>10.3f} {map_to_r(dt):>10.3f}")

    # 小 s 二次拟合 D = c s^2
    s_arr = np.array(s_arr); d_arr = np.array(d_arr)
    small = d_arr > 0
    if small.sum() >= 2:
        c = float(np.dot(s_arr[small]**2, d_arr[small]) / np.dot(s_arr[small]**2, s_arr[small]**2))
        print(f"\nsmall-s quadratic fit:  D_KL(full) ~= {c:.4f} · s^2")
        # 在 s=0.1 处与标定斜率对比
        s_test = 0.1
        D_at = c * s_test**2
        r_eq = map_to_r(D_at)
        print(f"  at s=0.10  ->  D_KL={D_at:.4f}  ~ 等价高斯噪声 r_eq={r_eq:.3f}")

    # 检出判定
    s_full = rows[-1]
    print(f"\nfull adapter (s=1.0): D_KL(full)={s_full['KL_full']:.4f}  r_eq={map_to_r(s_full['KL_full']):.3f}")
    # 求首次超过检测下限的 s
    thr_s = None
    for row in rows:
        if row["KL_full"] > FLOOR_D:
            thr_s = row["s"]; break
    print(f"detection floor (r=0.02 -> D_KL={FLOOR_D:.4f}); first sweep s above it = {thr_s}")
    # 超过 floor 的 s 比例阈值(插值)
    if rows[0]["KL_full"] <= FLOOR_D < rows[-1]["KL_full"]:
        d = np.array([x["KL_full"] for x in rows]); ss = np.array([x["s"] for x in rows])
        s_cross = float(np.interp(FLOOR_D, d, ss))
        print(f"  interpolated s where D_KL crosses floor: s ≈ {s_cross:.3f}")
        print(f"  -> 该点等价噪声 r_eq ≈ {map_to_r(FLOOR_D):.3f} (即 r=0.02, 标定最小档)")

    print("\n== cross-family Qwen -> Llama (tokenizer-mismatched) ==")
    cf = r["cross_family_qwen_to_llama"]
    print(f"shared surface tokens: {cf['shared_tokens']} | probes eval: {cf['n_probes_evaluated']}")
    print(f"D_KL(full, shared)={cf['KL_full_shared']:.4f}  D_KL(trunc, shared)={cf['KL_trunc_shared']:.4f}")
    print(f"  vs LoRA full-adapter D_KL={s_full['KL_full']:.4f} ; vs Gaussian ceiling ~{CALIB_D[-1]:.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lora_test_results.json"))
