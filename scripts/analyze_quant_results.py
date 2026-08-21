#!/usr/bin/env python3
"""
analyze_quant_results.py — 解释远程量化实验的 D_KL 结果: 指纹能否识别量化 / 多灵敏。

读 quant_test_results.json, 用论文高斯噪声标定曲线 (Table 3, r -> D_KL_full,
全词表, clamp -30) 把量化引起的 D_KL 映射回"等价相对噪声尺度 r_eq", 即
使量化模型与该量级高斯噪声扰动在指纹上不可区分。
"""
import json, os, sys
import numpy as np

# 论文 Table 3 高斯噪声标定: r -> D_KL_full (全词表, clamp -30)
CALIB = {
    "qwen":  dict(r=[0.00,0.02,0.05,0.10,0.20,0.40,0.60,1.00],
                  d=[0.0000,0.0164,0.0992,0.5246,3.2754,9.2107,12.9029,14.7039]),
    "llama": dict(r=[0.00,0.02,0.05,0.10,0.20,0.40,0.60,1.00],
                  d=[0.0000,0.0102,0.0562,0.2181,1.0944,5.7986,9.6807,11.3151]),
    "gemma": dict(r=[0.00,0.02,0.05,0.10,0.20,0.40,0.60,1.00],
                  d=[0.0000,0.0275,0.2301,0.9680,4.7647,13.1882,23.0286,26.6323]),
}
FLOOR_D = {k: v["d"][1] for k, v in CALIB.items()}   # r=0.02 处 D_KL (检测下限)


def map_to_r(tag, D):
    c = CALIB[tag]
    if D <= 0:
        return 0.0
    if D >= c["d"][-1]:
        return 1.0
    return float(np.interp(D, c["d"], c["r"]))


def main(path):
    r = json.load(open(path))
    print(f"method: {r['method']}")
    print(f"args: {r['args']}")

    tags = ["qwen"] + [c["tag"] for c in r.get("cross_models", [])]
    for tag in tags:
        m = r.get(tag)
        if m is None:
            m = next(c for c in r.get("cross_models", []) if c["tag"] == tag)
        cal = CALIB[tag]
        floor = FLOOR_D[tag]
        print(f"\n=== [{tag}] {m['model_id']} (L{m['layer']}) | probes n={m['n_probes']} ===")
        zn = m["zero_noise"]
        print(f"  zero-noise: D_KL(full)={zn['KL_full']:.6f}  D_KL(trunc20)={zn['KL_trunc20']:.6f}")

        # 跨模型单点 (cross_models 条目只有 cross_points)
        cp = m.get("cross_points")
        if cp:
            print(f"  int4/int8 points  (r_eq = equivalent Gaussian noise scale, {tag} calib)")
            print(f"    {'bits':>4} {'D_KL_full':>10} {'D_KL_trunc20':>12} {'r_eq(full)':>10}  exceed floor?")
            for row in cp:
                b = row["bits"]; df = row["KL_full"]; dt = row["KL_trunc20"]
                rqf = map_to_r(tag, df)
                flag = "YES" if df > floor else "no"
                print(f"    {b:>4} {df:>10.4f} {dt:>12.4f} {rqf:>10.3f}  {flag}")
            continue

        # 位宽扫描
        print(f"  bit-width sweep  (r_eq = equivalent Gaussian noise scale, {tag} calib)")
        print(f"    {'bits':>4} {'D_KL_full':>10} {'D_KL_trunc20':>12} {'r_eq(full)':>10} {'r_eq(trunc)':>11}  exceed floor?")
        for row in m["bit_sweep"]:
            b = row["bits"]; df = row["KL_full"]; dt = row["KL_trunc20"]
            rqf = map_to_r(tag, df); rqt = map_to_r(tag, dt)
            flag = "YES" if df > floor else "no"
            print(f"    {b:>4} {df:>10.4f} {dt:>12.4f} {rqf:>10.3f} {rqt:>11.3f}  {flag}")

        # 层比例扫描
        fs = m.get("frac_sweep")
        if fs:
            print(f"  layer-fraction sweep @ int4 (continuous)")
            print(f"    {'frac':>5} {'D_KL_full':>10} {'D_KL_trunc20':>12} {'r_eq(full)':>10}")
            for row in fs:
                df = row["KL_full"]; dt = row["KL_trunc20"]
                rqf = map_to_r(tag, df); rqt = map_to_r(tag, dt)
                print(f"    {row['frac']:>5.2f} {df:>10.4f} {dt:>12.4f} {rqf:>10.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "quant_test_results.json"))
