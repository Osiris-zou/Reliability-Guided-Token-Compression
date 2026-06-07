import argparse
import csv
import numpy as np


def paired_bootstrap(tome_path, ours_path, samples=10000, seed=0):
    tome = np.load(tome_path).astype(np.float32)
    ours = np.load(ours_path).astype(np.float32)

    if tome.shape != ours.shape:
        raise ValueError(f"Shape mismatch: {tome.shape} vs {ours.shape}")

    diff = ours - tome
    delta = float(diff.mean() * 100.0)

    rng = np.random.default_rng(seed)
    n = len(diff)

    boot = np.empty(samples, dtype=np.float32)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean() * 100.0

    low, high = np.percentile(boot, [2.5, 97.5])

    return {
        "num_images": n,
        "tome_top1": float(tome.mean() * 100.0),
        "ours_top1": float(ours.mean() * 100.0),
        "delta_top1": delta,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "samples": samples,
        "seed": seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tome", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    res = paired_bootstrap(
        tome_path=args.tome,
        ours_path=args.ours,
        samples=args.samples,
        seed=args.seed,
    )

    row = {
        "model": args.model,
        "r": args.r,
        **res,
    }

    print("\n========== Paired Bootstrap ==========")
    print(f"Model       : {args.model}")
    print(f"r           : {args.r}")
    print(f"Images      : {row['num_images']}")
    print(f"ToMe Top-1  : {row['tome_top1']:.3f}")
    print(f"Ours Top-1  : {row['ours_top1']:.3f}")
    print(f"Delta Top-1 : {row['delta_top1']:+.3f}")
    print(f"95% CI      : [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]")
    print("======================================\n")

    if args.out_csv:
        fieldnames = [
            "model", "r", "num_images", "tome_top1", "ours_top1",
            "delta_top1", "ci95_low", "ci95_high", "samples", "seed",
        ]
        write_header = True
        try:
            with open(args.out_csv, "r", encoding="utf-8-sig") as f:
                if f.readline():
                    write_header = False
        except FileNotFoundError:
            pass

        with open(args.out_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"Appended to: {args.out_csv}")


if __name__ == "__main__":
    main()
