#!/usr/bin/env python3
import numpy as np
import argparse
import pprint

def is_numeric_array(arr):
    return np.issubdtype(arr.dtype, np.number)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    print("\n=== Keys in NPZ ===")
    print(list(data.keys()))

    print("\n=== Parameters ===")
    for key in data:
        arr = data[key]

        print(f"\n{key}:")

        # --- Case 1: numeric array ---
        if isinstance(arr, np.ndarray) and is_numeric_array(arr):
            if arr.size > 20:
                print(f"  shape={arr.shape}")
                print(f"  mean={arr.mean():.4g}, min={arr.min():.4g}, max={arr.max():.4g}")
            else:
                print(arr)

        # --- Case 2: non-numeric array (strings / objects) ---
        elif isinstance(arr, np.ndarray):
            print(f"  shape={arr.shape}, dtype={arr.dtype}")
            pprint.pprint(arr.tolist())

        # --- Case 3: scalar value ---
        else:
            print(arr)

if __name__ == "__main__":
    main()
