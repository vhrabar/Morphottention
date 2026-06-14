#!/usr/bin/env python3

import argparse
import time
import torch




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()


    if torch.cuda.is_available():
        cuda_index = torch.cuda.current_device()
        device = torch.device("cuda", cuda_index)
        print("GPU available:")
        print(f"GPU: {torch.cuda.get_device_name(cuda_index)}")
        print(f"CC: {torch.cuda.get_device_capability(cuda_index)}")
        print(f"VRAM: {torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3:.2f} GiB")
    else:
        print("GPU error!")
        exit(1)

if __name__ == "__main__":
    main()
