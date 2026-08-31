import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=30.0)
args = parser.parse_args()

print(f"Starting long experiment; expected duration={args.seconds}s")
time.sleep(args.seconds)
print("Long experiment finished successfully.")
