import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="crash")
args = parser.parse_args()

print(f"Starting runtime-error experiment with mode={args.mode}")

if args.mode == "crash":
    raise RuntimeError("Synthetic runtime failure: invalid experiment configuration")

print("Experiment finished successfully.")
