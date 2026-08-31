import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.9)
args = parser.parse_args()

print(f"Starting accuracy experiment with threshold={args.threshold}")

# Deterministic synthetic result. No dataset is needed for this harness test.
accuracy = 0.90 if args.threshold <= 0.5 else 0.60

print(f"Accuracy: {accuracy:.2f}")
print("Evaluation finished.")
