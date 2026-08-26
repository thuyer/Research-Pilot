import argparse
import time
import json
import os

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1) 
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
log_file = "logs/train.log"

print(f"Starting experiment with lr={args.lr}, batch_size={args.batch_size}")

# 模拟 5 个 Step 的训练
losses = []
with open(log_file, "w") as f:
    for step in range(1, 6):
        if args.lr >= 0.01:
            loss = 5.0 + step * 2.0  # 发散
            if step >= 3:
                loss = float("nan") # 第 3 步开始变成 NaN
        else:
            loss = 5.0 / step       # 正常下降
            
        losses.append(loss)
        log_line = f"Step {step}/5 - Loss: {loss:.4f}\n"
        f.write(log_line)
        print(log_line.strip())
        time.sleep(0.5)

print("Training finished.")