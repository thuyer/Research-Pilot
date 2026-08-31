import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件

extractor_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

MODEL = os.environ.get("MODEL_ID", "gpt-4o")


EXTRACTOR_SYSTEM_PROMPT = """
你是无状态的实验结果解析器。唯一的职责是分析 stdout/stderr 并提取结构化指标。
1. 【忠于原文】：只能提取日志中显式打印出的数值指标。严禁推测、虚构或计算未打印的指标。
2. 【过滤无效值】：NaN、Inf、-Inf 属于非合法数值！如果目标核心指标出现无效值，绝对不能写入 metrics，metrics 必须返回空字典 {}，且必须将 outcome 标记为 "failed"，failure_type 标记为 "nan_or_inf"。
3. 【指标空集】：若没有提取到任何有效数值指标，"metrics" 必须返回空字典 {}。
4. 【目标指标对齐与归一化】：
   - 如果输入数据中指定了 "target_metric"（且不为 null），必须优先将日志中与之对应的核心指标命名为该 "target_metric" 的名字！
   - 尝试将其余原始指标映射为通用标准键名（如 "loss", "accuracy", "reward", "f1" 等）。
   - 如果同一个指标有多个过程值（如 step_1_loss, step_2_loss），只取【最后一次打印】的值。
5. 【纯 JSON 输出】：只输出合法格式的 JSON，严禁带有任何 Markdown 标记或解释文字。

【输出 JSON 格式要求】
{
  "metrics": {
    "指标名称(优先对齐 target_metric)": 数值(float/int)
  },
  "outcome": "success" | "failed" | "error",
  "failure_type": "nan_or_inf" | "runtime_error" | "timeout" | null,
  "failure_reason": "简短的原因说明" | null
}
"""

def extract_experiment_summary(raw_execute_result: dict) -> dict:
    """
    Input is a dictionary which contains the experiment result of execution.
    """
    raw_result = json.dumps(raw_execute_result, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下实验运行日志，并按要求提取指标：\n{raw_result}"}
    ]

    try:
        response = extractor_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},  # 强制要求输出合法 JSON
            max_tokens=1024,
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content.")
            
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError(f"No valid JSON object found in response: {content[:100]}...")
            
        return json.loads(json_match.group(0))

    except Exception as e:
        return {
            "metrics": {},
            "outcome": "error",
            "failure_type": "extractor_error",
            "failure_reason": f"Extractor failed to parse LLM output: {str(e)}"
        }