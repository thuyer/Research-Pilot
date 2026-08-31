from datetime import datetime
import json
import uuid
from pathlib import Path

"""
主要是对实验state的管理，涉及到读取时整理，保存和更新
"""

STATE_DIR = Path(__file__).resolve().parent / "states"

def init_state(task_id, user_prompt=None):
    return {
        "task_id": task_id,
        "user_prompt": user_prompt,
        "title": None,
        "script_path": None,
        "target_metric": None,
        "metric_direction": None,
        "runs": [],
        "failure_history": [],
        "best_run_id": None,
        "best_metric_value": None,
        "attempt_count": 0,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def set_task_context(state, script_path, target_metric, metric_direction, title): 
    """
    Sets the context information for the current tuning task, including the target script, 
    metric to optimize, and optimization direction. 

    Note:
        MUST be executed BEFORE invoking run_experiment to ensure proper initialization 
        of task parameters and state.

    Args:
        state (dict): The global task state dictionary.
        script_path (str): Path to the target execution script.
        target_metric (str): Name of the core metric to optimize (e.g., 'accuracy' or 'loss').
        metric_direction (str): Optimization direction, either 'max' or 'min'.
        title (str, optional): Custom title or brief summary for the task. Defaults to "".

    Returns:
        dict: The updated state dictionary containing the set context.
    """
    state["script_path"] = script_path
    state["target_metric"] = target_metric
    state["metric_direction"] = metric_direction
    state["title"] = title
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    return {
        "script_path": script_path,
        "target_metric": target_metric,
        "metric_direction": metric_direction
    }

def save_state(state:dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{state['task_id']}.json"
    
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def update_state(state, result_data):

    target_metric = state["target_metric"]
    metric_direction = state["metric_direction"]
    task_id = state["task_id"]
    script_path = result_data.get("script_path", "")    
    run_id = result_data.get("run_id")
    metrics = result_data.get("metrics", {})
    status = result_data.get("status")
    failure_type = result_data.get("failure_type") 
    outcome = result_data.get("outcome")

    run = {
        "run_id": run_id,
        "params": result_data.get("params"),
        "metrics": metrics,
    }
    state.setdefault("runs", []).append(run)

    if status != "completed" or failure_type is not None or outcome == "failed":
        # 提炼简短的失败原因
        reason_raw = result_data.get("failure_reason") or result_data.get("stdout_tail") or "Unknown error"
        # 截取前 150 个字符，防止 Token 膨胀
        clean_reason = str(reason_raw).strip().split('\n')[-1][:150] 

        failure_entry = {
            "run_id": run_id,
            "params": result_data.get("params", {}),
            "failure_type": failure_type or status,
            "reason": clean_reason
        }
        
        state.setdefault("failure_history", [])
        state["failure_history"].append(failure_entry)

    else:
        if target_metric and target_metric in metrics:
            best_value = state.get("best_metric_value")
            value = metrics[target_metric]

            is_better = False
            if best_value is None:
                is_better = True
            elif metric_direction  == "min":
                if value < best_value:
                    is_better = True
            elif metric_direction  == "max":
                if value > best_value:
                    is_better = True
            
            if is_better:
                state["best_metric_value"] = value
                state["best_run_id"] = run_id
            
    state["attempt_count"] += 1
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    return state


