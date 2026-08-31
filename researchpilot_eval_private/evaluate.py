import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path


EVALUATOR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATOR_DIR.parent / "researchpilot"
CASE_PATH = EVALUATOR_DIR / "case_v1.json"
RESULTS_PATH = EVALUATOR_DIR / "results.jsonl"

# core.py uses Path.cwd() as its workspace root, so establish it before import.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from core import loop
import state_manager


def load_cases(path: Path) -> list[dict]:
    """Load and minimally validate the private evaluation cases."""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("The case file must contain a JSON list.")

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every case must be a JSON object.")
        if not case.get("id") or not case.get("prompt"):
            raise ValueError("Every case needs an id and a prompt.")
        if not isinstance(case.get("expected_runs"), list):
            raise ValueError(f"Case {case['id']} needs an expected_runs list.")

    return cases


def run_single_case(case: dict) -> dict:
    """Run one isolated Agent task and preserve its complete return value."""
    started_at = time.perf_counter()
    agent_result = None
    error_message = None
    task_id = uuid.uuid4().hex
    initial_state = state_manager.init_state(task_id, case["prompt"])

    try:
        # Only the public task prompt is sent to the Agent.
        state_manager.save_state(initial_state)
        agent_result = loop(case["prompt"], state=initial_state)
    except Exception as error:
        error_message = str(error)

    return {
        "case_id": case["id"],
        "task_id": task_id,
        "agent_result": agent_result,
        "elapsed_seconds": time.perf_counter() - started_at,
        "error_message": error_message,
    }


def extract_run_calls(tool_trace: list[dict]) -> list[dict]:
    """Keep only the run_experiment calls from a tool trace."""
    return [
        item
        for item in tool_trace
        if item.get("tool") == "run_experiment"
    ]


def normalize_tool_result(call: dict) -> dict | None:
    """Return a run result as a dict whether the trace stored a dict or JSON text."""
    raw_result = call.get("result")
    if isinstance(raw_result, dict):
        return raw_result
    if isinstance(raw_result, str):
        try:
            parsed_result = json.loads(raw_result)
        except json.JSONDecodeError:
            return None
        return parsed_result if isinstance(parsed_result, dict) else None
    return None


def check_keywords(text: str, required: list[str], forbidden: list[str]) -> str | None:
    """Return a failure reason when log keyword constraints are violated."""
    normalized_text = text.lower()

    for keyword in required:
        if keyword.lower() not in normalized_text:
            return f"stdout does not contain required keyword: {keyword}"

    for keyword in forbidden:
        if keyword.lower() in normalized_text:
            return f"stdout contains forbidden keyword: {keyword}"

    return None


def check_run_rule(
    call: dict,
    result: dict,
    rule: dict,
    previous_call: dict | None,
) -> str | None:
    """Check one actual run against one declarative expected_runs rule."""
    call_args = call.get("args", {})
    actual_script_path = call_args.get("script_path")
    actual_params = call_args.get("args", {})

    expected_script_path = rule.get("script_path")
    if expected_script_path and actual_script_path != expected_script_path:
        return f"script_path is {actual_script_path!r}, expected {expected_script_path!r}"

    for name, expected_value in rule.get("params", {}).items():
        actual_value = actual_params.get(name)
        if actual_value != expected_value:
            return f"parameter {name!r} is {actual_value!r}, expected {expected_value!r}"

    expected_status = rule.get("status")
    if expected_status and result.get("status") != expected_status:
        return f"status is {result.get('status')!r}, expected {expected_status!r}"

    output_text = result.get("stdout_tail", result.get("std_output", ""))
    keyword_error = check_keywords(
        output_text,
        rule.get("stdout_contains", []),
        rule.get("stdout_not_contains", []),
    )
    if keyword_error:
        return keyword_error

    relations = rule.get("param_relations", {})
    if relations and previous_call is None:
        return "parameter relation requires a previous experiment run"

    if previous_call is not None:
        previous_params = previous_call.get("args", {}).get("args", {})
        for name, relation in relations.items():
            current_value = actual_params.get(name)
            previous_value = previous_params.get(name)
            if relation == "lt_previous":
                if current_value is None or previous_value is None or current_value >= previous_value:
                    return f"parameter {name!r} must be smaller than the previous value"
            else:
                return f"unsupported parameter relation: {relation!r}"

    return None


def check_state_consistency(state: dict, actual_runs: list[dict]) -> str | None:
    """Check that the persisted task state reflects the observed experiment calls."""
    task_id = state.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return "state.task_id is missing"

    if state.get("attempt_count") != len(actual_runs):
        return (
            f"state.attempt_count is {state.get('attempt_count')!r}, "
            f"but observed {len(actual_runs)} experiment calls"
        )

    state_runs = state.get("runs")
    if not isinstance(state_runs, list):
        return "state.runs is not a list"

    if len(state_runs) != len(actual_runs):
        return (
            f"state.runs contains {len(state_runs)} records, "
            f"but observed {len(actual_runs)} experiment calls"
        )

    state_run_ids = {
        run.get("run_id")
        for run in state_runs
        if isinstance(run, dict)
    }

    for call in actual_runs:
        result = normalize_tool_result(call)
        if result is None:
            continue
        run_id = result.get("run_id")
        if run_id not in state_run_ids:
            return f"run_id {run_id!r} is missing from state.runs"

        is_failure = (
            result.get("status") != "completed"
            or result.get("failure_type") is not None
            or result.get("outcome") == "failed"
        )
        if is_failure:
            failure_ids = {
                item.get("run_id")
                for item in state.get("failure_history", [])
                if isinstance(item, dict)
            }
            if run_id not in failure_ids:
                return f"failed run_id {run_id!r} is missing from failure_history"

    metric_name = state.get("target_metric")
    direction = state.get("metric_direction")
    candidates = []
    if metric_name and direction in {"min", "max"}:
        for run in state_runs:
            metrics = run.get("metrics", {})
            value = metrics.get(metric_name) if isinstance(metrics, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                candidates.append((run.get("run_id"), value))

    best_run_id = state.get("best_run_id")
    best_value = state.get("best_metric_value")
    if candidates:
        expected_value = (
            min(value for _, value in candidates)
            if direction == "min"
            else max(value for _, value in candidates)
        )
        expected_ids = {
            run_id for run_id, value in candidates if value == expected_value
        }
        if best_run_id not in expected_ids or best_value != expected_value:
            return (
                f"best state is inconsistent: run_id={best_run_id!r}, "
                f"value={best_value!r}, expected one of {expected_ids!r} "
                f"with value {expected_value!r}"
            )
    elif best_run_id is not None or best_value is not None:
        return "state has best run information but no comparable metric"

    return None


def evaluate_case(case: dict, agent_result: dict | None) -> dict:
    """Score one Agent result using only the case's expected_runs declaration."""
    if not isinstance(agent_result, dict):
        return {
            "case_id": case["id"],
            "passed": False,
            "reason": "Agent did not return a result.",
            "experiment_count": 0,
        }

    tool_trace = agent_result.get("tool_trace", [])
    if not isinstance(tool_trace, list):
        return {
            "case_id": case["id"],
            "passed": False,
            "reason": "Agent result has no valid tool_trace.",
            "experiment_count": 0,
        }

    actual_runs = extract_run_calls(tool_trace)
    expected_runs = case["expected_runs"]

    final_state = agent_result.get("state")
    if not isinstance(final_state, dict):
        return {
            "case_id": case["id"],
            "passed": False,
            "reason": "Agent result has no valid final state.",
            "experiment_count": len(actual_runs),
            "tool_call_count": len(tool_trace),
        }

    if len(actual_runs) != len(expected_runs):
        return {
            "case_id": case["id"],
            "passed": False,
            "reason": (
                f"Expected {len(expected_runs)} experiment runs, "
                f"but observed {len(actual_runs)}."
            ),
            "experiment_count": len(actual_runs),
            "tool_call_count": len(tool_trace),
        }

    state_error = check_state_consistency(final_state, actual_runs)
    if state_error:
        return {
            "case_id": case["id"],
            "passed": False,
            "reason": f"State consistency check failed: {state_error}",
            "experiment_count": len(actual_runs),
            "tool_call_count": len(tool_trace),
        }

    for index, (call, rule) in enumerate(zip(actual_runs, expected_runs)):
        result = normalize_tool_result(call)
        if result is None:
            return {
                "case_id": case["id"],
                "passed": False,
                "reason": f"Run {index + 1} returned an unreadable tool result.",
                "experiment_count": len(actual_runs),
            }

        previous_call = actual_runs[index - 1] if index > 0 else None
        rule_error = check_run_rule(call, result, rule, previous_call)
        if rule_error:
            return {
                "case_id": case["id"],
                "passed": False,
                "reason": f"Run {index + 1}: {rule_error}",
                "experiment_count": len(actual_runs),
                "tool_call_count": len(tool_trace),
            }

    return {
        "case_id": case["id"],
        "passed": True,
        "reason": "All expected experiment runs satisfied their rules.",
        "experiment_count": len(actual_runs),
        "tool_call_count": len(tool_trace),
        "stopped_reason": agent_result.get("stopped_reason"),
        "task_id": final_state.get("task_id"),
    }


def append_result(record: dict) -> None:
    """Append one complete evaluation record without overwriting prior results."""
    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    cases = load_cases(CASE_PATH)
    scores = []

    for case in cases:
        execution = run_single_case(case)
        score = evaluate_case(case, execution["agent_result"])
        record = {
            "evaluated_at": datetime.now().isoformat(),
            **execution,
            "score": score,
        }
        append_result(record)
        scores.append(score)
        print(f"{score['case_id']}: {'PASS' if score['passed'] else 'FAIL'} - {score['reason']}")

    total = len(scores)
    passed = sum(score["passed"] for score in scores)
    average_runs = (
        sum(score["experiment_count"] for score in scores) / total
        if total
        else 0.0
    )
    print(f"Summary: {passed}/{total} passed, average experiment runs: {average_runs:.2f}")


if __name__ == "__main__":
    main()
