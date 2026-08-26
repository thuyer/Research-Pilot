import json
import os
import sys
import time
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

    try:
        # Only the public task prompt is sent to the Agent.
        agent_result = loop(case["prompt"])
    except Exception as error:
        error_message = str(error)

    return {
        "case_id": case["id"],
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

    keyword_error = check_keywords(
        result.get("std_output", ""),
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
