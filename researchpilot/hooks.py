import time
from datetime import datetime


class AgentHooks:
    def __init__(self):

        self.events = []
        self.start_counter = None


    def before_tool_call(self, tool_name, tool_args):

        event = {
            "event": "tool_started",
            "tool": tool_name,
            "args": tool_args,
            "started_at": datetime.now().astimezone().isoformat()
        }

        self.events.append(event)
        self.start_counter = time.perf_counter()


    def after_tool_call(self, tool_name, tool_args, result):
        if self.start_counter is None:
            elapsed_ms = None
        else:
            elapsed_ms = (time.perf_counter() - self.start_counter) * 1000
        if isinstance(result, dict):
            result_summary = {
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "metrics": result.get("metrics", {})
            }
        elif isinstance(result, list):
            result_summary = {
                "type": "list",
                "count": len(result)
            }
        else:
            result_summary = {
                "preview": str(result)[-200:]
            }

        event = {
            "event": "tool_completed",
            "tool": tool_name,
            "result_summary": result_summary,
            "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None
        }
        self.events.append(event)
        self.start_counter = None


    def on_tool_error(self, tool_name, tool_args, error):
        if self.start_counter is None:
            elapsed_ms = None
        else:
            elapsed_ms = (time.perf_counter() - self.start_counter) * 1000
        event = {
            "event": "tool_error",
            "tool": tool_name,
            "args": tool_args,
            "error_type": error["type"],
            "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None,
            "message": error["message"]
        }
        
        self.events.append(event)
        self.start_counter = None