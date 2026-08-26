import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import subprocess
from pathlib import Path
import sys
import time
import uuid
from datetime import datetime

load_dotenv()  # 读取 .env 文件

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") # 如果用了中转站，会自动读取中转地址
)

MODEL = os.environ.get("MODEL_ID", "gpt-4o")
SYSTEM = """
You are ResearchPilot, an LLM research experiment assistant.
Follow this workflow for each user task:
1. Inspect only the requested experiment script first.
2. Run the experiment with the user's initial parameters.
3. Base any next experiment on the returned result from the immediately previous run.
4. Do not read README files, training documents, or unrelated files unless the requested script is insufficient.
5. Do not load historical experiment records unless the user explicitly asks for historical comparison.
6. Stop as soon as the requested experiment succeeds or the problem is diagnosed.
Do not modify files automatically.
"""
tools = [
    {
        "type": "function",
        "function":{
            "name": "list_files",
            "description": "List files and directories in a specified directory to inspect the workspace structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from workspace root. Defaults to '.' for root."
                    }
                },
                "required": [],
            },
        }     
    },
    {
        "type": "function",
        "function":{
            "name": "read_file",
            "description": "Read line-delimited text content from a file with pagination support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the target file to read."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The 0-based line number to start reading from. Defaults to 0."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Defaults to 5000."
                    }
                },
                "required": ["path"],
            },
        }     
    },
    {
        "type": "function",
        "function":{
            "name": "search_code",
            "description": "Search for exact string keywords across code files in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The exact substring or identifier to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory or file path to scope the search. Defaults to '.' (entire workspace)."
                    }
                },
                "required": ["keyword"],
            },
        }     
    },
    {
        "type": "function",
        "function":{
            "name": "run_experiment",
            "description": "Execute a Python experiment script under 'experiment_repo' directory and capture output logs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "Relative path of the Python script under experiment_repo (e.g., 'experiment_repo/train.py')."
                    },
                    "args": {
                        "type": "object",
                        "description": "Key-value dictionary of command-line arguments to pass to the script, e.g., {'lr': 0.001, 'batch_size': 32}."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Execution timeout in seconds. Defaults to 120."
                    },
                },
                "required": ["script_path"],
            },
        }     
    },
    {
        "type": "function",
        "function":{
            "name": "load_experiment_records",
            "description": "Retrieve previous experiment execution history and logs for a given script, sorted newest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "Relative path of the Python experiment script under experiment_repo (e.g., 'experiment_repo/train.py')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of records to return. Defaults to 50."
                    }
                },
                "required": ["script_path"]
            },
        }     
    },
]

SAFE_ROOT = Path.cwd().resolve()  # Set the safe root dir to the current working dir.
RUN_DIR = SAFE_ROOT / "runs"
EXPERIMENT_DIR = SAFE_ROOT / "experiment_repo"

def list_files(path="."):
    """
    List files in a directory for llm invocations.
    Args:
        path(str): The directory path to list files from. Defaults to the current directory.
    Returns:
        list: A list of file names in the specified directory.
    """
    try:
        path = path.strip()  # Remove any leading/trailing whitespace
        target_path = SAFE_ROOT / path.lstrip("/")  # Ensure the path is relative to SAFE_ROOT
        target_path = target_path.resolve()

        # Check if the target path is within the safe root dir.
        if  SAFE_ROOT not in target_path.parents and target_path != SAFE_ROOT:
            return "Error: Path escapes the safe root directory."
        # Check if the target path is existing and is a dir.
        if not target_path.exists():
            return f"Error: Path {target_path} does not exist."
        if not target_path.is_dir():
            return f"Error: Path {target_path} is not a directory."
        return [{"name": item.name, "type": "file" if item.is_file() else "directory"} for item in target_path.iterdir()]
    except PermissionError:
        return f"Error: Permission denied for path {target_path}."
    except TypeError:
        return "Error: Invalid path type. Please provide a string."
    except Exception as e:
        return f"Error: Unexpected error occurred: {e}"

def read_file(path: str, offset: int = 0, limit: int = 5000) -> str:
    """
    Read the content of file for llm invocations.
    Args:
        path: The path to read file.
        offset: The beginning line which need to be read of this time.Defaults to 0.
        limit: The limit of the total reading lines count. Defaults to 5000.
    Returns:
        str: The content of the specified file.
    """
    try:
        path = path.strip()  # Remove any leading/trailing whitespace
        target_path = SAFE_ROOT / path.lstrip("/")  # Ensure the path is relative to SAFE_ROOT
        target_path = target_path.resolve()

        # Check if the target path is within the safe root dir.
        if  SAFE_ROOT not in target_path.parents and target_path != SAFE_ROOT:
            return "Error: Path escapes the safe root directory."
        # Check if the target path is existing and is a file.
        if not target_path.exists():
            return f"Error: Path {target_path} does not exist."
        if not target_path.is_file():
            return f"Error: Path {target_path} is not a file."
        # Check the file size to avoid reading very large files
        if target_path.stat().st_size > 100 * 1024 * 1024:
            return f"Error: File {target_path} is too large to read."
        if limit < 1 or limit > 5000:
            limit = 5000
        
        # Return the content of the file, truncated if necessary
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        if total_lines == 0:
            return "The file is empty."
        if offset < 0:
            offset = max(0, offset + total_lines)
        if offset >= total_lines:
            return f"Offset{offset} exceeds file lengths {total_lines}lines."
        content = "".join(lines[offset : offset + limit])
        if len(content) > 8000:
            return content[:8000] + f"...(truncated, Total {len(content)} characters)"
        return content

    except FileNotFoundError:
        path_display = target_path if 'target_path' in locals() else path
        return f"Error: File {path_display} not found."
    except UnicodeDecodeError:
        try:
            return target_path.read_text(encoding="gbk")
        except UnicodeDecodeError:
            return f"Error: Cannot decode file {target_path}. It may not be a text file."
    except PermissionError:
        return f"Error: Permission denied for path {target_path}."
    except TypeError:
        return "Error: Invalid path type. Please provide a string."
    except Exception as e:
        return f"Error: Unexpected error occurred: {e}"

def search_code(keyword: str, path: str = ".") -> list:
    """
    Search for a keyword in code files within a directory.
    Args:
        keyword(str): The keyword to search for.
        path(str): The directory path to search in. Defaults to the current directory.
    Returns:
        list: A list of dictionaries containing file paths and lines where the keyword was found.
    """
    ## TODO 这里之后能不能改成让llm来筛选
    IGNORED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".pdf", ".zip", ".tar", ".gz", 
    ".7z", ".exe", ".dll", '.o', '.mp3', '.mp4', '.avi', '.mov', ".git", ".venv", "venv", "__pycache__", "node_modules"}
    MAX_TOTAL_RESULTS = 100  # Limit the total number of results to avoid overwhelming output

    try:
        path = path.strip()  # Remove any leading/trailing whitespace
        target_path = SAFE_ROOT / path.lstrip("/")  # Ensure the path is relative to SAFE_ROOT
        target_path = target_path.resolve()

        # Check if the target path is within the safe root dir.
        if  SAFE_ROOT not in target_path.parents and target_path != SAFE_ROOT:
            return "Error: Path escapes the safe root directory."
        # Check if the target path is existing and is a dir.
        if not target_path.exists():
            return f"Error: Path {target_path} does not exist."
        
        results = []

        # Define a helper function to search keyword within a single file
        def search_in_file(file_path, keyword):
            matches = []
            max_results_per_file = 20  # Limit the number of matches per file
            if file_path.stat().st_size > 5 * 1024 * 1024:
                return matches  # Skip large files
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_number, line in enumerate(f, start=1):
                    if len(matches) >= max_results_per_file:
                        break  # Stop searching after reaching the limit
                    if keyword in line:
                        matches.append({"file": str(file_path), "line": line_number, "content": line.strip()})
            return matches

        if target_path.is_file():
            if target_path.suffix.lower() in IGNORED_EXTENSIONS:
                return []
            results.extend(search_in_file(target_path, keyword))
        elif target_path.is_dir():
            for file_path in target_path.rglob("*"):
                if file_path.suffix.lower() in IGNORED_EXTENSIONS:
                    continue
                if file_path.is_file():
                    if len(results) >= MAX_TOTAL_RESULTS:
                        break  # Stop searching after reaching the total limit
                    results.extend(search_in_file(file_path, keyword))
        else:
            return f"Error: Path {target_path} is not a file or directory."
        
        return results
    except PermissionError:
        return f"Error: Permission denied for path {target_path}."
    except FileNotFoundError as e:
        return f"Error: File not found: {e}"
    except Exception as e:
        return f"Error: Unexpected error occurred: {e}"


def params_to_args(params:dict):
    """
    Convert a dictionary of parameters to a list of command-line arguments.
    Args:
        params (dict): A dictionary of parameters, e.g., {"lr": 0.001, "batch_size": 32}
    Returns:
        list: A list of command-line arguments, e.g., ["--lr", "0.001", "--batch_size", "32"]
    """
    args = []
    if not isinstance(params,dict) or not params:
        return args
    for key, value in params.items():
        if not key:
            continue
        if not key.startswith("--"):
            args.append(f"--{key}")
        else:
            args.append(key)
        args.append(str(value))
    return args

def save_experiment_log(data: dict):
    # Ensure the log directory exists
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history_file = RUN_DIR / "manifest.jsonl"

    with open(history_file, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def run_experiment(script_path: str, args: dict = None, timeout_seconds: int = 120):
    """
    Run the experiment, and return the result and log it to the log file.
    Args:
        script_path: The path of the experiment script to run.
        args: The arguments dict to run the experiment script. Such as {"lr": 0.001, "batch_size": 32}
        timeout_seconds: The timeout for the experiment.
    Returns:
        dict: The result data  of the experiment.
    """
    run_id = uuid.uuid4().hex
    result_data = {
        "run_id": run_id,
        "script_path": str(script_path),
        "params": args,
        "status": "unknown",
        "exit_code": -1,
        "std_output": "",
        "std_error":"",
        "created_time": datetime.now().isoformat(),
        "elapsed_time": 0.0
    }

    try:
        script_path = script_path.strip()  # Remove any leading/trailing whitespace
        target_path = SAFE_ROOT / script_path.lstrip("/")
        target_path = target_path.resolve()

        if SAFE_ROOT not in target_path.parents and target_path != SAFE_ROOT:
            result_data["status"] = "rejected"
            result_data["std_error"] = "The file is out of SAFE_ROOT."
            save_experiment_log(result_data)
            return result_data
        if EXPERIMENT_DIR not in target_path.parents:
            result_data["status"] = "rejected"
            result_data["std_error"] = "The file is out of EXPERIMENT_DIR."
            save_experiment_log(result_data)
            return result_data
        if not target_path.exists():
            result_data["status"] = "rejected"
            result_data["std_error"] = "The file doesn't exit."
            save_experiment_log(result_data)
            return result_data
        if not target_path.is_file():
            result_data["status"] = "rejected"
            result_data["std_error"] = "This script_path is not to a file."
            save_experiment_log(result_data)
            return result_data
        if target_path.suffix != ".py":
            result_data["status"] = "rejected"
            result_data["std_error"] = "Only support the script of python.This isn't a python script."
            save_experiment_log(result_data)
            return result_data
        

        cli_args = params_to_args(args) if args else []
        cmd = []
        cmd.append(sys.executable)  
        cmd.append(target_path)     
        cmd.extend(cli_args)

        # Run the command using subprocess
        start_time = time.time()
        result = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        end_time = time.time()
        
        result_data["params"] = args
        result_data["elapsed_time"] = end_time - start_time
        result_data["status"] = "completed" if result.returncode == 0 else "failed"
        result_data["exit_code"] = result.returncode        
        if result.stdout:
            result_data["std_output"] = result.stdout.strip()[-1000:]
        if result.stderr:
            result_data["std_error"] = result.stderr.strip()[-1000:]

    except subprocess.TimeoutExpired:
        end_time = time.time()
        result_data["status"] = "timeout"
        result_data["elapsed_time"] = end_time - start_time
        
    except Exception as e:
        result_data["status"] = "error"
        result_data["std_error"] = f"Unexpected error occurred: {e}"
    
    save_experiment_log(result_data)
    return result_data

def load_experiment_records(script_path: str, limit: int = 50):
    """
    Load all experiment records from the manifest matching the given script_path.
    Args:
        script_path: The just ran experiment path
        limit: The count of return logs
    Return:
        list: A list of matched records. Each record is a dict with keys:
              run_id, status, exit_code, std_output, std_error, elapsed_time, params, created_time.
              Returns an empty list if the manifest doesn't exist or errors occur.
    """
    matched_record = []
    try:
        
        script_path = str(script_path.strip()).lstrip("/")
        log_manifest = RUN_DIR / "manifest.jsonl"
        if not log_manifest.exists():
            return matched_record
        

        with open(log_manifest, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue

                    line = json.loads(line)
                    if line.get("script_path") == script_path:
                        record_data = {
                            "run_id": line.get("run_id", ""),
                            "status": line.get("status", "unknown"),
                            "exit_code": line.get("exit_code", -1),
                            "std_output": line.get("std_output", ""),
                            "std_error": line.get("std_error", ""),
                            "elapsed_time": line.get("elapsed_time", ""),
                            "params": line.get("params", {}),
                            "created_time": line.get("created_time", "")
                        }
                        matched_record.append(record_data)
                except json.JSONDecodeError as e:
                    continue
        matched_record.sort(key= lambda x : x.get("created_time", ""), reverse=True)
        if limit < 1:
            limit = 50
        if limit > 100:
            limit = 100
        return matched_record[:limit]

    except Exception as e:
        # 静默处理了 TODO
        return []
        

    



TOOL_HANDLERS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "run_experiment": run_experiment,
    "load_experiment_records":load_experiment_records
}

def loop(text, max_steps=12):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text}
    ]
    
    tool_trace = []

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            max_tokens=2048,
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        messages.append(message.model_dump())
        tool_calls = message.tool_calls

        #如果是工具调用
        tool_calls = message.tool_calls
        if tool_calls:
            
            for tool_call in tool_calls:
                tool_call_id = tool_call.id
                tool_function = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                handler = TOOL_HANDLERS.get(tool_function)
                if handler:
                    try:
                        output = handler(**tool_args)

                    except Exception as e:
                        output = f"Error executing {tool_function}: {e}"
                else:
                    output = f"Tool {tool_function} is not registered."

                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                
                tool_trace.append({
                    "tool": tool_function,
                    "args": tool_args,
                    "result": output
                })
                messages.append({"role": "tool", "content": output, "tool_call_id": tool_call_id})
        else:
            return {
                "final_answer": message.content,
                "tool_trace": tool_trace,
                "stopped_reason": "model_finished",
                "steps": step + 1
            }

    return {
        "final_answer": "Agent reached the maximum number of steps.",
        "tool_trace": tool_trace,
        "stopped_reason": "max_steps",
        "steps": max_steps
    }

# 简单测试循环
if __name__ == "__main__":
    while True:
        text = input("User: ")
        if text.lower() in ["exit", "quit"]:
            break
        result = loop(text)
        print(f"Assistant: {result['final_answer']}")
        # print(json.dumps(result["tool_trace"], ensure_ascii=False, indent=2))
