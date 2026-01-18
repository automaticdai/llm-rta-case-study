import os
import json
import subprocess
from typing import Dict, Any, List

import yaml  # pip install pyyaml
from anthropic import Anthropic  # pip install anthropic


def load_config(path: str = "agents.yaml") -> Dict[str, Any]:
    """Load YAML configuration for agents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_json_from_text(text: str) -> Any:
    """
    Extract the first JSON-like object from a string.

    Prefer content inside ```json ... ``` blocks.
    Strategy:
      1) Try strict json.loads on the object.
      2) If that fails, try ast.literal_eval after normalising triple quotes.
      3) If that still fails, as a last resort extract just the "files" array
         (path/content pairs) and return {"plan": "", "files": [...]}.
    """
    import re
    import ast

    # Prefer a fenced ```json ... ``` block if present
    code_block_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = code_block_match.group(1) if code_block_match else text

    candidate = candidate.strip()

    # Isolate something that looks like a top-level object
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    obj_str = candidate[start : end + 1]

    # 1) Strict JSON
    try:
        return json.loads(obj_str)
    except Exception:
        pass

    # 2) Python literal after normalising triple-quotes
    try:
        obj_str_fixed = obj_str.replace('"""', "'''")
        return ast.literal_eval(obj_str_fixed)
    except Exception:
        pass

    # 3) Last resort: extract files only (path/content pairs)
    files: List[Dict[str, str]] = []
    file_pattern = re.compile(
        r'{"path"\s*:\s*"([^"]+)"\s*,\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"\s*}',
        re.DOTALL,
    )
    for m in file_pattern.finditer(candidate):
        path = m.group(1)
        content = m.group(2)
        files.append({"path": path, "content": content})

    if files:
        return {"plan": "", "files": files}

    # If we reach here, give up with a helpful error
    raise ValueError("Failed to parse JSON or recover path/content pairs from model response.")


def call_agent(
    client: Anthropic,
    model: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
) -> str:
    """
    Call an Anthropic model using the Messages API and return plain text.
    `messages` should be a list of {"role": "...", "content": "..."} dicts.
    """
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.2,
        system=system_prompt,
        messages=messages,
    )
    # Assemble all text blocks into one string
    parts: List[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def read_current_files() -> str:
    """Read all relevant source files and return their contents as a formatted string."""
    file_paths = [
        "rta/__init__.py",
        "rta/models.py",
        "rta/analysis.py",
        "rta/generators.py",
        "tests/test_rta.py",
        "tests/test_rta_random.py",
    ]
    sections = []
    for path in file_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                sections.append(f"=== {path} ===\n{content}")
            except Exception as e:
                sections.append(f"=== {path} ===\n[Error reading file: {e}]")
    if not sections:
        return "(No files exist yet)"
    return "\n\n".join(sections)


def apply_file_updates(files: List[Dict[str, str]]) -> None:
    """Write each file's content to disk (overwriting existing files)."""
    for f in files:
        path = f["path"]
        content = f["content"]
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(content)
        print(f"[orchestrator] Updated file: {path}")


def run_pytest() -> Dict[str, Any]:
    """Run pytest -q and capture result.

    Ensures the project root is on PYTHONPATH so 'rta' can be imported.
    """
    print("[orchestrator] Running pytest...")
    env = os.environ.copy()
    cwd = os.getcwd()
    env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        ["pytest", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return {
        "returncode": proc.returncode,
        "output": proc.stdout,
        "status": "pass" if proc.returncode == 0 else "fail",
    }


def main() -> None:
    config = load_config("agents.yaml")
    model = config.get("model", "claude-3-5-sonnet-20241022")

    coder_conf = config["agents"]["coder"]
    checker_conf = config["agents"]["checker"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your environment before running."
        )

    client = Anthropic(api_key=api_key)

    last_checker_feedback = "No feedback yet; this is the initial iteration."
    max_iterations = 20

    for iteration in range(1, max_iterations + 1):
        print(f"\n========== ITERATION {iteration} ==========\n")

        # --- Step 1: Ask Agent_Coder to propose code changes ---
        # Use fresh messages each iteration since we include current files in the prompt
        current_files = read_current_files()

        coder_user_prompt = f"""
You are in iteration {iteration} of implementing the RTA library.

Context for this step:
- Feedback from Agent_Checker (if any): {last_checker_feedback}

CURRENT FILE CONTENTS (read from disk):
{current_files}

Project layout (for reference):
- rta/__init__.py          : package marker (should exist).
- rta/models.py            : task and task set definitions.
- rta/analysis.py          : RTA algorithms and schedulability checks.
- rta/generators.py        : (optional) random task generators such as UUniFast.
- experiments/             : (optional) scripts such as schedulability-vs-utilisation plotting.
- tests/test_rta.py        : unit tests for base-case RTA.
- tests/test_rta_random.py : (optional) random/UUniFast-based tests.

Task for this iteration:
- Propose small, incremental improvements towards a correct and tested base-case RTA.
- Focus on the no-jitter, no-blocking fixed-priority single-core model.
- Follow the JSON protocol strictly:

{{
  "plan": "...",
  "files": [
    {{ "path": "rta/models.py", "content": "FULL FILE CONTENT HERE WITH \\n NEWLINES" }},
    ...
  ]
}}

IMPORTANT:
- Do NOT use triple-quoted strings anywhere.
- For each file, represent the entire file contents as a single JSON string with \\n newlines
  and any double quotes escaped as \\".
"""

        coder_text = call_agent(
            client=client,
            model=model,
            system_prompt=coder_conf["system"],
            messages=[{"role": "user", "content": coder_user_prompt}],
        )
        print("[Agent_Coder raw response]:\n", coder_text[:800], "\n")

        try:
            coder_json = extract_json_from_text(coder_text)
        except Exception as e:
            print("[orchestrator] ERROR parsing Agent_Coder JSON:", e)
            break

        plan = coder_json.get("plan", "")
        files = coder_json.get("files", []) or []
        print(f"[Agent_Coder plan]: {plan}")

        if files:
            apply_file_updates(files)
        else:
            print("[orchestrator] WARNING: Agent_Coder returned no files to update.")

        # Ensure rta/__init__.py exists so `from rta import ...` works in pytest
        if not os.path.exists("rta/__init__.py"):
            os.makedirs("rta", exist_ok=True)
            with open("rta/__init__.py", "w", encoding="utf-8") as f:
                f.write("# RTA package\n")
            print("[orchestrator] Created rta/__init__.py")

        # --- Step 2: Run tests ---
        test_result = run_pytest()
        print("[pytest status]:", test_result["status"])
        print("[pytest output (truncated to 1000 chars)]:\n")
        print(test_result["output"][:1000])

        # --- Step 3: Ask Agent_Checker to review and decide whether to continue ---
        file_list_str = ", ".join(f.get("path", "?") for f in files) if files else "(none)"

        checker_user_prompt = f"""
You are reviewing iteration {iteration}.

What Agent_Coder reported as their plan:
- {plan}

Files modified this iteration:
- {file_list_str}

Latest pytest result:
- status: {test_result["status"]}
- return code: {test_result["returncode"]}
- full output:
{test_result["output"]}

Your job:
- Decide whether we should "continue" or are "done" for the base-case RTA.
- Provide feedback_for_coder with clear, actionable suggestions.
- Optionally propose new or improved tests in new_tests.

STRICT OUTPUT REMINDER:
Return ONLY a JSON object inside a ```json fenced code block, of the form:

{{
  "status": "continue" or "done",
  "feedback_for_coder": "Text using \\n for newlines and escaped double quotes (\\"like this\\").",
  "new_tests": [
    {{
      "description": "what this test checks",
      "suggested_code": "pytest-style test function or helper code with \\n newlines"
    }}
  ]
}}

Do NOT include any commentary outside the JSON object.
"""

        checker_text = call_agent(
            client=client,
            model=model,
            system_prompt=checker_conf["system"],
            messages=[{"role": "user", "content": checker_user_prompt}],
        )
        print("[Agent_Checker raw response]:\n", checker_text[:800], "\n")

        try:
            checker_json = extract_json_from_text(checker_text)
        except Exception as e:
            print("[orchestrator] ERROR parsing Agent_Checker JSON:", e)
            break

        status = checker_json.get("status", "continue")
        feedback = checker_json.get("feedback_for_coder", "")
        new_tests = checker_json.get("new_tests", []) or []

        print(f"[Agent_Checker status]: {status}")
        print(f"[Agent_Checker feedback]: {feedback}")

        # Append any suggested tests to tests/test_rta.py
        if new_tests:
            tests_path = "tests/test_rta.py"
            os.makedirs(os.path.dirname(tests_path), exist_ok=True)
            with open(tests_path, "a", encoding="utf-8") as tf:
                tf.write("\n\n# === New tests suggested by Agent_Checker ===\n")
                for t in new_tests:
                    desc = t.get("description", "")
                    code = t.get("suggested_code", "")
                    if desc:
                        tf.write(f"# {desc}\n")
                    if code:
                        tf.write(code)
                        if not code.endswith("\n"):
                            tf.write("\n")
            print("[orchestrator] Appended new tests from Agent_Checker to tests/test_rta.py")

        last_checker_feedback = feedback or "No feedback provided."

        if status.lower().strip() == "done":
            print("[orchestrator] Agent_Checker indicated we are DONE. Stopping iterations.")
            break

    print("\n[orchestrator] Finished.")


if __name__ == "__main__":
    main()
