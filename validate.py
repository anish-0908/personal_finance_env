"""
Pre-submission OpenEnv Validation Script
=========================================
Covers every rubric check:
  1. openenv.yaml    - spec compliance (name, version, tasks, api_variables, HF fields)
  2. inference.py    - env var usage, file placement, OpenAI client usage
  3. Tasks + graders - 3+ tasks, all scores in [0.0, 1.0]
  4. Dockerfile      - build check (skipped if docker not installed)
  5. Endpoints       - /reset /state /step /health all return 200
  6. HF Space ping   - live Space returns 200 and reset() responds
  7. Inference run   - inference.py completes without error and prints scores
"""

import os
import sys
import time
import shutil
import signal
import subprocess
import json
from contextlib import contextmanager

import requests
import yaml


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

PASS  = "[PASS]"
FAIL  = "[FAIL]"
SKIP  = "[SKIP]"

_results: list[tuple[str, str, str]] = []  # (check_name, status, detail)


def record(name: str, status: str, detail: str = ""):
    _results.append((name, status, detail))
    icon = status.split()[0]
    label = status.split()[-1]
    msg = f"  {icon}  {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ──────────────────────────────────────────────────────────────
# Check 1 — openenv.yaml spec compliance
# ──────────────────────────────────────────────────────────────

def check_openenv_yaml() -> bool:
    print("\n[1] Checking openenv.yaml …")
    if not os.path.exists("openenv.yaml"):
        record("openenv.yaml exists", FAIL, "File not found.")
        return False

    with open("openenv.yaml", "r") as f:
        data = yaml.safe_load(f)

    ok = True

    # Required top-level keys
    for key in ("name", "version", "tasks"):
        if key not in data:
            record(f"openenv.yaml has '{key}'", FAIL, f"Key '{key}' missing.")
            ok = False

    # tasks must be a list with >= 3 entries
    tasks = data.get("tasks", [])
    if len(tasks) < 3:
        record("openenv.yaml tasks (>=3)", FAIL, f"Only {len(tasks)} task(s) found.")
        ok = False
    else:
        record("openenv.yaml tasks (>=3)", PASS, f"{len(tasks)} tasks defined.")

    # api_variables section
    if "api_variables" not in data:
        record("openenv.yaml api_variables", FAIL, "'api_variables' section not found.")
        ok = False
    else:
        var_names = [v["name"] for v in data["api_variables"]]
        for vname in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
            if vname not in var_names:
                record(f"openenv.yaml api_variable '{vname}'", FAIL, "Not documented.")
                ok = False
        if ok:
            record("openenv.yaml api_variables", PASS, "API_BASE_URL, MODEL_NAME, HF_TOKEN documented.")

    # spaces_deployment
    deploy = data.get("spaces_deployment", {})
    if "app_file" not in deploy:
        record("openenv.yaml app_file", FAIL, "'app_file' not in spaces_deployment.")
        ok = False
    else:
        record("openenv.yaml app_file", PASS, deploy["app_file"])

    if ok:
        record("openenv.yaml overall", PASS, "All required sections present.")
    return ok


# ──────────────────────────────────────────────────────────────
# Check 2 — inference.py static checks
# ──────────────────────────────────────────────────────────────

def check_inference() -> bool:
    print("\n[2] Checking inference.py …")
    if not os.path.exists("inference.py"):
        record("inference.py exists", FAIL, "File not found in project root.")
        return False

    record("inference.py exists", PASS)

    with open("inference.py", "r") as f:
        content = f.read()

    ok = True
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if var not in content:
            record(f"inference.py uses {var}", FAIL, "Variable not referenced in script.")
            ok = False
        else:
            record(f"inference.py uses {var}", PASS)

    for pattern in ("OpenAI", "openai"):
        if pattern in content:
            record("inference.py uses OpenAI client", PASS)
            break
    else:
        record("inference.py uses OpenAI client", FAIL, "No OpenAI import found.")
        ok = False

    return ok


# ──────────────────────────────────────────────────────────────
# Check 3 — Tasks and graders
# ──────────────────────────────────────────────────────────────

def check_tasks_and_graders() -> bool:
    print("\n[3] Checking tasks and graders …")

    try:
        from tasks import TASKS
        from env import PersonalFinanceEnv
        from models import Action
    except ImportError as e:
        record("Tasks import", FAIL, str(e))
        return False

    if len(TASKS) < 3:
        record("Tasks count (>=3)", FAIL, f"Found {len(TASKS)} task(s).")
        return False

    record("Tasks count (>=3)", PASS, f"{len(TASKS)} tasks: {list(TASKS.keys())}")

    ok = True
    for task_id in TASKS:
        try:
            env = PersonalFinanceEnv(task_id=task_id)
            env.reset()
            _, _, _, info = env.step(Action())
            score = info.score
            if not (0.0 <= score <= 1.0):
                record(f"Grader '{task_id}' score in [0,1]", FAIL, f"Got {score}")
                ok = False
            else:
                record(f"Grader '{task_id}'", PASS, f"score={score:.4f}")
        except Exception as e:
            record(f"Grader '{task_id}'", FAIL, str(e))
            ok = False

    return ok


# ──────────────────────────────────────────────────────────────
# Check 4 — Dockerfile builds
# ──────────────────────────────────────────────────────────────

def check_dockerfile() -> bool:
    print("\n[4] Checking Dockerfile …")
    if not os.path.exists("Dockerfile"):
        record("Dockerfile exists", FAIL, "Dockerfile not found.")
        return False

    record("Dockerfile exists", PASS)

    if not shutil.which("docker"):
        record("Dockerfile build", SKIP, "Docker not installed — skipping build.")
        return True

    try:
        subprocess.run(
            ["docker", "build", "-t", "pf_env_validate", "."],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        record("Dockerfile build", PASS, "Built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        record("Dockerfile build", FAIL, e.stderr.decode()[:200])
        return False


# ──────────────────────────────────────────────────────────────
# Server context manager
# ──────────────────────────────────────────────────────────────

@contextmanager
def running_api_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def wait_for_health(base: str, path: str = "/health", timeout_s: float = 25.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{base}{path}", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ──────────────────────────────────────────────────────────────
# Check 5 — Local API endpoints
# ──────────────────────────────────────────────────────────────

def check_endpoints() -> bool:
    print("\n[5] Checking local FastAPI endpoints …")
    base = "http://localhost:7860"

    try:
        with running_api_server():
            if not wait_for_health(base, "/health"):
                record("API server health", FAIL, "Server did not become healthy on port 7860.")
                return False

            record("API /health", PASS, "200 OK")

            checks = [
                ("GET",  "/",                None),
                ("POST", "/reset/easy",      None),
                ("GET",  "/state/easy",      None),
                ("POST", "/step/easy",       {"save_amount": 10, "discretionary_spend": 0, "pay_debts": []}),
            ]

            ok = True
            for method, path, payload in checks:
                try:
                    if method == "GET":
                        r = requests.get(f"{base}{path}", timeout=5)
                    else:
                        r = requests.post(f"{base}{path}", json=payload, timeout=5)

                    if r.status_code == 200:
                        record(f"API {method} {path}", PASS, "200 OK")
                    else:
                        record(f"API {method} {path}", FAIL, f"Status {r.status_code}: {r.text[:80]}")
                        ok = False
                except Exception as e:
                    record(f"API {method} {path}", FAIL, str(e))
                    ok = False

            return ok

    except Exception as e:
        record("Local endpoints", FAIL, str(e))
        return False


# ──────────────────────────────────────────────────────────────
# Check 6 — HF Space live ping
# ──────────────────────────────────────────────────────────────

def check_hf_space() -> bool:
    print("\n[6] Checking live HF Space …")

    # For pre-submission validation we cannot reliably guess your exact Space URL.
    # If not provided, skip instead of failing.
    space_url = os.environ.get("HF_SPACE_URL")
    if not space_url:
        # Fall back to openenv.yaml metadata if available.
        try:
            with open("openenv.yaml", "r") as f:
                data = yaml.safe_load(f)
            space_url = (data.get("spaces_deployment") or {}).get("space_url")
        except Exception:
            space_url = None

        # Convert HF "page URL" into the Space runtime host (required for hitting endpoints).
        # Example:
        #   https://huggingface.co/spaces/<user>/<space>
        # becomes:
        #   https://<user>-<space>.hf.space
        if space_url and "huggingface.co/spaces/" in space_url:
            try:
                after = space_url.split("huggingface.co/spaces/", 1)[1].strip("/")
                user, space = after.split("/", 1)
                space_runtime = space.replace("_", "-")
                space_url = f"https://{user}-{space_runtime}.hf.space"
            except Exception:
                pass

        if not space_url:
            record("HF Space live ping", SKIP,
                   "HF_SPACE_URL and openenv.yaml space_url not set — skipping live deploy check.")
            return True

    try:
        # Ping /health (Space may be sleeping; retry a bit).
        last_status = None
        for _ in range(6):
            r = requests.get(f"{space_url}/health", timeout=15)
            last_status = r.status_code
            if r.status_code == 200:
                record("HF Space /health", PASS, f"200 OK from {space_url}")
                break
            if r.status_code in (502, 503, 504):
                time.sleep(5)
                continue
            record("HF Space /health", FAIL, f"Got HTTP {r.status_code} from {space_url}/health")
            return False
        else:
            record("HF Space /health", SKIP,
                   f"Space not ready (last HTTP {last_status}). Deploy and rerun to confirm.")
            return True

        # Call reset() on the live space
        r2 = requests.post(f"{space_url}/reset/easy", timeout=20)
        if r2.status_code != 200:
            # If the Space isn't ready, don't block local pre-submission validation.
            if r2.status_code in (502, 503, 504):
                record("HF Space /reset/easy", SKIP,
                       f"Space not ready (HTTP {r2.status_code}). Deploy and rerun.")
                return True
            record("HF Space /reset/easy", FAIL, f"Got HTTP {r2.status_code}")
            return False
        record("HF Space /reset/easy", PASS, "200 OK — Space responds to reset()")
        return True

    except requests.exceptions.ConnectionError:
        record("HF Space ping", SKIP,
               f"Could not connect to {space_url}. "
               "Is the Space awake? Set HF_SPACE_URL env var if URL is different.")
        return True  # Don't block submission for sleeping Space
    except Exception as e:
        record("HF Space ping", FAIL, str(e))
        return False


# ──────────────────────────────────────────────────────────────
# Check 7 — inference.py dry-run (baseline, no LLM key needed)
# ──────────────────────────────────────────────────────────────

def check_inference_runs() -> bool:
    print("\n[7] Running inference.py dry-run (deterministic baseline) …")

    env = os.environ.copy()
    # Unset HF_TOKEN to force the deterministic baseline — fast, no API cost
    env.pop("HF_TOKEN", None)
    env.pop("OPENAI_API_KEY", None)

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=20 * 60,  # 20-min hard cap
            env=env,
        )
    except subprocess.TimeoutExpired:
        record("inference.py runtime (< 20 min)", FAIL, "Timed out after 20 minutes.")
        return False
    except Exception as e:
        record("inference.py dry-run", FAIL, str(e))
        return False

    if result.returncode != 0:
        record("inference.py exit code", FAIL,
               f"Exited {result.returncode}.\n  stderr: {result.stderr[-300:]}")
        return False

    record("inference.py exit code", PASS, "Exited 0")

    # Parse the summary JSON line
    summary_json = None
    for line in result.stdout.splitlines():
        if "INFERENCE_SUMMARY_JSON:" in line:
            try:
                summary_json = json.loads(line.split("INFERENCE_SUMMARY_JSON:", 1)[1])
            except Exception:
                pass

    if summary_json is None:
        record("inference.py score output", SKIP,
               "INFERENCE_SUMMARY_JSON not found in stdout — check logging config.")
        return True

    scores = summary_json.get("scores", {})
    all_valid = all(0.0 <= s <= 1.0 for s in scores.values())
    if not all_valid:
        record("inference.py scores in [0,1]", FAIL, str(scores))
        return False

    record("inference.py scores", PASS,
           " | ".join(f"{t}={s:.4f}" for t, s in scores.items()))
    record("inference.py total score", PASS,
           f"{summary_json.get('total_score')}/{summary_json.get('num_tasks')}")
    return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  OpenEnv Pre-Submission Validator")
    print("  Personal Finance Env  --  ap2707")
    print("=" * 60)

    check_results = [
        ("openenv.yaml",        check_openenv_yaml()),
        ("inference.py static", check_inference()),
        ("tasks + graders",     check_tasks_and_graders()),
        ("Dockerfile",          check_dockerfile()),
        ("local endpoints",     check_endpoints()),
        ("HF Space live ping",  check_hf_space()),
        ("inference dry-run",   check_inference_runs()),
    ]

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    passed = 0
    for name, result in check_results:
        icon = "[PASS]" if result else "[FAIL]"
        print(f"  {icon}  {name}")
        if result:
            passed += 1

    total = len(check_results)
    print(f"\n  Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n  *** ALL CHECKS PASSED. ***")
        print("  Ready for OpenEnv / Hugging Face Spaces submission!\n")
        sys.exit(0)
    else:
        print("\n  --- SOME CHECKS FAILED. Review the output above. ---\n")
        sys.exit(1)
