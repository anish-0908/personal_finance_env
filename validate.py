from __future__ import annotations

import os
import shutil
import subprocess

import requests
import yaml

from tasks import TASKS


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_openenv_yaml() -> bool:
    """Verify openenv.yaml exists and contains the required ``app_file`` key."""
    print("Checking openenv.yaml...")
    if not os.path.exists("openenv.yaml"):
        print("FAIL  openenv.yaml is missing.")
        return False

    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)

    if "app_file" not in data.get("spaces_deployment", {}):
        print("FAIL  'app_file' not found under 'spaces_deployment' in openenv.yaml.")
        return False

    print("PASS  openenv.yaml exists and is valid.")
    return True


def check_dockerfile() -> bool:
    """Attempt a local Docker build; skip gracefully when Docker is absent."""
    print("Checking Dockerfile build...")
    if not shutil.which("docker"):
        print("PASS  Skipped — Docker is not installed on this system.")
        return True

    try:
        subprocess.run(
            ["docker", "build", "-t", "pf_env_test", "."],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("PASS  Dockerfile built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL  Dockerfile build failed (exit code {e.returncode}).")
        return False
    except Exception as e:
        print(f"FAIL  Dockerfile build raised an unexpected error: {e}")
        return False


def check_endpoints() -> bool:
    """Ping the live FastAPI server and validate key endpoint responses."""
    print("Checking FastAPI endpoints...")
    base = "http://localhost:7860"

    try:
        # Root
        r = requests.get(f"{base}/", timeout=10)
        if r.status_code != 200:
            print(f"FAIL  GET / returned HTTP {r.status_code}.")
            return False

        # Reset
        r = requests.post(f"{base}/reset/easy", timeout=10)
        if r.status_code != 200:
            print(f"FAIL  POST /reset/easy returned HTTP {r.status_code}.")
            return False

        # Step
        payload = {"save_amount": 10, "discretionary_spend": 0, "pay_debts": []}
        r = requests.post(f"{base}/step/easy", json=payload, timeout=10)
        if r.status_code != 200:
            print(f"FAIL  POST /step/easy returned HTTP {r.status_code}.")
            return False

        print("PASS  /reset and /step endpoints returned 200.")
        return True

    except requests.ConnectionError:
        print("FAIL  Could not connect — is the FastAPI server running on port 7860?")
        return False
    except Exception as e:
        print(f"FAIL  Endpoint check raised an unexpected error: {e}")
        return False


def check_tasks_and_graders() -> bool:
    """Ensure at least 3 tasks exist and every grader returns a score in [0, 1]."""
    print("Checking tasks and graders...")

    if len(TASKS) < 3:
        print(f"FAIL  Only {len(TASKS)} task(s) found — need at least 3.")
        return False

    from env import PersonalFinanceEnv
    from models import Action

    for task_id in TASKS:
        env = PersonalFinanceEnv(task_id=task_id)
        env.reset()
        _, _, _, info = env.step(Action())
        if not (0.0 <= info.score <= 1.0):
            print(
                f"FAIL  Grader for '{task_id}' returned score {info.score:.4f} "
                "which is outside [0.0, 1.0]."
            )
            return False

    print(f"PASS  Found {len(TASKS)} tasks; all graders produce scores in [0.0, 1.0].")
    return True


def check_inference() -> bool:
    """Confirm inference.py is present and references the required env variables."""
    print("Checking inference.py...")

    if not os.path.exists("inference.py"):
        print("FAIL  inference.py is missing.")
        return False

    with open("inference.py") as f:
        content = f.read()

    required_vars = ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
    missing = [v for v in required_vars if v not in content]
    if missing:
        print(
            f"FAIL  inference.py is missing mandatory env-variable reference(s): "
            + ", ".join(missing)
        )
        return False

    print("PASS  inference.py exists and references all required environment variables.")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting pre-submission OpenEnv validation...\n")

    checks = [
        check_openenv_yaml,
        check_inference,
        check_tasks_and_graders,
        check_dockerfile,
        check_endpoints,
    ]

    results = [check() for check in checks]

    print()
    if all(results):
        print("ALL CHECKS PASSED — ready for OpenEnv / Hugging Face Spaces deployment!")
    else:
        failed = sum(1 for r in results if not r)
        print(f"{failed} CHECK(S) FAILED — please review the errors above.")
