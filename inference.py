import os
import json
import time
import logging
import httpx
from openai import OpenAI

from env import PersonalFinanceEnv
from tasks import TASKS
from models import Action

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ──────────────────────────────────────────────
# Infra restriction: inference must finish < 20 min
# ──────────────────────────────────────────────
MAX_RUNTIME_SECONDS = 18 * 60  # 18 min hard cap (leaves 2 min buffer)
_SCRIPT_START = time.time()


def _elapsed() -> float:
    return time.time() - _SCRIPT_START


def _time_remaining() -> float:
    return max(0.0, MAX_RUNTIME_SECONDS - _elapsed())


def get_action_schema() -> dict:
    """Build a clean JSON Schema for Action without $defs / $ref cycles."""
    return {
        "type": "object",
        "title": "Action",
        "properties": {
            "pay_debts": {
                "type": "array",
                "description": "List of debts to pay this month.",
                "items": {
                    "type": "object",
                    "properties": {
                        "debt_name": {"type": "string", "description": "Name of the debt to pay towards."},
                        "amount": {"type": "number", "description": "Amount to pay towards the debt."}
                    },
                    "required": ["debt_name", "amount"]
                }
            },
            "save_amount": {
                "type": "number",
                "default": 0.0,
                "description": "Amount to transfer to savings this month."
            },
            "discretionary_spend": {
                "type": "number",
                "default": 0.0,
                "description": "Amount spent on non-essentials this month."
            }
        }
    }


def baseline_action(obs) -> Action:
    """
    Deterministic fallback policy — never overspends, always makes progress.
    Used when HF_TOKEN is missing/unusable so the script still completes.
    """
    checking = float(obs.checking_balance)
    budget = max(0.0, checking - 1.0)  # Keep $1 buffer

    pay_debts = []
    save_amount = 0.0

    if obs.debts:
        # Avalanche: pay highest-interest debt first
        debts_sorted = sorted(obs.debts, key=lambda d: float(d.interest_rate), reverse=True)
        target = debts_sorted[0]
        amt = min(budget, float(target.balance))
        if amt > 0:
            pay_debts = [{"debt_name": target.name, "amount": amt}]
            budget -= amt
    else:
        # No debts: save everything
        save_amount = budget

    return Action(pay_debts=pay_debts, save_amount=save_amount, discretionary_spend=0.0)


def run_inference():
    # ── Required environment variables (OpenEnv spec) ──
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    hf_token     = os.environ.get("HF_TOKEN")
    model_name   = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    # ── Build OpenAI client ──
    client = None
    if hf_token:
        http_client = httpx.Client()
        client = OpenAI(
            base_url=api_base_url,
            api_key=hf_token,
            http_client=http_client,
        )
        logging.info(f"Using LLM: {model_name} at {api_base_url}")
    else:
        logging.warning(
            "HF_TOKEN is not set — running deterministic baseline (no LLM calls). "
            "Set API_BASE_URL, MODEL_NAME, HF_TOKEN to enable LLM inference."
        )

    action_schema = get_action_schema()
    scores: dict[str, float] = {}

    for task_id in TASKS.keys():
        # ── Runtime guard ──
        remaining = _time_remaining()
        if remaining < 30:
            logging.error(f"Approaching 20-min runtime cap. Skipping task '{task_id}'.")
            scores[task_id] = 0.0
            continue

        logging.info(f"\n{'='*50}")
        logging.info(f"  Starting Task: {task_id.upper()}")
        logging.info(f"  Time elapsed: {_elapsed():.0f}s / {MAX_RUNTIME_SECONDS}s cap")
        logging.info(f"{'='*50}")

        env = PersonalFinanceEnv(task_id=task_id)
        obs = env.reset()

        done = False
        step = 0
        info = None

        system_prompt = (
            "You are an AI financial advisor agent. "
            "Your goal is to complete the user's financial tasks by allocating funds appropriately. "
            "Analyze balances, debts, and income each month. "
            "Decide save_amount, pay_debts, and discretionary_spend carefully. "
            "IMPORTANT: checking_balance must NEVER go below 0. "
            "Total of save_amount + all pay_debts amounts must not exceed the checking balance."
        )

        messages = [{"role": "system", "content": system_prompt}]

        tools = [{
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Take financial actions for the current month.",
                "parameters": action_schema,
            }
        }]

        while not done:
            step += 1

            # Per-step runtime guard
            if _time_remaining() < 30:
                logging.warning(f"Runtime cap hit at step {step} of task '{task_id}'. Stopping early.")
                break

            obs_json = obs.model_dump_json()
            messages.append({"role": "user", "content": f"Current State:\n{obs_json}"})

            logging.info(
                f"  Step {step:>2}: Month={obs.month:>2}, "
                f"Checking=${obs.checking_balance:>9.2f}, "
                f"Savings=${obs.savings_balance:>9.2f}"
            )

            # ── Choose action ──
            if client is None:
                action = baseline_action(obs)
            else:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "take_action"}},
                        temperature=0.0,
                    )
                except Exception as e:
                    logging.error(f"LLM API error: {e}. Falling back to baseline.")
                    action = baseline_action(obs)
                else:
                    tool_calls = response.choices[0].message.tool_calls
                    if tool_calls and tool_calls[0].function.name == "take_action":
                        arguments = tool_calls[0].function.arguments
                        try:
                            action_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                            action = Action(**action_data)
                        except Exception as e:
                            logging.error(f"Failed to parse action '{arguments}': {e}. Using baseline.")
                            action = baseline_action(obs)
                    else:
                        logging.error("Model returned no valid tool call. Using baseline.")
                        action = baseline_action(obs)

            logging.info(f"  Action: {action.model_dump()}")
            obs, reward, done, info = env.step(action)

            messages.append({
                "role": "assistant",
                "content": f"Executed action: {json.dumps(action.model_dump())}",
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Result:\n"
                    f"{json.dumps(reward.reason)}\n"
                    f"Reward: {reward.value}\n"
                    f"Done: {done}"
                ),
            })

        # ── Task score ──
        score = float(getattr(info, "score", 0.0)) if info is not None else 0.0
        score = max(0.0, min(1.0, score))
        scores[task_id] = score
        logging.info(f"  Task '{task_id}' finished. Score: {score:.4f}")

    # ── Final summary (machine-readable) ──
    total = sum(scores.values())
    avg   = total / len(scores) if scores else 0.0
    elapsed = _elapsed()

    summary = {
        "scores": scores,
        "total_score": round(total, 4),
        "average_score": round(avg, 4),
        "num_tasks": len(scores),
        "elapsed_seconds": round(elapsed, 1),
    }

    logging.info("\n" + "="*50)
    logging.info("  INFERENCE COMPLETE — FINAL SCORES")
    logging.info("="*50)
    for t, s in scores.items():
        logging.info(f"  {t:<10}: {s:.4f}")
    logging.info(f"  {'TOTAL':<10}: {total:.4f} / {len(scores)}")
    logging.info(f"  {'AVERAGE':<10}: {avg:.4f}")
    logging.info(f"  Elapsed   : {elapsed:.1f}s")
    logging.info("="*50)

    # Print JSON summary for programmatic consumption by validators
    print("\nINFERENCE_SUMMARY_JSON:" + json.dumps(summary))


if __name__ == "__main__":
    run_inference()
