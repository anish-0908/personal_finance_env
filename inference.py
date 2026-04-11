from __future__ import annotations

import json
import logging
import os
import time

import httpx
from openai import OpenAI

from env import PersonalFinanceEnv
from models import Action
from tasks import TASKS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime cap (OpenEnv infra: inference must finish within 20 minutes)
# ---------------------------------------------------------------------------

MAX_RUNTIME_SECONDS: float = 18 * 60  # 18 min — leaves a 2-min buffer
_SCRIPT_START: float = time.monotonic()  # monotonic avoids NTP jumps


def _elapsed() -> float:
    return time.monotonic() - _SCRIPT_START


def _time_remaining() -> float:
    return max(0.0, MAX_RUNTIME_SECONDS - _elapsed())


# ---------------------------------------------------------------------------
# Action schema (inlined to avoid $ref / $defs serialisation issues)
# ---------------------------------------------------------------------------

def get_action_schema() -> dict:
    """Return a minimal, self-contained JSON Schema for :class:`~models.Action`."""
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
                        "debt_name": {
                            "type": "string",
                            "description": "Name of the debt to pay towards.",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Amount to pay towards the debt.",
                        },
                    },
                    "required": ["debt_name", "amount"],
                },
            },
            "save_amount": {
                "type": "number",
                "default": 0.0,
                "description": "Amount to transfer to savings this month.",
            },
            "discretionary_spend": {
                "type": "number",
                "default": 0.0,
                "description": "Amount spent on non-essentials this month.",
            },
        },
    }


# ---------------------------------------------------------------------------
# Deterministic fallback policy
# ---------------------------------------------------------------------------

def baseline_action(obs) -> Action:
    """Avalanche-method fallback: never overspends, always makes progress.

    Used when ``HF_TOKEN`` is absent or an LLM call fails, ensuring the
    script always completes within the runtime cap.
    """
    checking = float(obs.checking_balance)
    budget = max(0.0, checking - 1.0)  # keep $1 buffer

    pay_debts: list[dict] = []
    save_amount = 0.0

    if obs.debts:
        # Avalanche: target highest-interest debt first
        target = max(obs.debts, key=lambda d: float(d.interest_rate))
        amt = min(budget, float(target.balance))
        if amt > 0:
            pay_debts = [{"debt_name": target.name, "amount": amt}]
            budget -= amt
    else:
        # No debts — save everything remaining
        save_amount = budget

    return Action(pay_debts=pay_debts, save_amount=save_amount, discretionary_spend=0.0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    # ── Required environment variables (OpenEnv spec) ──────────────────
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    hf_token = os.environ.get("HF_TOKEN")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    # ── Build OpenAI client ────────────────────────────────────────────
    client: OpenAI | None = None
    if hf_token:
        client = OpenAI(
            base_url=api_base_url,
            api_key=hf_token,
            http_client=httpx.Client(),
        )
        logger.info("Using LLM: %s at %s", model_name, api_base_url)
    else:
        logger.warning(
            "HF_TOKEN is not set — running deterministic baseline (no LLM calls). "
            "Set API_BASE_URL, MODEL_NAME, and HF_TOKEN to enable LLM inference."
        )

    action_schema = get_action_schema()
    scores: dict[str, float] = {}

    system_prompt = (
        "You are an AI financial advisor agent. "
        "Your goal is to complete the user's financial tasks by allocating funds appropriately. "
        "Analyze balances, debts, and income each month. "
        "Decide save_amount, pay_debts, and discretionary_spend carefully. "
        "IMPORTANT: checking_balance must NEVER go below 0. "
        "Total of save_amount + all pay_debts amounts must not exceed the checking balance."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Take financial actions for the current month.",
                "parameters": action_schema,
            },
        }
    ]

    for task_id in TASKS:
        # ── Runtime guard (per-task) ───────────────────────────────────
        remaining = _time_remaining()
        if remaining < 30:
            logger.error(
                "Approaching the 20-min runtime cap — skipping task '%s'.", task_id
            )
            scores[task_id] = 0.0
            continue

        logger.info("\n%s", "=" * 50)
        logger.info("  Starting Task: %s", task_id.upper())
        logger.info("  Elapsed: %.0fs / %.0fs cap", _elapsed(), MAX_RUNTIME_SECONDS)
        logger.info("%s", "=" * 50)
        print(f"[START] task={task_id}", flush=True)

        env = PersonalFinanceEnv(task_id=task_id)
        obs = env.reset()

        done = False
        step = 0
        info = None
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        while not done:
            step += 1

            # Per-step runtime guard
            if _time_remaining() < 30:
                logger.warning(
                    "Runtime cap hit at step %d of task '%s' — stopping early.",
                    step, task_id,
                )
                break

            obs_json = obs.model_dump_json()
            messages.append({"role": "user", "content": f"Current State:\n{obs_json}"})

            logger.info(
                "  Step %2d: Month=%2d, Checking=%10.2f, Savings=%10.2f",
                step, obs.month, obs.checking_balance, obs.savings_balance,
            )

            # ── Choose action ──────────────────────────────────────────
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
                    logger.error("LLM API error: %s — falling back to baseline.", e)
                    action = baseline_action(obs)
                else:
                    tool_calls = response.choices[0].message.tool_calls
                    if tool_calls and tool_calls[0].function.name == "take_action":
                        raw = tool_calls[0].function.arguments
                        try:
                            action_data = json.loads(raw) if isinstance(raw, str) else raw
                            action = Action(**action_data)
                        except Exception as e:
                            logger.error(
                                "Failed to parse action '%s': %s — using baseline.", raw, e
                            )
                            action = baseline_action(obs)
                    else:
                        logger.error("No valid tool call returned — using baseline.")
                        action = baseline_action(obs)

            logger.info("  Action: %s", action.model_dump())
            obs, reward, done, info = env.step(action)
            print(f"[STEP] step={step} reward={reward.value}", flush=True)

            messages.append({
                "role": "assistant",
                "content": f"Executed action: {json.dumps(action.model_dump())}",
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Result:\n{json.dumps(reward.reason)}\n"
                    f"Reward: {reward.value}\n"
                    f"Done: {done}"
                ),
            })

        # ── Record task score ──────────────────────────────────────────
        score = float(getattr(info, "score", 0.0)) if info is not None else 0.0
        score = max(0.0, min(1.0, score))
        scores[task_id] = score
        logger.info("  Task '%s' finished. Score: %.4f", task_id, score)
        print(f"[END] task={task_id} score={score} steps={step}", flush=True)

    # ── Final summary ──────────────────────────────────────────────────
    total = sum(scores.values())
    avg = total / len(scores) if scores else 0.0
    elapsed = _elapsed()

    summary = {
        "scores": scores,
        "total_score": round(total, 4),
        "average_score": round(avg, 4),
        "num_tasks": len(scores),
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info("\n%s", "=" * 50)
    logger.info("  INFERENCE COMPLETE — FINAL SCORES")
    logger.info("%s", "=" * 50)
    for t, s in scores.items():
        logger.info("  %-10s: %.4f", t, s)
    logger.info("  %-10s: %.4f / %d", "TOTAL", total, len(scores))
    logger.info("  %-10s: %.4f", "AVERAGE", avg)
    logger.info("  %-10s: %.1fs", "Elapsed", elapsed)
    logger.info("%s", "=" * 50)

    # Emit machine-readable JSON for programmatic validators
    print(f"\nINFERENCE_SUMMARY_JSON:{json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    run_inference()