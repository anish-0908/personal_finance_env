from __future__ import annotations

import json
import logging
import os

import jsonref
from openai import OpenAI
from pydantic import TypeAdapter

from env import PersonalFinanceEnv
from models import Action
from tasks import TASKS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_action_schema() -> dict:
    """Return a clean JSON Schema for :class:`~models.Action` (no ``$defs`` / cycles)."""
    adapter = TypeAdapter(Action)
    schema = jsonref.replace_refs(adapter.json_schema(), proxies=False)
    schema.pop("$defs", None)
    return schema


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_baseline() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set — skipping API calls.")
        return

    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"
    action_schema = get_action_schema()
    total_score = 0.0

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

    system_prompt = (
        "You are an AI financial advisor agent. "
        "Your goal is to complete the user's financial tasks by allocating funds appropriately. "
        "Analyze balances, debts, and income. "
        "Decide save_amount, pay_debts, and discretionary_spend. "
        "IMPORTANT: checking_balance must NEVER go below 0. "
        "save_amount + total pay_debts must not exceed the checking balance."
    )

    for task_id in TASKS:
        logger.info("--- Starting Task: %s ---", task_id.upper())

        env = PersonalFinanceEnv(task_id=task_id)
        obs = env.reset()

        done = False
        step = 0
        info = None

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        while not done:
            step += 1
            obs_json = obs.model_dump_json()
            messages.append({"role": "user", "content": f"Current State:\n{obs_json}"})

            logger.info(
                "Step %d: Month=%d, Checking=$%.2f, Savings=$%.2f",
                step, obs.month, obs.checking_balance, obs.savings_balance,
            )
            for debt in obs.debts:
                logger.info(
                    "  Debt %s: $%.2f (Min Pay: $%.2f)",
                    debt.name, debt.balance, debt.minimum_payment,
                )

            # LLM call
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "take_action"}},
                    temperature=0.0,
                )
            except Exception as e:
                logger.error("OpenAI API error: %s", e)
                break

            # Parse tool response
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and tool_calls[0].function.name == "take_action":
                raw = tool_calls[0].function.arguments
                try:
                    action_data = json.loads(raw) if isinstance(raw, str) else raw
                    action = Action(**action_data)
                except Exception as e:
                    logger.error("Failed to parse action '%s': %s — using default.", raw, e)
                    action = Action()
            else:
                logger.error("No valid tool call returned by model — using default action.")
                action = Action()

            logger.info("Action: %s", action.model_dump())
            obs, reward, done, info = env.step(action)

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

        score = info.score if info is not None else 0.0
        logger.info("Finished Task '%s'. Score: %.2f\n", task_id, score)
        total_score += score

    logger.info("Final Total Score: %.2f / %d", total_score, len(TASKS))


if __name__ == "__main__":
    run_baseline()