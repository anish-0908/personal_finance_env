import os
import json
import jsonref
import logging
from openai import OpenAI
from pydantic import TypeAdapter
from env import PersonalFinanceEnv
from tasks import TASKS
from models import Action

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_action_schema():
    # Helper to generate standard JSON schema for function calling / structured outputs
    adapter = TypeAdapter(Action)
    schema = adapter.json_schema()
    # Resolve any $refs without using Proxy dictionaries to avoid JSON serialization errors
    resolved = jsonref.replace_refs(schema, proxies=False)
    # the resolved dictionary may still contain $defs which some endpoints reject
    if "$defs" in resolved:
        del resolved["$defs"]
    return resolved


def run_baseline():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.warning("No OPENAI_API_KEY found. Skipping API calls.")
        return

    client = OpenAI(api_key=api_key)

    model = "gpt-4o-mini"
    action_schema = get_action_schema()

    total_score = 0.0

    for task_id in TASKS.keys():
        logging.info(f"--- Starting Task: {task_id.upper()} ---")

        env = PersonalFinanceEnv(task_id=task_id)
        obs = env.reset()

        done = False
        step = 0
        info = None  # ✅ FIX: prevent crash

        system_prompt = (
            "You are an AI financial advisor agent. "
            "Your goal is to complete the user's financial tasks by allocating funds appropriately. "
            "Analyze balances, debts, and income. "
            "Decide save_amount, pay_debts, discretionary_spend. "
            "IMPORTANT: checking_balance must NEVER go below 0. "
            "save_amount + pay_debts must not exceed checking balance."
        )

        messages = [{"role": "system", "content": system_prompt}]

        tools = [{
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Take financial actions for the current month.",
                "parameters": action_schema
            }
        }]

        while not done:
            step += 1

            obs_json = obs.model_dump_json()
            messages.append({
                "role": "user",
                "content": f"Current State:\n{obs_json}"
            })

            logging.info(
                f"Step {step}: Month={obs.month}, "
                f"Checking=${obs.checking_balance:.2f}, "
                f"Savings=${obs.savings_balance:.2f}"
            )

            for d in obs.debts:
                logging.info(
                    f"  Debt {d.name}: ${d.balance:.2f} "
                    f"(Min Pay: ${d.minimum_payment:.2f})"
                )

            # 🔥 OpenAI Call
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "take_action"}},
                    temperature=0.0
                )
            except Exception as e:
                logging.error(f"OpenAI API Error: {e}")
                break

            # 🔥 Parse tool response safely
            tool_calls = response.choices[0].message.tool_calls

            if tool_calls and tool_calls[0].function.name == "take_action":
                arguments = tool_calls[0].function.arguments

                try:
                    # ✅ FIX: handle both string & dict
                    if isinstance(arguments, str):
                        action_data = json.loads(arguments)
                    else:
                        action_data = arguments

                    action = Action(**action_data)

                except Exception as e:
                    logging.error(f"Failed to parse Action output: {arguments}. Error: {e}")
                    action = Action()  # fallback

            else:
                logging.error("No valid action returned by model.")
                action = Action()

            # 🔥 Apply action
            logging.info(f"Action: {action.model_dump()}")

            obs, reward, done, info = env.step(action)

            # ✅ FIX: JSON-safe messages
            messages.append({
                "role": "assistant",
                "content": f"Executed action: {json.dumps(action.model_dump())}"
            })

            messages.append({
                "role": "user",
                "content": (
                    f"Result:\n"
                    f"{json.dumps(reward.reason)}\n"
                    f"Reward: {reward.value}\n"
                    f"Done: {done}"
                )
            })

        # ✅ FIX: safe info usage
        if info is not None:
            logging.info(f"Finished Task {task_id}. Score: {info.score:.2f}\n")
            total_score += info.score
        else:
            logging.info(f"Finished Task {task_id}. Score: 0.00 (Error)\n")

    logging.info(f"Final Total Score: {total_score:.2f} / {len(TASKS)}")


if __name__ == "__main__":
    run_baseline()