from __future__ import annotations

import copy
from typing import Tuple

from models import Action, Debt, EnvInfo, Observation, Reward  # noqa: F401 (Debt re-exported)
from tasks import TASKS


class PersonalFinanceEnv:
    """Gym-style environment for personal finance decision-making simulations.

    A single episode corresponds to one task (easy / medium / hard).  Each
    ``step`` advances the simulation by one calendar month.

    Attributes:
        task_id: Identifier of the active task.
        task:    The :class:`~tasks.BaseTask` instance driving reward logic.
        current_state: The most-recent :class:`~models.Observation`, or
            ``None`` before the first ``reset()``.
    """

    def __init__(self, task_id: str = "easy") -> None:
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Valid options: {list(TASKS.keys())}"
            )
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.current_state: Observation | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to the task's initial state.

        Returns:
            The initial :class:`~models.Observation`.
        """
        self.current_state = self.task.get_initial_state()
        return self.current_state

    def state(self) -> Observation:
        """Return the current observation (auto-resets if never initialised).

        Returns:
            The current :class:`~models.Observation`.
        """
        if self.current_state is None:
            return self.reset()
        return self.current_state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, EnvInfo]:
        """Apply *action* and advance the simulation by one month.

        Args:
            action: An :class:`~models.Action` specifying how to allocate funds.

        Returns:
            A 4-tuple of ``(observation, reward, done, info)``.
        """
        if self.current_state is None:
            self.reset()

        prev_obs = copy.deepcopy(self.current_state)
        new_obs = copy.deepcopy(self.current_state)

        # ── 1. Apply agent action ──────────────────────────────────────
        # Transfer to savings
        save_amt = max(0.0, action.save_amount)
        if save_amt > 0:
            new_obs.checking_balance -= save_amt
            new_obs.savings_balance += save_amt

        # Discretionary spending
        spend_amt = max(0.0, action.discretionary_spend)
        if spend_amt > 0:
            new_obs.checking_balance -= spend_amt

        # Debt payments
        for pay_action in action.pay_debts:
            pay_amt = max(0.0, pay_action.amount)
            if pay_amt > 0:
                debt = next(
                    (d for d in new_obs.debts if d.name == pay_action.debt_name),
                    None,
                )
                if debt:
                    actual_pay = min(pay_amt, debt.balance)  # cannot overpay
                    debt.balance -= actual_pay
                    new_obs.checking_balance -= actual_pay

        # ── 2. End-of-month accounting ─────────────────────────────────
        # Remove fully-paid debts
        new_obs.debts = [d for d in new_obs.debts if d.balance > 0]

        # Accrue monthly interest on remaining debts
        for debt in new_obs.debts:
            monthly_interest = debt.balance * (debt.interest_rate / 12.0)
            debt.balance += monthly_interest

        # Receive next month's income and pay fixed expenses
        new_obs.month += 1
        new_obs.checking_balance += new_obs.monthly_income
        new_obs.checking_balance -= new_obs.fixed_expenses_due

        self.current_state = new_obs

        # ── 3. Reward & termination ────────────────────────────────────
        reward_val, reason, done, score = self.task.calculate_reward_and_done(
            prev_obs, new_obs
        )
        reward = Reward(value=reward_val, reason=reason)
        info = EnvInfo(task_name=self.task_id, done=done, score=score)

        return new_obs, reward, done, info
