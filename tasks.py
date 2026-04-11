from __future__ import annotations

from abc import ABC, abstractmethod

from models import Debt, Observation


class BaseTask(ABC):
    """Abstract base class for all personal-finance simulation tasks."""

    id: str
    max_months: int
    task_description: str

    @abstractmethod
    def get_initial_state(self) -> Observation:
        """Return the initial Observation for this task."""

    @abstractmethod
    def calculate_reward_and_done(
        self, prev_obs: Observation, current_obs: Observation
    ) -> tuple[float, str, bool, float]:
        """Compute (reward_value, reward_reason, done, score) after one step."""


# ---------------------------------------------------------------------------
# Easy Task
# ---------------------------------------------------------------------------

class EasyTask(BaseTask):
    """Pay off a single Credit Card within 4 months."""

    id = "easy"
    max_months = 4

    # Initial state constants
    _INITIAL_CHECKING = 1_000.0
    _INITIAL_SAVINGS = 0.0
    _MONTHLY_INCOME = 2_500.0
    _FIXED_EXPENSES = 1_500.0
    _CC_BALANCE = 1_200.0
    _CC_APR = 0.20
    _CC_MIN_PAYMENT = 50.0

    task_description = (
        "Pay off the 'Credit Card' debt completely within 4 months without letting "
        "your checking balance drop below zero. You receive your salary every month, "
        "and fixed expenses are automatically deducted."
    )

    def get_initial_state(self) -> Observation:
        return Observation(
            month=1,
            checking_balance=self._INITIAL_CHECKING,
            savings_balance=self._INITIAL_SAVINGS,
            monthly_income=self._MONTHLY_INCOME,
            fixed_expenses_due=self._FIXED_EXPENSES,
            debts=[
                Debt(
                    name="Credit Card",
                    balance=self._CC_BALANCE,
                    interest_rate=self._CC_APR,
                    minimum_payment=self._CC_MIN_PAYMENT,
                )
            ],
            task_description=self.task_description,
        )

    def calculate_reward_and_done(
        self, prev_obs: Observation, current_obs: Observation
    ) -> tuple[float, str, bool, float]:
        if current_obs.checking_balance < 0:
            return -10.0, "Checking balance dropped below zero. Bankruptcy.", True, 0.01

        cc_debt = next((d for d in current_obs.debts if d.name == "Credit Card"), None)
        cc_bal = cc_debt.balance if cc_debt else 0.0

        prev_cc = next((d for d in prev_obs.debts if d.name == "Credit Card"), None)
        prev_bal = prev_cc.balance if prev_cc else 0.0

        reward = 0.0
        reason = "Month advanced."
        done = False
        score = 0.01

        # Reward proportional to debt reduction
        if cc_bal < prev_bal:
            reward += (prev_bal - cc_bal) / 100.0
            reason += f" Paid down debt by ${prev_bal - cc_bal:.2f}."

        if cc_bal <= 0:
            reward += 10.0
            reason += " Successfully paid off Credit Card!"
            done = True
            score = 0.99
        elif current_obs.month >= self.max_months:
            done = True
            raw_score = max(0.0, (self._CC_BALANCE - cc_bal) / self._CC_BALANCE)
            score = max(0.01, min(0.99, raw_score))
            reason += f" Reached maximum months. Final debt: ${cc_bal:.2f}."
            if raw_score >= 0.99:
                reward += 10.0

        return reward, reason, done, score


# ---------------------------------------------------------------------------
# Medium Task
# ---------------------------------------------------------------------------

class MediumTask(BaseTask):
    """Build an emergency savings fund of at least $3 000 within 6 months."""

    id = "medium"
    max_months = 6

    # Initial state constants
    _INITIAL_CHECKING = 1_000.0
    _INITIAL_SAVINGS = 500.0
    _MONTHLY_INCOME = 4_000.0
    _FIXED_EXPENSES = 2_500.0
    _SAVINGS_GOAL = 3_000.0

    task_description = (
        "Build an emergency savings fund of at least $3,000 within 6 months. "
        "Maintain your checking balance above $0."
    )

    def get_initial_state(self) -> Observation:
        return Observation(
            month=1,
            checking_balance=self._INITIAL_CHECKING,
            savings_balance=self._INITIAL_SAVINGS,
            monthly_income=self._MONTHLY_INCOME,
            fixed_expenses_due=self._FIXED_EXPENSES,
            debts=[],
            task_description=self.task_description,
        )

    def calculate_reward_and_done(
        self, prev_obs: Observation, current_obs: Observation
    ) -> tuple[float, str, bool, float]:
        if current_obs.checking_balance < 0:
            return -10.0, "Checking balance dropped below zero.", True, 0.01

        reward = 0.0
        reason = "Month advanced."
        done = False
        score = 0.01

        if current_obs.savings_balance > prev_obs.savings_balance:
            saved = current_obs.savings_balance - prev_obs.savings_balance
            reward += saved / 100.0
            reason += f" Added ${saved:.2f} to savings."

        if current_obs.savings_balance >= self._SAVINGS_GOAL:
            reward += 10.0
            reason += " Successfully built emergency fund!"
            done = True
            score = 0.99
        elif current_obs.month >= self.max_months:
            done = True
            raw_score = min(1.0, current_obs.savings_balance / self._SAVINGS_GOAL)
            score = max(0.01, min(0.99, raw_score))
            reason += f" Reached maximum months. Final savings: ${current_obs.savings_balance:.2f}."
            if raw_score >= 0.99:
                reward += 10.0

        return reward, reason, done, score


# ---------------------------------------------------------------------------
# Hard Task
# ---------------------------------------------------------------------------

class HardTask(BaseTask):
    """Eliminate high-interest Credit Card debt AND build $1 000 savings in 12 months."""

    id = "hard"
    max_months = 12

    # Initial state constants
    _INITIAL_CHECKING = 1_500.0
    _INITIAL_SAVINGS = 0.0
    _MONTHLY_INCOME = 5_000.0
    _FIXED_EXPENSES = 3_000.0
    _CC_BALANCE = 4_000.0
    _CC_APR = 0.22
    _CC_MIN_PAYMENT = 120.0
    _CAR_BALANCE = 15_000.0
    _CAR_APR = 0.06
    _CAR_MIN_PAYMENT = 350.0
    _SAVINGS_GOAL = 1_000.0

    task_description = (
        "You have 12 months. Goal: Eliminate your 22% APR Credit Card debt "
        "(starts at $4,000) AND build up at least $1,000 in your savings account. "
        "Do not drop checking below $0."
    )

    def get_initial_state(self) -> Observation:
        return Observation(
            month=1,
            checking_balance=self._INITIAL_CHECKING,
            savings_balance=self._INITIAL_SAVINGS,
            monthly_income=self._MONTHLY_INCOME,
            fixed_expenses_due=self._FIXED_EXPENSES,
            debts=[
                Debt(
                    name="Credit Card",
                    balance=self._CC_BALANCE,
                    interest_rate=self._CC_APR,
                    minimum_payment=self._CC_MIN_PAYMENT,
                ),
                Debt(
                    name="Car Loan",
                    balance=self._CAR_BALANCE,
                    interest_rate=self._CAR_APR,
                    minimum_payment=self._CAR_MIN_PAYMENT,
                ),
            ],
            task_description=self.task_description,
        )

    def calculate_reward_and_done(
        self, prev_obs: Observation, current_obs: Observation
    ) -> tuple[float, str, bool, float]:
        if current_obs.checking_balance < 0:
            return -10.0, "Checking balance dropped below zero.", True, 0.01

        reward = 0.0
        reason = "Month advanced."
        done = False
        score = 0.01

        cur_cc = next((d for d in current_obs.debts if d.name == "Credit Card"), None)
        cur_cc_bal = cur_cc.balance if cur_cc else 0.0

        prev_cc = next((d for d in prev_obs.debts if d.name == "Credit Card"), None)
        prev_cc_bal = prev_cc.balance if prev_cc else 0.0

        # Incremental rewards
        if cur_cc_bal < prev_cc_bal:
            reward += (prev_cc_bal - cur_cc_bal) / 100.0

        if current_obs.savings_balance > prev_obs.savings_balance:
            reward += (current_obs.savings_balance - prev_obs.savings_balance) / 200.0

        if cur_cc is None and prev_cc is not None:
            reward += 5.0
            reason += " Paid off Credit Card!"

        # Terminal conditions
        if cur_cc_bal == 0 and current_obs.savings_balance >= self._SAVINGS_GOAL:
            done = True
            score = 0.99
            reward += 10.0
            reason += " Surpassed goal successfully!"
        elif current_obs.month >= self.max_months:
            done = True
            cc_score = max(0.0, (self._CC_BALANCE - cur_cc_bal) / self._CC_BALANCE)
            sav_score = min(1.0, current_obs.savings_balance / self._SAVINGS_GOAL)
            raw_score = (cc_score + sav_score) / 2.0
            score = max(0.01, min(0.99, raw_score))
            reason += f" Simulation ended. CC score: {cc_score:.2f}, Savings score: {sav_score:.2f}."

        return reward, reason, done, score


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: dict[str, BaseTask] = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask(),
}
