import copy
from typing import Tuple, Dict
from models import Observation, Action, Reward, EnvInfo, Debt
from tasks import TASKS

class PersonalFinanceEnv:
    def __init__(self, task_id: str = "easy"):
        """Initialize the Personal Finance Environment
        Args:
            task_id: One of 'easy', 'medium', 'hard'.
        """
        if task_id not in TASKS:
            raise ValueError(f"Task {task_id} not found.")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.current_state = None
        
    def reset(self) -> Observation:
        """Reset the environment to the initial state defined by the task."""
        self.current_state = self.task.get_initial_state()
        return self.current_state

    def state(self) -> Observation:
        """Return the current state (observation) of the environment."""
        if self.current_state is None:
            return self.reset()
        return self.current_state

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, EnvInfo]:
        """Apply an action to the environment, advancing it by one month.
        Args:
            action: And Action pydantic object.
        Returns:
            Observation, Reward, done (bool), and EnvInfo matching openenv spec ideas.
        """
        if self.current_state is None:
            self.reset()
            
        prev_obs = copy.deepcopy(self.current_state)
        new_obs = copy.deepcopy(self.current_state)
        
        # 1. Apply Action
        # Transfer to savings
        save_amt = max(0.0, action.save_amount)
        if save_amt > 0:
            new_obs.checking_balance -= save_amt
            new_obs.savings_balance += save_amt
            
        # Optional discretionary spend
        spend_amt = max(0.0, action.discretionary_spend)
        if spend_amt > 0:
            new_obs.checking_balance -= spend_amt

        # Pay debts
        for pay_action in action.pay_debts:
            pay_amt = max(0.0, pay_action.amount)
            if pay_amt > 0:
                # Find debt
                debt = next((d for d in new_obs.debts if d.name == pay_action.debt_name), None)
                if debt:
                    # You cannot pay more than what you owe
                    actual_pay = min(pay_amt, debt.balance)
                    debt.balance -= actual_pay
                    new_obs.checking_balance -= actual_pay

        # 2. End of Month accounting
        # Filter out fully paid debts
        new_obs.debts = [d for d in new_obs.debts if d.balance > 0]
        
        # Apply Interest to remaining debts
        for d in new_obs.debts:
             # Add monthly interest
             monthly_interest = d.balance * (d.interest_rate / 12.0)
             d.balance += monthly_interest
             
        # Add next month's salary & subtract next month's fixed expenses
        new_obs.month += 1
        new_obs.checking_balance += new_obs.monthly_income
        new_obs.checking_balance -= new_obs.fixed_expenses_due
        
        self.current_state = new_obs
        
        # 3. Calculate Reward & Done
        reward_val, reason, done, score = self.task.calculate_reward_and_done(prev_obs, new_obs)
        reward = Reward(value=reward_val, reason=reason)
        info = EnvInfo(task_name=self.task_id, done=done, score=score)
        
        return new_obs, reward, done, info
