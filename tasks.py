from models import Observation, Debt

class BaseTask:
    id: str
    max_months: int
    task_description: str
    
    def get_initial_state(self) -> Observation:
        raise NotImplementedError
        
    def calculate_reward_and_done(self, prev_obs: Observation, current_obs: Observation) -> tuple[float, str, bool, float]:
        # Returns reward_value, reward_reason, done, score
        raise NotImplementedError

class EasyTask(BaseTask):
    id = "easy"
    max_months = 4
    task_description = "Pay off the 'Credit Card' debt completely within 4 months without letting your checking balance drop below zero. You receive your salary every month, and fixed expenses are automatically deducted."
    
    def get_initial_state(self) -> Observation:
        return Observation(
            month=1,
            checking_balance=1000.0,
            savings_balance=0.0,
            monthly_income=2500.0,
            fixed_expenses_due=1500.0,
            debts=[
                Debt(name="Credit Card", balance=1200.0, interest_rate=0.20, minimum_payment=50.0)
            ],
            task_description=self.task_description
        )
        
    def calculate_reward_and_done(self, prev_obs: Observation, current_obs: Observation) -> tuple[float, str, bool, float]:
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
        
        # Reward for paying down debt
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
            raw_score = max(0.0, (1200.0 - cc_bal) / 1200.0)
            score = max(0.01, min(0.99, raw_score))
            reason += f" Reached maximum months. Final debt: ${cc_bal:.2f}."
            if raw_score >= 0.99:
                reward += 10.0
            
        return reward, reason, done, score

class MediumTask(BaseTask):
    id = "medium"
    max_months = 6
    task_description = "Build an emergency savings fund of at least $3000 within 6 months. Maintain your checking balance above 0."
    
    def get_initial_state(self) -> Observation:
        return Observation(
            month=1,
            checking_balance=1000.0,
            savings_balance=500.0,
            monthly_income=4000.0,
            fixed_expenses_due=2500.0,
            debts=[],
            task_description=self.task_description
        )

    def calculate_reward_and_done(self, prev_obs: Observation, current_obs: Observation) -> tuple[float, str, bool, float]:
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
            
        if current_obs.savings_balance >= 3000.0:
            reward += 10.0
            reason += " Successfully built emergency fund!"
            done = True
            score = 0.99
        elif current_obs.month >= self.max_months:
            done = True
            raw_score = min(1.0, current_obs.savings_balance / 3000.0)
            score = max(0.01, min(0.99, raw_score))
            reason += f" Reached maximum months. Final savings: ${current_obs.savings_balance:.2f}."
            if raw_score >= 0.99:
                 reward += 10.0
                 
        return reward, reason, done, score

class HardTask(BaseTask):
    id = "hard"
    max_months = 12
    task_description = "You have 12 months. Goal: Eliminate your 22% APR Credit Card debt (starts at $4000) AND build up at least $1000 in your savings account. Do not drop checking below $0."
    
    def get_initial_state(self) -> Observation:
        return Observation(
            month=1,
            checking_balance=1500.0,
            savings_balance=0.0,
            monthly_income=5000.0,
            fixed_expenses_due=3000.0,
            debts=[
                Debt(name="Credit Card", balance=4000.0, interest_rate=0.22, minimum_payment=120.0),
                Debt(name="Car Loan", balance=15000.0, interest_rate=0.06, minimum_payment=350.0)
            ],
            task_description=self.task_description
        )

    def calculate_reward_and_done(self, prev_obs: Observation, current_obs: Observation) -> tuple[float, str, bool, float]:
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
        
        if cur_cc_bal < prev_cc_bal:
             reward += (prev_cc_bal - cur_cc_bal) / 100.0
             
        if current_obs.savings_balance > prev_obs.savings_balance:
             reward += (current_obs.savings_balance - prev_obs.savings_balance) / 200.0
        
        if cur_cc is None and prev_cc is not None:
             reward += 5.0
             reason += " Paid off Credit Card!"
             
        if cur_cc_bal == 0 and current_obs.savings_balance >= 1000.0:
             done = True
             score = 0.99
             reward += 10.0
             reason += " Surpassed goal successfully!"
        elif current_obs.month >= self.max_months:
            done = True
            cc_score = max(0.0, (4000.0 - cur_cc_bal) / 4000.0)
            sav_score = min(1.0, current_obs.savings_balance / 1000.0)
            raw_score = (cc_score + sav_score) / 2.0
            score = max(0.01, min(0.99, raw_score))
            reason += f" Simulation ended. CC score: {cc_score:.2f}, Savings score: {sav_score:.2f}"
            
        return reward, reason, done, score

TASKS = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask()
}
