from pydantic import BaseModel, Field
from typing import List

class PayDebtAction(BaseModel):
    debt_name: str = Field(..., description="Name of the debt to pay towards.")
    amount: float = Field(..., description="Amount to pay towards the debt.")

class Action(BaseModel):
    pay_debts: List[PayDebtAction] = Field(default_factory=list, description="List of debts to pay this month.")
    save_amount: float = Field(default=0.0, description="Amount to transfer to savings this month.")
    discretionary_spend: float = Field(default=0.0, description="Amount spent on non-essentials this month.")

class Debt(BaseModel):
    name: str = Field(..., description="Name of the debt")
    balance: float = Field(..., description="Remaining balance on the debt")
    interest_rate: float = Field(..., description="Annual percentage rate (APR)")
    minimum_payment: float = Field(..., description="Required minimum monthly payment")

class Observation(BaseModel):
    month: int = Field(..., description="Current month number in the simulation (1 to max_months)")
    checking_balance: float = Field(..., description="Cash available in checking account")
    savings_balance: float = Field(..., description="Cash in savings account")
    monthly_income: float = Field(..., description="Income deposited this month")
    fixed_expenses_due: float = Field(..., description="Fixed expenses (rent, utilities) that were automatically deducted")
    debts: List[Debt] = Field(default_factory=list, description="Active debts")
    task_description: str = Field(..., description="Description of the goal to achieve")

class Reward(BaseModel):
    value: float = Field(..., description="Reward value for this step")
    reason: str = Field(..., description="Explanation of the reward")

class EnvInfo(BaseModel):
    task_name: str = Field(..., description="Current task ID")
    done: bool = Field(..., description="Is the episode done")
    score: float = Field(..., description="Current score between 0.0 and 1.0")
