from pydantic import BaseModel, Field


class PayDebtAction(BaseModel):
    """A single debt-payment instruction within an Action."""

    debt_name: str = Field(..., description="Name of the debt to pay towards.")
    amount: float = Field(..., ge=0.0, description="Amount to pay towards the debt.")


class Action(BaseModel):
    """Agent action for a single simulation month."""

    pay_debts: list[PayDebtAction] = Field(
        default_factory=list,
        description="List of debts to pay this month.",
    )
    save_amount: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount to transfer to savings this month.",
    )
    discretionary_spend: float = Field(
        default=0.0,
        ge=0.0,
        description="Amount spent on non-essentials this month.",
    )


class Debt(BaseModel):
    """A single debt instrument held by the user."""

    name: str = Field(..., description="Human-readable name of the debt.")
    balance: float = Field(..., ge=0.0, description="Remaining balance on the debt.")
    interest_rate: float = Field(..., ge=0.0, description="Annual percentage rate (APR) as a decimal (e.g. 0.20 = 20%).")
    minimum_payment: float = Field(..., ge=0.0, description="Required minimum monthly payment.")


class Observation(BaseModel):
    """Complete observable state of the environment at the start of a month."""

    month: int = Field(..., ge=1, description="Current month number in the simulation (1-indexed).")
    checking_balance: float = Field(..., description="Cash available in the checking account.")
    savings_balance: float = Field(..., ge=0.0, description="Cash in the savings account.")
    monthly_income: float = Field(..., ge=0.0, description="Gross income deposited each month.")
    fixed_expenses_due: float = Field(..., ge=0.0, description="Fixed expenses (rent, utilities) automatically deducted each month.")
    debts: list[Debt] = Field(default_factory=list, description="Active debts still carrying a balance.")
    task_description: str = Field(..., description="Natural-language description of the goal to achieve.")


class Reward(BaseModel):
    """Reward signal returned after each environment step."""

    value: float = Field(..., description="Scalar reward value for this step.")
    reason: str = Field(..., description="Human-readable explanation of how the reward was computed.")


class EnvInfo(BaseModel):
    """Auxiliary information returned alongside each environment step."""

    task_name: str = Field(..., description="ID of the current task (e.g. 'easy', 'medium', 'hard').")
    done: bool = Field(..., description="Whether the episode has ended.")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalised task score in [0.0, 1.0].")
