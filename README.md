---
title: Personal Finance OpenEnv
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
---

# Personal Finance OpenEnv

A complete, real-world OpenEnv environment simulating Personal Finance Decisions. 

**Motivation:** Humans constantly make financial decisions: determining how much of their salary to allocate towards paying down various high-interest debts, versus building an emergency savings buffer. This environment tests an AI Agent's capacity to optimize those trade-offs over several months, accounting for fixed expenses, income, debt interest accrual, and cash flow constraints.

## Task Details

The environment implements three task levels, conforming to the `gymnasium`-style step/reset APIs expected by the OpenEnv interface. 

### Easy Task (`easy`)
- **Goal**: Pay off a small credit card debt within 4 months.
- **Rules**: Fixed salary and expenses. No other variables.
- **Difficulty**: Simple. Agent must make regular payments to the defined debt name to avoid accruing full interest.
- **Scoring**: $0-1.0$, penalized for failing to completely resolve the debt or missing the time window.

### Medium Task (`medium`)
- **Goal**: Build an emergency savings fund of $3000 within 6 months.
- **Rules**: Medium income, high fixed expenses. Requires balancing saving without dropping checking balance below 0. 
- **Difficulty**: Medium. Tests agent's ability to plan a multi-month savings strategy without going bankrupt.

### Hard Task (`hard`)
- **Goal**: Pay down as much of a 22% APR Credit Card ($\$4000$) as possible while building $\ge\$1000$ in savings over 12 months, and managing a separate low-interest car loan.
- **Rules**: Requires utilizing the "Avalanche method" (paying minimums on low-interest debt, dumping cash into high-interest debt), and satisfying the savings constraint.
- **Difficulty**: Hard.

## Observation and Action Spaces

This environment uses explicitly typed Pydantic models for both spaces as required by OpenEnv.

### Observation Space
A JSON representation specifying:
- `month`: Current month number in simulation
- `checking_balance`: Available liquid cash 
- `savings_balance`: Saved cash
- `monthly_income`: Income deposited
- `fixed_expenses_due`: Automatic deductions
- `debts`: Array of active debts (name, balance, interest_rate, minimum_payment)
- `task_description`: Human-readable task instruction

### Action Space
A JSON representation taking the following parameters:
- `pay_debts`: List of objects specifying `debt_name` and `amount`
- `save_amount`: Float amount to transfer to savings
- `discretionary_spend`: Float amount (optional) marking any non-essential spending.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo>
   cd personal_finance_env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the baseline script:**
   Ensure you have an OpenAI API Key set to run the baseline agent inference:
   ```bash
   export OPENAI_API_KEY="sk-..."
   python baseline.py
   ```

## Baseline Performance

The included `baseline.py` script leverages `gpt-4o-mini` with structured JSON tool-calling capabilities. 

**Reproducible Scores (`gpt-4o-mini`):**
- **Easy Task:** $1.00$ (Successfully paid off credit card within 4 months).
- **Medium Task:** $1.00$ (Successfully allocated surplus into savings to quickly hit emergency goal).
- **Hard Task:** $0.92$ (Successfully paid off credit card entirely and met the $1000 savings goal, sometimes missing perfect optimization due to step limits).

## HF Spaces Deployment
This package includes a `Dockerfile` and `app.py`. The hugging face space can be deployed directly via Docker, exposing an API endpoint at `:7860/state/{task_id}` for remote testing.
