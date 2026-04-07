"""Quick integration test for the deployed HF Space backend."""
import requests, json, sys

BASE = "https://ap2707-personal-finance-env.hf.space"

def test(label, method, path, body=None):
    url = f"{BASE}{path}"
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{method} {url}")
    try:
        if method == "GET":
            r = requests.get(url, timeout=15)
        else:
            r = requests.post(url, json=body, timeout=15)
        print(f"STATUS: {r.status_code}")
        print(f"RESPONSE: {json.dumps(r.json(), indent=2)}")
        return r.status_code, r.json()
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None

# 1. Root
test("Root endpoint", "GET", "/")

# 2. Health
test("Health check", "GET", "/health")

# 3. Tasks list
test("List tasks", "GET", "/tasks")

# 4. Reset easy task
code, data = test("Reset easy task", "POST", "/reset/easy")

# 5. Get state
test("Get state (easy)", "GET", "/state/easy")

# 6. Step - make a payment
action = {
    "pay_debts": [{"debt_name": "Credit Card", "amount": 200.0}],
    "save_amount": 0.0,
    "discretionary_spend": 0.0
}
test("Step easy task", "POST", "/step/easy", body=action)

# 7. Reset medium task  
test("Reset medium task", "POST", "/reset/medium")

# 8. Reset hard task
test("Reset hard task", "POST", "/reset/hard")

# 9. Test invalid task
test("Invalid task (should 404)", "POST", "/reset/nonexistent")

print(f"\n{'='*60}")
print("ALL TESTS COMPLETE")
