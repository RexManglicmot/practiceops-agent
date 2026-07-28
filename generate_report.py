"""Runs the full PracticeOps Agents evaluation plus a few chained live-demo cases
against live Ollama, and rebuilds report.html from report_template.html. Not part
of the shipped app.py / agents.py / evaluate.py deliverable."""
import json
from pathlib import Path

from agents import OLLAMA_MODEL, action_agent, diagnostic_agent
from evaluate import precision_recall_f1, _labels

PROJECT_DIR = Path(__file__).parent

with open(PROJECT_DIR / "test_cases.json") as f:
    test_cases = json.load(f)

agent1_predicted, agent1_expected = [], []
agent2_predicted, agent2_expected = [], []
per_case = []

total_responses = 0
valid_json_count = 0
total_predicted_labels = 0
unsupported_count = 0

for case in test_cases:
    result1 = diagnostic_agent(case["metrics"])
    agent1_predicted.append(result1["issues"])
    agent1_expected.append(case["expected_issues"])

    result2 = action_agent(case["expected_issues"])
    agent2_predicted.append(result2["actions"])
    agent2_expected.append(case["expected_actions"])

    for result in (result1, result2):
        total_responses += 1
        valid_json_count += int(result["valid_json"])
        unsupported_count += result["unsupported_count"]
        total_predicted_labels += len(_labels(result)) + result["unsupported_count"]

    per_case.append({
        "practice_id": case["practice_id"],
        "specialty": case["metrics"]["specialty"],
        "metrics": case["metrics"],
        "expected_issues": case["expected_issues"],
        "predicted_issues": result1["issues"],
        "issues_match": set(result1["issues"]) == set(case["expected_issues"]),
        "expected_actions": case["expected_actions"],
        "predicted_actions": result2["actions"],
        "actions_match": set(result2["actions"]) == set(case["expected_actions"]),
    })

p1, r1, f1_1 = precision_recall_f1(agent1_predicted, agent1_expected)
p2, r2, f1_2 = precision_recall_f1(agent2_predicted, agent2_expected)
json_validity_rate = valid_json_count / total_responses if total_responses else 0.0
unsupported_label_rate = (
    unsupported_count / total_predicted_labels if total_predicted_labels else 0.0
)

# Live demo: full chained pipeline (Agent 1's real output feeds Agent 2), the
# same flow the Streamlit app uses, for a handful of illustrative practices.
demo_inputs = [
    {
        "label": "Spec worked example",
        "metrics": {
            "specialty": "Dermatology", "providers": 8, "monthly_appointments": 2400,
            "no_show_rate": 0.17, "portal_adoption_rate": 0.32,
            "average_check_in_minutes": 19, "weekly_phone_calls": 720,
            "claim_denial_rate": 0.06, "provider_utilization": 0.84,
        },
    },
    {"label": "Healthy practice", "metrics": next(c["metrics"] for c in test_cases if not c["expected_issues"])},
    {"label": "All issues present", "metrics": next(c["metrics"] for c in test_cases if len(c["expected_issues"]) == 6)},
]

live_demos = []
for demo in demo_inputs:
    d1 = diagnostic_agent(demo["metrics"])
    d2 = action_agent(d1["issues"])
    live_demos.append({
        "label": demo["label"],
        "metrics": demo["metrics"],
        "issues": d1["issues"],
        "actions": d2["actions"],
    })

report = {
    "summary": {
        "diagnostic": {"precision": p1, "recall": r1, "f1": f1_1},
        "action": {"precision": p2, "recall": r2, "f1": f1_2},
        "json_validity_rate": json_validity_rate,
        "unsupported_label_rate": unsupported_label_rate,
        "case_count": len(test_cases),
        "model": OLLAMA_MODEL,
    },
    "per_case": per_case,
    "live_demos": live_demos,
}

template = (PROJECT_DIR / "report_template.html").read_text()
json_str = json.dumps(report).replace("</script", "<\\/script")
html = template.replace("__REPORT_DATA_JSON__", json_str)
(PROJECT_DIR / "report.html").write_text(html)

print(f"Wrote {PROJECT_DIR / 'report.html'}")
print(json.dumps(report["summary"], indent=2))
