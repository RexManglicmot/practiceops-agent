import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")

ISSUE_LABELS = [
    "high_no_show",
    "low_portal_use",
    "long_check_in",
    "front_desk_overload",
    "billing_risk",
    "low_provider_utilization",
]

ACTION_LABELS = [
    "send_appointment_reminders",
    "promote_portal_enrollment",
    "enable_digital_intake",
    "review_front_desk_workflow",
    "review_claim_denials",
    "rebalance_provider_schedule",
]

ISSUE_TO_ACTIONS = {
    "high_no_show": "send_appointment_reminders",
    "low_portal_use": "promote_portal_enrollment",
    "long_check_in": "enable_digital_intake",
    "front_desk_overload": "review_front_desk_workflow",
    "billing_risk": "review_claim_denials",
    "low_provider_utilization": "rebalance_provider_schedule",
}

DIAGNOSTIC_SYSTEM_PROMPT = f"""You are the Diagnostic Agent in PracticeOps Agents, a system \
that analyzes medical practice operational metrics.

Analyze the practice metrics provided by the user and apply these rules exactly:
- no_show_rate >= 0.15                  -> high_no_show
- portal_adoption_rate < 0.40           -> low_portal_use
- average_check_in_minutes >= 15        -> long_check_in
- weekly_phone_calls >= 600             -> front_desk_overload
- claim_denial_rate >= 0.08             -> billing_risk
- provider_utilization < 0.70           -> low_provider_utilization

Select all applicable issues from this allowed list only: {ISSUE_LABELS}
Do not create new labels or invent labels not in the list.
Only assess operational workflow issues; never diagnose patients or reference clinical conditions.
Return valid JSON only, in exactly this shape: {{"issues": ["label1", "label2"]}}
If no rule matches, return {{"issues": []}}."""

ACTION_SYSTEM_PROMPT = f"""You are the Action Agent in PracticeOps Agents, a system that \
recommends operational actions for medical practices.

Review the detected operational issues provided by the user and apply this mapping exactly.
Every issue in the input maps to exactly one action below - include all of them, do not skip any:
{json.dumps(ISSUE_TO_ACTIONS, indent=2)}

Do not create new actions or invent actions not in the list.
Return valid JSON only, in exactly this shape: {{"actions": ["label1", "label2"]}}
If no issues were provided, return {{"actions": []}}."""


def call_ollama(system_prompt: str, user_prompt: str) -> tuple[dict | None, bool]:
    """Calls the local Ollama model. Returns (parsed_json_or_None, was_valid_json)."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": user_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw_text = response.json()["response"]
        return json.loads(raw_text), True
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None, False


def _split_supported(labels: list, allowed: list) -> tuple[list, int]:
    supported = [label for label in labels if label in allowed]
    unsupported_count = len(labels) - len(supported)
    return supported, unsupported_count


def diagnostic_agent(metrics: dict) -> dict:
    """Analyzes practice metrics and returns detected operational issues."""
    user_prompt = "Practice metrics:\n" + json.dumps(metrics, indent=2)
    parsed, valid_json = call_ollama(DIAGNOSTIC_SYSTEM_PROMPT, user_prompt)

    raw_issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
    issues, unsupported_count = _split_supported(raw_issues, ISSUE_LABELS)

    return {
        "issues": issues,
        "valid_json": valid_json,
        "unsupported_count": unsupported_count,
    }


def action_agent(issues: list) -> dict:
    """Selects operational actions for a list of detected issues."""
    if not issues:
        return {"actions": [], "valid_json": True, "unsupported_count": 0}

    user_prompt = "Detected issues:\n" + json.dumps({"issues": issues}, indent=2)
    parsed, valid_json = call_ollama(ACTION_SYSTEM_PROMPT, user_prompt)

    raw_actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
    actions, unsupported_count = _split_supported(raw_actions, ACTION_LABELS)

    return {
        "actions": actions,
        "valid_json": valid_json,
        "unsupported_count": unsupported_count,
    }
