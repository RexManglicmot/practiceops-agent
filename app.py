import streamlit as st

from agents import action_agent, diagnostic_agent

ISSUE_DISPLAY = {
    "high_no_show": "High no-show rate",
    "low_portal_use": "Low portal adoption",
    "long_check_in": "Long check-in time",
    "front_desk_overload": "High front-desk workload",
    "billing_risk": "Elevated claim denial rate",
    "low_provider_utilization": "Low provider utilization",
}

ACTION_DISPLAY = {
    "send_appointment_reminders": "Increase automated appointment reminders",
    "promote_portal_enrollment": "Promote portal enrollment during check-in",
    "enable_digital_intake": "Introduce digital intake forms",
    "review_front_desk_workflow": "Review front-desk call routing",
    "review_claim_denials": "Review claim denial workflow",
    "rebalance_provider_schedule": "Rebalance provider schedule",
}

st.set_page_config(page_title="PracticeOps Agents", page_icon="\U0001f3e5")
st.title("PracticeOps Agents")
st.caption(
    "Two-agent LLM system that analyzes medical practice operational metrics and "
    "recommends controlled operational actions. Does not diagnose patients, process "
    "PHI, or make clinical decisions."
)

with st.form("metrics_form"):
    st.subheader("Practice metrics")
    col1, col2 = st.columns(2)
    with col1:
        specialty = st.text_input("Specialty", value="Dermatology")
        providers = st.number_input("Providers", min_value=1, value=8)
        monthly_appointments = st.number_input(
            "Monthly appointments", min_value=0, value=2400
        )
        no_show_rate = st.slider("No-show rate (%)", 0, 100, 17)
        portal_adoption_rate = st.slider("Portal adoption (%)", 0, 100, 32)
    with col2:
        average_check_in_minutes = st.number_input(
            "Average check-in time (minutes)", min_value=0, value=19
        )
        weekly_phone_calls = st.number_input(
            "Weekly phone calls", min_value=0, value=720
        )
        claim_denial_rate = st.slider("Claim denial rate (%)", 0, 100, 6)
        provider_utilization = st.slider("Provider utilization (%)", 0, 100, 84)

    submitted = st.form_submit_button("Run Analysis")

if submitted:
    metrics = {
        "specialty": specialty,
        "providers": providers,
        "monthly_appointments": monthly_appointments,
        "no_show_rate": no_show_rate / 100,
        "portal_adoption_rate": portal_adoption_rate / 100,
        "average_check_in_minutes": average_check_in_minutes,
        "weekly_phone_calls": weekly_phone_calls,
        "claim_denial_rate": claim_denial_rate / 100,
        "provider_utilization": provider_utilization / 100,
    }

    with st.spinner("Agent 1 (Diagnostic) analyzing metrics..."):
        diagnosis = diagnostic_agent(metrics)

    with st.spinner("Agent 2 (Action) selecting actions..."):
        recommendation = action_agent(diagnosis["issues"])

    st.divider()
    st.subheader("Practice status")
    if diagnosis["issues"]:
        st.error("Needs Attention")
    else:
        st.success("Healthy")

    st.subheader("Detected issues")
    if diagnosis["issues"]:
        for issue in diagnosis["issues"]:
            st.markdown(f"- {ISSUE_DISPLAY.get(issue, issue)}")
    else:
        st.markdown("No operational issues detected.")

    st.subheader("Recommended actions")
    if recommendation["actions"]:
        for action in recommendation["actions"]:
            st.markdown(f"- {ACTION_DISPLAY.get(action, action)}")
    else:
        st.markdown("No actions recommended.")

    st.divider()
    st.caption(
        f"Agent trace: Agent 1 identified {len(diagnosis['issues'])} operational "
        f"issue(s). Agent 2 selected {len(recommendation['actions'])} corresponding "
        "action(s)."
    )
    if not diagnosis["valid_json"] or not recommendation["valid_json"]:
        st.warning("One or more agent responses could not be parsed as valid JSON.")
