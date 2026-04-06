import streamlit as st
from test import run_query

st.title("🤖 RL Customer Support Triage System")

st.markdown("AI system that decides how to handle customer queries")

user_input = st.text_input("💬 Enter customer query:")

if st.button("Analyze"):

    if user_input:
        result = run_query(user_input)

        st.subheader("🔍 Analysis")

        st.write("**Query:**", user_input)
        st.write("**Final State:**", result["state"])
        st.write("**Decision:**", result["action"])
        st.write("**Reward:**", result["reward"])

        if result["action"] == "ESCALATE":
            st.error("🚨 Escalated to higher support")
        elif result["action"] == "HUMAN":
            st.warning("👨‍💻 Assigned to human agent")
        elif result["action"] == "COMPENSATE":
            st.info("💰 Compensation offered")
        else:
            st.success("🤖 Handled by AI")

    else:
        st.warning("Please enter a query.")