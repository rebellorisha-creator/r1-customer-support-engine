import streamlit as st
from test import run_query

st.title(" RL-Based Customer Support Decision Engine")
st.caption("Intelligent system for routing customer queries using Reinforcement Learning")

user_input = st.text_input(" Enter customer query:")

if st.button("Analyze"):

    if user_input:
        result = run_query(user_input)

        state = result["state"]
        action = result["action"]
        reward = result["reward"]

        # Assuming state = (sentiment, urgency, complexity, _, action)
        sentiment = state[0]
        urgency = state[1]
        complexity = state[2]

        st.subheader("🔍 Analysis")
        st.write("**Query:**", user_input)

        # ----------------------------
        # FEATURES
        # ----------------------------
        st.subheader(" Extracted Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"Sentiment: {sentiment}")

        with col2:
            st.warning(f"Urgency: {urgency}")

        with col3:
            st.success(f"Complexity: {complexity}")

        # ----------------------------
        # DECISION
        # ----------------------------
        st.subheader("⚡ Decision")

        if action == "ESCALATE":
            st.error(" Escalated to higher support")
        elif action == "HUMAN":
            st.warning(" Assigned to human agent")
        elif action == "COMPENSATE":
            st.info(" Compensation offered")
        else:
            st.success(" Handled by AI")

        # ----------------------------
        # REASON
        # ----------------------------
        st.subheader(" Reason for Decision")

        reasons = []

        if sentiment == "angry":
            reasons.append("User shows negative sentiment")

        if urgency in ["medium", "high"]:
            reasons.append("Requires timely attention")

        if complexity == "complex":
            reasons.append("Issue is complex")

        if not reasons:
            reasons.append("Query is simple and can be handled automatically")

        for r in reasons:
            st.write(f"• {r}")

        # ----------------------------
        # REWARD ( FIXED POSITION)
        # ----------------------------
        st.subheader(" Reward Signal")

        st.metric("Reward Score", reward)

        if reward > 0.7:
            st.success("Good decision → Positive reward")
        elif reward > 0.4:
            st.warning("Acceptable decision → Moderate reward")
        else:
            st.error("Poor decision → Penalty applied")

        st.caption("Reward indicates how effective the decision is based on the learned RL policy")

    else:
        st.warning("Please enter a query.")