import streamlit as st
import json
import pandas as pd
import re
import plotly.graph_objects as go

# Load evaluation results
with open("evaluation_results_context_focus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Organize responses by question
questions = list(set([entry["question"] for entry in data]))
questions.sort()

st.set_page_config(page_title="LLM Evaluation: RAG vs ChatGPT", layout="wide")

st.title("📊 RAG vs. Pure ChatGPT – Climate Insights Dashboard")
st.markdown("**Case Study:** *Impact of Temperature Rise on Cotton Yield in Pakistan*")
st.markdown("Compare how RAG-augmented responses differ from standalone LLM output using real-world data.")

# Sidebar - select question
selected_question = st.sidebar.selectbox("Select a question to explore:", questions)

# Filter data
pure_entry = next(e for e in data if e["question"] == selected_question and e["response_type"] == "pure_chatgpt_response")
rag_entry = next(e for e in data if e["question"] == selected_question and e["response_type"] == "rag_response")

# --- Display Section ---
st.subheader("🧠 Question")
st.markdown(f"**{selected_question}**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💬 Pure ChatGPT Response")
    st.info(pure_entry["answer"], icon="🤖")

with col2:
    st.markdown("#### 🔎 RAG-Augmented Response")
    st.success(rag_entry["answer"], icon="📚")

# --- Scoring Breakdown ---
def extract_scores(raw_text):
    scores = {}
    for line in raw_text.strip().splitlines():
        match = re.match(r"(\w+):\s*(\d)", line.strip())
        if match:
            metric, score = match.groups()
            scores[metric] = int(score)
    return scores

pure_scores = extract_scores(pure_entry["evaluation_scores"])
rag_scores = extract_scores(rag_entry["evaluation_scores"])
metrics = list(pure_scores.keys())

# Plotting
fig = go.Figure()
fig.add_trace(go.Bar(x=metrics, y=[pure_scores[m] for m in metrics], name="Pure ChatGPT", marker_color="gray"))
fig.add_trace(go.Bar(x=metrics, y=[rag_scores[m] for m in metrics], name="RAG", marker_color="green"))

fig.update_layout(
    title="Evaluation Scores by Metric",
    yaxis=dict(range=[0, 5]),
    barmode="group",
    height=400,
    margin=dict(l=10, r=10, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

# --- Interpretation ---
st.markdown("### 📌 Insights")
if rag_scores.get("Use of Retrieved Content", 0) > pure_scores.get("Use of Retrieved Content", 0):
    st.success("✅ RAG successfully grounded its answer in retrieved content.")
else:
    st.warning("⚠️ Pure ChatGPT response may not rely on retrieved knowledge.")

st.markdown("Use this dashboard to demonstrate how retrieval grounding improves factual accuracy in climate data communication.")

st.markdown("---")
st.caption("Created using Streamlit, LangChain, FAISS, and OpenAI GPT.")
