"""CompressRoute: Streamlit Interactive Model Selector & Benchmarking Dashboard.

Enables natural language constraint-aware routing, interactive Pareto trade-off
visualizations, live model inference execution, and SQLite-backed feedback collection.
"""

from typing import Dict, Any, List
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from selector.route import select, load_registry
from selector.constraint_parser import parse_constraints
from selector.pareto import compute_pareto_frontier
from app.database import init_db, save_feedback, get_feedback_records


st.set_page_config(
    page_title="CompressRoute | Intelligent Model Selector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .badge-pareto { background-color: #FEF3C7; color: #92400E; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-int8 { background-color: #DCFCE7; color: #166534; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-fp32 { background-color: #DBEAFE; color: #1E40AF; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_registry():
    """Loads and caches the registry."""
    return load_registry("benchmark/results/registry.json")


@st.cache_resource
def load_onnx_session(onnx_path: str):
    """Loads and caches ONNX Runtime inference session."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])


def run_live_inference(
    model_entry: Dict[str, Any],
    text: str,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Executes live single-utterance inference on user text if local models exist."""
    onnx_file = None
    name = model_entry.get("name", "")

    if "fp32" in name:
        if "teacher" in name:
            onnx_file = os.path.join(models_dir, "onnx", "teacher_bert_fp32.onnx")
        elif "baseline" in name:
            onnx_file = os.path.join(models_dir, "onnx", "baseline_distilbert_fp32.onnx")
        elif "student" in name:
            onnx_file = os.path.join(models_dir, "onnx", "student_distilled_fp32.onnx")
    elif "int8" in name:
        if "teacher" in name:
            onnx_file = os.path.join(models_dir, "onnx", "teacher_bert_int8.onnx")
        elif "baseline" in name:
            onnx_file = os.path.join(models_dir, "onnx", "baseline_distilbert_int8.onnx")
        elif "student" in name:
            onnx_file = os.path.join(models_dir, "onnx", "student_distilled_int8.onnx")

    # If ONNX file is not available locally (e.g., on cloud demo), return honest fallback without fake predictions
    if not (onnx_file and os.path.exists(onnx_file)):
        p50_lat = model_entry.get("latency_ms", {}).get("bs1", {}).get("p50", None)
        return {
            "intent_id": None,
            "confidence": None,
            "latency_ms": p50_lat,
            "is_live": False
        }

    tok_dir = os.path.join(models_dir, "teacher_bert")
    if "distilbert" in name:
        tok_dir = os.path.join(models_dir, "baseline_distilbert")
    elif "student" in name:
        tok_dir = os.path.join(models_dir, "student_distilled")

    # Load local tokenizer if available, otherwise fallback to standard base model
    if os.path.exists(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        base_tok = "distilbert-base-uncased" if "distilbert" in name else "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(base_tok)

    inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="np")
    sess = load_onnx_session(onnx_file)
    ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    t0 = time.perf_counter()
    logits = sess.run(["logits"], ort_inputs)[0]
    lat_ms = (time.perf_counter() - t0) * 1000.0
    pred_idx = int(np.argmax(logits, axis=-1)[0])
    exp_logits = np.exp(logits[0] - np.max(logits[0]))
    confidence = float(exp_logits[pred_idx] / np.sum(exp_logits))
    return {
        "intent_id": pred_idx,
        "confidence": confidence,
        "latency_ms": round(lat_ms, 2),
        "is_live": True
    }


# ==============================================================================
# UI HEADER
# ==============================================================================
st.markdown("<div class='main-title'>⚡ CompressRoute: Constraint-Aware NLP Model Selector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fine-tuned, distilled, and INT8 quantized models evaluated on real hardware — routed to fit your operational constraints.</div>", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("🎯 Routing Constraints & Preset")

preset_choice = st.sidebar.selectbox(
    "Optimization Goal",
    options=["balanced", "fastest", "max_accuracy", "edge"],
    format_func=lambda x: {
        "balanced": "⚖️ Balanced (Speed + Accuracy)",
        "fastest": "🚀 Fastest (Lowest Latency)",
        "max_accuracy": "🎯 Max Accuracy",
        "edge": "📱 Edge / Minimal Memory"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Constraint Overrides")

override_hw = st.sidebar.selectbox("Hardware", ["Auto-detect from text", "CPU", "Edge", "GPU"])
override_lat = st.sidebar.slider("Max Latency SLA (ms)", min_value=5, max_value=250, value=50, step=5)
enable_lat_cap = st.sidebar.checkbox("Enforce Latency Cap", value=False)

override_mem = st.sidebar.slider("Max Memory / Size (MB)", min_value=50, max_value=450, value=250, step=10)
enable_mem_cap = st.sidebar.checkbox("Enforce Memory Cap", value=False)

# Sample query buttons
st.sidebar.markdown("---")
st.sidebar.subheader("Quick-Fill Example Queries")
ex1 = st.sidebar.button("Banking Chatbot (< 20ms on CPU)")
ex2 = st.sidebar.button("Mobile Edge Assistant (< 130MB)")
ex3 = st.sidebar.button("Enterprise Banking (Max Accuracy)")

default_query = "bank chatbot, CPU only, under 20 ms"
if ex1:
    default_query = "bank chatbot, CPU only, under 20 ms"
elif ex2:
    default_query = "mobile edge banking assistant with < 130mb memory"
elif ex3:
    default_query = "bank chatbot on server CPU with max accuracy"

# Main Text Input
user_query = st.text_input("Describe your deployment scenario in plain English:", value=default_query)

# Apply overrides if enabled
active_query = user_query
if enable_lat_cap:
    active_query += f", under {override_lat} ms"
if enable_mem_cap:
    active_query += f", memory < {override_mem} MB"
if override_hw != "Auto-detect from text":
    active_query += f", {override_hw} only"

# Run Selector
results = select(active_query, preset=preset_choice)

# Tab Navigation
tab_recs, tab_charts, tab_try, tab_history = st.tabs([
    "🏆 Top Recommendations",
    "📊 Benchmarks & Pareto Trade-Off",
    "🧪 Live Inference ('Try It')",
    "💬 Feedback & History"
])

# ==============================================================================
# TAB 1: RECOMMENDATIONS
# ==============================================================================
with tab_recs:
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Detected Task", results["detected_task"].replace("_", " ").title(), f"{results['task_confidence']*100:.1f}% confidence")
    with col_info2:
        lat_bound = f"<= {results['constraints']['max_latency_ms']} ms" if results['constraints']['max_latency_ms'] else "None"
        st.metric("Latency SLA", lat_bound)
    with col_info3:
        mem_bound = f"<= {results['constraints']['max_memory_mb']} MB" if results['constraints']['max_memory_mb'] else "None"
        st.metric("Memory Limit", mem_bound)

    st.markdown("---")

    if results["status"] == "no_fit":
        st.error(f"❌ **No Viable Model Fits All Constraints**\n\n{results['explanation']}")
    else:
        st.subheader("Top 3 Recommended Models")
        for model in results["top_models"]:
            rank = model["rank"]
            pareto_html = "<span class='badge-pareto'>★ PARETO OPTIMAL</span>" if model.get("is_pareto") else ""
            prec_html = f"<span class='badge-int8'>{model['precision']}</span>" if model['precision'] == "INT8" else f"<span class='badge-fp32'>{model['precision']}</span>"

            st.markdown(f"""
            <div class='metric-card'>
                <h4>#{rank}. {model['display_name']} {pareto_html} {prec_html}</h4>
                <p><b>Rationale:</b> {model['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("p95 Latency (BS=1)", f"{model['latency_ms']['bs1']['p95']:.1f} ms")
            with m_col2:
                st.metric("Throughput", f"{model['throughput']['bs1']:.1f} smp/s")
            with m_col3:
                st.metric("Model Size", f"{model['size_mb']:.1f} MB")
            with m_col4:
                st.metric("Banking77 Accuracy", f"{model['accuracy']*100:.2f}%")

# ==============================================================================
# TAB 2: BENCHMARK CHARTS
# ==============================================================================
with tab_charts:
    st.subheader("Pareto Frontier & Performance Curves")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        tradeoff_path = "reports/tradeoff_accuracy_latency.png"
        if os.path.exists(tradeoff_path):
            st.image(tradeoff_path, caption="Accuracy vs. p95 Latency (Bubble size = Model Disk Size)")
    with chart_col2:
        scaling_path = "reports/batch_scaling_latency.png"
        if os.path.exists(scaling_path):
            st.image(scaling_path, caption="Batch Scaling Latency (ms) across Batch Sizes")

    thr_path = "reports/throughput_scaling.png"
    if os.path.exists(thr_path):
        st.image(thr_path, caption="Throughput (samples/sec) across Batch Sizes", use_column_width=True)

# ==============================================================================
# TAB 3: LIVE INFERENCE ("TRY IT")
# ==============================================================================
with tab_try:
    st.subheader("🧪 Test Live Intent Classification on Real Models")
    st.write("Type a customer query to see real-time inference latency and predicted class on your local CPU.")

    test_input = st.text_input("Customer Query:", "I think my card was compromised and I want to cancel it.")

    registry = get_cached_registry()
    model_choices = {m["display_name"]: m for m in registry}
    selected_name = st.selectbox("Select Model to Run:", list(model_choices.keys()))

    if st.button("🚀 Run Live Inference"):
        chosen_model = model_choices[selected_name]
        with st.spinner("Processing..."):
            res = run_live_inference(chosen_model, test_input)
            if res.get("is_live"):
                st.success(f"⚡ **Live Local CPU Execution** | **Predicted Intent ID:** `{res['intent_id']}` | **Confidence:** `{res['confidence']*100:.1f}%` | **Execution Latency:** `{res['latency_ms']} ms`")
            else:
                st.warning("⚠️ Live prediction is not available in this cloud demo because model files are not included. Run the app locally after training to try it. Benchmark numbers below are real measurements.")
                if res.get("latency_ms") is not None:
                    st.info(f"📊 **Measured Benchmark Latency (p50, batch size 1):** `{res['latency_ms']} ms`")

# ==============================================================================
# TAB 4: USER FEEDBACK
# ==============================================================================
with tab_history:
    st.subheader("💬 Recommendation Feedback")
    st.write("Help improve CompressRoute by rating the recommendations.")

    if results["status"] == "success" and results["top_models"]:
        recommended_model_name = results["top_models"][0]["display_name"]
        st.write(f"Current #1 Recommendation: **{recommended_model_name}**")

        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            rating_choice = st.radio("Rating:", ["👍 Good Recommendation", "👎 Poor Recommendation"], horizontal=True)
        with fb_col2:
            comment_text = st.text_input("Optional Notes / Context:")

        if st.button("Submit Feedback"):
            rating_val = 1 if "Good" in rating_choice else -1
            rec_id = save_feedback(active_query, recommended_model_name, rating_val, comment_text)
            st.success(f"Feedback saved to SQLite (Record #{rec_id})! Thank you.")

    st.markdown("---")
    st.subheader("Recent Feedback Log")
    records = get_feedback_records(limit=10)
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.info("No feedback entries recorded yet.")
