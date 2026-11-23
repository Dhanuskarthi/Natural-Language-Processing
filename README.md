# CompressRoute: Model Compression & Natural Language Intent Routing

CompressRoute is an end-to-end NLP engineering framework designed to compress transformer models and recommend the optimal model variant based on user constraints (e.g. latency, memory, hardware budget, and accuracy requirements).

---

## 🎯 Architecture & Overview

CompressRoute combines two core paradigms:

1. **Model Compression & Optimization**:
   - **Teacher Model**: Fine-tuned `bert-base-uncased` on the 77-class Banking77 intent detection dataset.
   - **Baseline**: Fine-tuned `distilbert-base-uncased`.
   - **Knowledge Distillation**: Custom loss function ($\mathcal{L} = \alpha \mathcal{L}_{CE} + (1-\alpha) T^2 \mathcal{L}_{KL}$) training a lighter student model (4–6 layers) initialized from teacher weights.
   - **Quantization & Graph Optimization**: Export models to ONNX FP32 and apply INT8 dynamic/static quantization using ONNX Runtime.
   - **Comprehensive Benchmarking**: Hardware-specific profiling across batch sizes (1, 8, 32, 64) and sequence lengths (32, 128) recording p50/p95/p99 latency, throughput, memory footprint, and test accuracy in a standardized registry (`benchmark/results/registry.json`).

2. **Intelligent NLP Model Selector**:
   - **Task Classification**: Identifies user NLP tasks (intent classification, sentiment analysis, summarization, QA, translation) from plain English prompts.
   - **Constraint Parsing**: Extracts latency bounds (e.g., "< 20ms"), memory restrictions, and hardware targets (CPU, GPU, Edge) using rule-based and regex parsing.
   - **Multi-Objective Scoring & Pareto Frontier**: Ranks compatible candidates using Pareto-optimality and customizable weighting presets (*Max Accuracy*, *Balanced*, *Fastest*, *Edge*).
   - **Interactive Streamlit Interface**: Real-time evaluation, trade-off visualizations, side-by-side metric comparison, and live interactive model inference.

---

## 📁 Repository Layout

```text
Natural-Language-Processing/
├── config.yaml              # Central configuration (hyperparameters, paths, weights)
├── requirements.txt         # Pinned project dependencies
├── README.md                # Project documentation and reproduction guide
├── .gitignore               # Strict exclusion of heavy weights, caches, and datasets
├── data/                    # Dataset loading, inspection, and fixed split logic
├── compress/                # Fine-tuning, distillation, ONNX export, and quantization
│   ├── data_utils.py        # Banking77 loading, stats, and deterministic splits
│   ├── finetune_teacher.py  # Teacher (BERT) and baseline (DistilBERT) training
│   ├── distill.py           # Custom KD trainer with temperature & alpha scaling
│   ├── export_onnx.py       # ONNX FP32 graph export and parity verification
│   └── quantize.py          # INT8 dynamic and static calibration
├── benchmark/               # Benchmarking suite & model registry
│   ├── metrics.py           # Accuracy, Macro-F1, latency, throughput, memory profiler
│   ├── run_benchmark.py     # Multi-batch / multi-length execution harness
│   └── results/             # registry.json, trade-off charts, latency curves
├── selector/                # Task detection, constraint extraction, and ranking
│   ├── task_classifier.py   # NLP task detector
│   ├── constraint_parser.py # Regex/rule engine for SLA and hardware bounds
│   ├── scoring.py           # Multi-objective normalized scoring engine
│   ├── pareto.py            # Non-dominated Pareto frontier extraction
│   └── explain.py           # Natural language justification generator
├── app/                     # Interactive UI
│   └── streamlit_app.py     # Streamlit web application with SQLite feedback store
├── tests/                   # Automated unit and integration tests
├── notebooks/               # Colab / Kaggle reproducible training notebooks
└── reports/                 # Comprehensive final technical report
```

---

## 🚀 Quickstart & Setup

### 1. Environment Installation
Ensure Python 3.10+ is installed (Python 3.11 recommended):

```bash
# Clone the repository
git clone https://github.com/Dhanuskarthi/Natural-Language-Processing.git
cd Natural-Language-Processing

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verification
Check that key runtimes are ready:
```bash
python -c "import torch, transformers, onnxruntime; print('Environment OK!')"
```

### 3. CPU Fast Testing Mode
All training scripts support the `--fast` flag, which executes on a small stratified subset with 1 epoch for quick end-to-end validation without a GPU.
