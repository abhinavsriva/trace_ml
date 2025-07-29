# TraceML

A simple CLI tool to automatically trace PyTorch training memory usage.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![GitHub Stars](https://img.shields.io/github/stars/abhinavsriva/trace_ml?style=social)](https://github.com/abhinavsriva/trace_ml/stargazers)

---

## 💡 The Problem

Training large machine learning models often feels like a black box. One minute everything's running — the next, you're staring at a cryptic `"CUDA out of memory"` error.  

Pinpointing *which* part of the model is consuming too much memory or slowing things down is frustrating and time-consuming. Traditional profiling tools can be overly complex or lack the granularity deep learning developers need.

---

## ✨ Why TraceML?

`traceml` is a lightweight CLI tool to instrument your PyTorch training scripts and get real-time, granular insights into:

- System and process-level CPU & RAM usage  
- PyTorch layer-level memory allocation (via `gc` or decorator/instance tracing)

All shown live in your terminal — no config, no setup, just plug-and-trace.

---

## 📦 Installation

```bash
pip install -e .
```

---

## 🚀 Usage

```bash
traceml run <your_training_script.py>
```

TraceML wraps your training script and prints memory insights to the terminal as your model trains.


### 🔍 Examples

```bash
# Default: garbage collection-based tracing (no changes to code needed)
traceml run src/examples/tracing_with_gc

# Trace an explicitly defined model instance (e.g., functional API or Sequential model)
traceml run src/examples/tracing_with_model_instance

# Trace a model using a class decorator (recommended for structured training code)
traceml run src/examples/tracing_with_class_decorator
```


---

## ✅ Current Features

- 📊 **Live CPU & RAM usage** (System + Current Process)  
- 🔍 **PyTorch layer-level memory tracking**:
  - ✅ Default: via `gc` scanning (zero setup)
  - 🧠 Via `@trace_model` class decorator
  - 🔧 Via `trace_model_instance()` function for manual model instance tracing
- 🎮 **Live GPU memory & utilization tracking** (per device)
- 📦 **Model memory summaries** (per-layer + total)
- 🧾 **Historical snapshot viewer** with scrollable panel

---

## 🔭 Coming Soon

- 🎯 **Activation & gradient memory tracking**
- 📒 **Jupyter Notebook support**
- 💾 **Export logs as JSON / CSV**

---

## 🙌 Contribute & Feedback

Found it useful? Please provide ⭐ to the repo.  
Got ideas, feedback, or feature requests? I would love to hear from you.

📧 Email: [abhinavsriva@gmail.com](mailto:abhinavsriva@gmail.com)
