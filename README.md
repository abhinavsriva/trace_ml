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
- PyTorch layer-level memory allocation (via `gc`)

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

---

## ✅ Current Features

- 📊 **Live CPU & RAM usage** (System + Current Process)  
- 🔍 **PyTorch layer-level memory tracking** (via garbage collection)

---

## 🔭 Coming Soon

- 🧩 Layer memory tracing via decorators  
- 🎯 Activation & gradient memory usage  
- 🎮 Live **GPU memory and utilization tracking**  

---

## 🙌 Contribute & Feedback

Found it useful? Please provide ⭐ to the repo.  
Got ideas, feedback, or feature requests? I would love to hear from you.

📧 Email: [abhinavsriva@gmail.com](mailto:abhinavsriva@gmail.com)
