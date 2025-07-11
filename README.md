# Trace ML
A simple CLI to automatically trace PyTorch training memory. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/abhinavsriva/trace_ml?style=social)](https://github.com/abhinavsriva/trace_ml/stargazers)

## 💡 The Problem

Training large machine learning models often feels like a black box. Most of the time, traininig crashes with a cryptic "CUDA out of memory" error. Pinpointing *which* part of your model is consuming excessive memory or causing performance bottlenecks is a frustrating and time-consuming challenge. Traditional profiling tools can be complex or lack the granular detail needed for deep learning.

## ✨ Why Traceml?

`traceml` provides a simple, command-line interface to automatically instrument your PyTorch training runs, giving you actionable insights into resource utilization and memory allocation at a granular level.

## 📦 Installation

```bash
pip install traceml
