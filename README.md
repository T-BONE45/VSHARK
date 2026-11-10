# vram-guard

> **Production-grade VRAM profiling, hardware auditing, and memory-bounded training/inference configuration injection for low-tier consumer GPUs.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red.svg)](https://pytorch.org/)

`vram-guard` is a command-line utility built to prevent Out-Of-Memory (OOM) crashes before a single line of training or inference code executes. It dynamically audits your local CUDA devices, profiles exact memory allocations (weights, gradients, optimizer states, and activations) based on architectural heuristics, and automatically injects safe fallbacks when your workload exceeds hardware thresholds.

---

## Key Features

* **Dynamic Hardware Profiling:** Automatically detects CUDA devices, total/free VRAM capacities, and compute capabilities across single or multi-GPU rigs using PyTorch.
* **Granular Memory-Bounding Model:** Computes memory footprints for parameter sizes (1B to 70B+), quantization tiers (`fp16`, `bf16`, `int8`, `int4`, `qlora`), sequence lengths, and batch sizes.
* **Automated OOM Safeguards & Fallbacks:** Iteratively negotiates sequence lengths, enforces QLoRA, or downscales micro-batch sizes if your target configuration threatens to crash your GPU.
* **Hugging Face Interoperability:** Instantly exports plug-and-play Python configuration snippets and `TrainingArguments` dictionaries tailored to your exact hardware limits.

---

## Installation

Clone or download the repository, then install it locally:

```bash
git clone [https://github.com/YOUR_USERNAME/vram-guard.git](https://github.com/YOUR_USERNAME/vram-guard.git)
cd vram-guard
pip install .
