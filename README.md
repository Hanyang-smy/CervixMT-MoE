# CervixMT-MoE: Position-Aware Mixture-of-Experts for Multi-Task Cervical Cell Classification

This repository provides the official code and demonstration for the paper: **"CervixMT-MoE: Position-Aware Mixture-of-Experts for Multi-Task Cervical Cell Classification"**. 

## Overview

CervixMT-MoE is a multi-task classification framework built on a Position-Aware Mixture-of-Experts (PA-MoE) mechanism. It introduces a Dual-Path Position Module and a multi-source feature fusion head to jointly predict lesion categories and epithelial-layer information for cervical cells.

## Repository Contents

To facilitate reproducibility, this repository contains the following:

- `models.py`: The complete and detailed PyTorch implementation of the network architecture, including the Swin Transformer backbone, Dual-Path Position Module (DPPM), Position-Aware MoE (PA-MoE), and classification heads.
- `demo.py`: A fully runnable demonstration script. It generates random dummy inputs, initializes the model, and performs a forward pass to verify the tensor shapes and the correctness of the architecture logic.
- `requirements.txt`: The required Python libraries for running the demo.

## Quick Start (Running the Demo)

You can easily verify the architecture code by running the demo. 

1. **Install requirements:**
   ```bash
   pip install torch torchvision timm
