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
2. **Run the demonstration script:**
   ```bash
   python demo.py

 

Expected Output: The script will output the shape of the input tensor, the shape of the main classification logits, and the position classification logits, confirming the model is successfully constructed and executed.

## ⚠️ Data and Weights Availability (Ethical & IP Constraints)
To ensure transparency while strictly complying with ethical regulations and institutional policies, please note the following:

Dataset Privacy: The private HMUCervix dataset cannot be publicly released. This is due to strict patient privacy regulations and the Institutional Review Board (IRB) policies. Researchers from academic institutions may request access to the dataset for non-commercial purposes by signing a formal data-use agreement.

Pretrained Weights & Full Training Scripts: The fully trained model weights (.pth files) and the complete training pipeline (including detailed data augmentation scheduling and hyperparameter tuning scripts) are not publicly released. This is to protect the intellectual property (IP) of our laboratory's ongoing clinical research.

However, all the exact hyperparameters, data splits, and augmentation strategies used in our paper are explicitly described in the manuscript (Section IV.A) to facilitate independent replication by the community.
