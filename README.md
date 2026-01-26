# VASP-Agent Project Guide
This repository contains the implementation of the paper [**[An Agentic Framework for Autonomous Materials Computation]**](https://arxiv.org/abs/2512.19458).
This document outlines the steps for environment setup, basic benchmarking, and custom dataset evaluation.

## 1. Installation

### 1.1 VASP Requirement
Before running the code, please ensure that **VASP** is correctly installed in your environment.
This repository interfaces with VASP via the standard command:
> `vasp_std`

Please verify that `vasp_std` is accessible in your system's PATH and is executable.

### 1.2 Python Dependencies
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

## 2. Basic Benchmark
To evaluate all bandgap tasks, run the main script with the bandgap configuration:

```bash
python main.py --config configs/bandgap.json
```

## 3. Custom Dataset Testing
To run evaluations on a specific subset of materials (custom test set), follow these steps:

### 3.1 Create Test Directory 
Create a test subdirectory within the data folder:

```bash
mkdir -p data/test
```

### 3.2 Prepare Data 

Copy the specific material folders you want to test from data/bandgap or data/relax into the data/test directory.

### 3.3 Run Custom Evaluation 
Execute the evaluation using the test configuration:
```bash
python main.py --config configs/test.json
```