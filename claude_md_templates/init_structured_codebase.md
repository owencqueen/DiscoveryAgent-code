# README: Package Structure for Code + Analysis

This repository contains both **library code** (reusable functionality) and **analysis code** (data experiments, exploratory workflows, etc.). To keep things clean and maintainable, follow the structure below.

---

## 📂 Directory Structure

IMPORTNAT: You should keep all your code organized in the following manner:

my_project/
│
├── src/                  # Source code for the reusable package
│   └── my_project/       # Actual Python package code
│       ├── __init__.py
│       ├── core.py       # Core functionality
│       ├── utils.py      # Helper functions
│       └── ...           # Other modules
│
├── experiments/          # Analysis, experiments, and data-driven code
│   ├── notebooks/        # Jupyter notebooks for exploration
│   ├── scripts/          # Standalone analysis scripts
│   ├── configs/          # YAML/JSON config files for experiments
│   └── results/          # Outputs: figures, logs, metrics
│
│
├── data/                 # Raw or sample data (small, versionable files)
│   └── processed/
│
├── README.md             # High-level project description
└── .gitignore

You don't need to use all of these components, but the overall directory structure should be maintained when adding and running new code.

---

## 📦 Source Code (`src/my_project/`)

- Write **reusable, modular code** here.  
- Keep it independent from specific experiments.  
- Code in this folder should not assume access to experiment data paths or notebooks.  
- Follow good package practices:
  - Use `__init__.py` to expose main functionality.
  - Add docstrings and typing hints.
  - Keep modules small and cohesive.

---

## 🔬 Experiments (`experiments/`)

- Use this for **data analysis, experiments, and workflows**.
- Separate exploratory code from production code:
  - **`notebooks/`**: Interactive development and exploration.  
  - **`scripts/`**: Reproducible experiment runs (can be invoked from command line).  
  - **`configs/`**: Store hyperparameters, experiment settings, or data paths.  
  - **`results/`**: Generated outputs (figures, logs, metrics, serialized models).
- Experiment scripts should **import functions from `src/my_project/`**, not duplicate them.

---

## 🧪 Tests (`tests/`)

- Write unit tests for code inside `src/my_project/`.
- Use `pytest` (recommended).
- Tests should run quickly and not depend on large datasets.

---

## 📊 Data (`data/`)

- Store processed subsets of data.
- IMPORTANT: store only minimal data, whatever is necessary to speed up analyses. Reuse stored data when possible.

---

## 📝 Best Practices

- **Library code** = clean, tested, documented.  
- **Analysis code** = messy is okay, but keep it reproducible (configs + scripts).  
- **Never mix experiment code into `src/`**. Import from `src/` instead.  
- **Version control**: Don’t commit large datasets or intermediate results.  
