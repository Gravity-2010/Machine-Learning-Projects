# Machine Learning Projects

Practicing core Machine Learning concepts by implementing them in hands-on notebooks. Each notebook is intentionally concise and self-contained so you can experiment, tweak hyperparameters, and visualize results quickly.

> **Repo owner:** Gravity-2010  
> **Goal:** Learn-by-building — short, focused projects that teach foundational ML ideas.

---

## Table of Contents

- [Project List](#project-list)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [How to Run the Notebooks](#how-to-run-the-notebooks)
- [Data Sources](#data-sources)
- [Learning Objectives](#learning-objectives)
- [Results & What to Expect](#results--what-to-expect)
- [Project Ideas to Extend](#project-ideas-to-extend)
- [Repo Structure](#repo-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project List

1. **Clustering Algorithms.ipynb** — Try classic unsupervised methods (e.g., K-Means, Hierarchical, DBSCAN). Explore how different distance metrics and feature scaling affect clusters.
2. **Gradient Descent and Perceptron Algorithm.ipynb** — Build intuition for optimization and linear classifiers. Derive/update rules, visualize decision boundaries, and compare batch vs stochastic updates.
3. **Learning about weights using Digits dataset and Perceptrons.ipynb** — Use `sklearn.datasets.load_digits` to train a perceptron/multi-class one-vs-rest. Inspect learned weights and confusion matrices.
4. **Reinforcement Learning.ipynb** — Implement a simple RL loop (value iteration, policy iteration, or tabular Q-learning). Run on a toy environment to understand state–action value updates.

> If you open a notebook and see missing imports, install the matching extras listed below. Notebooks are intentionally lightweight and rely on common ML packages.

---

## Quick Start

```bash
# 1) Create and activate a fresh environment (conda recommended)
conda create -n ml-projects python=3.11 -y
conda activate ml-projects

# 2) Install the core dependencies
pip install -U numpy pandas scikit-learn matplotlib jupyterlab

# 3) (Optional) Extras often needed by these notebooks
pip install seaborn scipy plotly

# 4) (Only for the RL notebook) Install a simple environment
# Option A: Gymnasium (actively maintained)
pip install gymnasium
# Optionally install classic-control envs:
pip install gymnasium[classic-control]

# Option B: OpenAI Gym (legacy) — if the notebook uses it
pip install gym==0.26.2
```

> Prefer **Gymnasium** if you’re starting fresh. If the RL notebook imports `gym`, either switch to Gymnasium with a small import edit or install the legacy `gym` package.

---

## Environment Setup

- **Python:** 3.10 or 3.11 recommended  
- **Core libs:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`  
- **Nice to have:** `seaborn`, `scipy`, `plotly`  
- **RL extras (if used):** `gymnasium` (or legacy `gym`), `pygame` (for some renderers)

Create a reproducible environment file (optional):

```bash
# Export the environment after installs
pip freeze > requirements.txt
```

Then teammates can run:

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run the Notebooks

```bash
# From the repo root
jupyter lab
# or
jupyter notebook
```

- Open any `*.ipynb` and run cells from top to bottom.
- If a dataset download is required, the notebook typically handles it with `sklearn.datasets` or a small helper cell.
- To reset experiments, **Kernel → Restart & Clear Output**.

**Tips**  
- If plots look off, check that you ran the preprocessing/scaling cells first.  
- For clustering, try toggling `StandardScaler()` vs raw features.  
- For perceptrons, try different learning rates (`eta0`) and epochs to see convergence behavior.  

---

## Data Sources

- **Digits dataset:** `from sklearn.datasets import load_digits` (built-in, no manual download).  
- **Toy clustering data:** `make_blobs`, `make_moons`, or classic **Iris** dataset are commonly used.  
- **RL environments:** `gymnasium.make("CartPole-v1")` or GridWorld-style custom environments in the notebook.

> If any notebook expects a file path, it will usually generate or download it automatically. Otherwise, a cell at the top will describe how to place the file locally.

---

## Learning Objectives

- **Clustering**
  - Understand when to use K-Means vs Density-based methods (DBSCAN) vs Hierarchical.
  - See how scaling, initialization, and `k` selection (elbow/silhouette) impact results.

- **Gradient Descent & Perceptron**
  - Derive the update steps and link them to geometry of the decision boundary.
  - Compare batch vs stochastic gradient descent; visualize learning curves.

- **Digits + Perceptron Weights**
  - Interpret learned weights as “templates”.  
  - Build intuition using confusion matrices and per-class precision/recall.

- **Reinforcement Learning**
  - Understand the Bellman equations at a practical level.  
  - Implement a simple tabular Q-learning or policy/value iteration loop.  
  - Observe exploration vs exploitation and the role of the discount factor.

---

## Results & What to Expect

- **Clustering:** 2D plots of clusters; inertia/silhouette scores to compare runs.  
- **Perceptron:** Decision boundaries and accuracy curves; effects of learning rate and epochs.  
- **Digits:** Heatmaps of weight vectors per class; confusion matrix insights.  
- **RL:** Episode reward trends over time; stable policies after sufficient training.

Reproducibility tip: set seeds, e.g. `np.random.seed(42)` or estimator’s `random_state=42`.

---

## Project Ideas to Extend

- **Clustering**
  - Try k-means++ vs random init; compare metrics on Iris/Wine datasets.
  - Dimensionality reduction first (PCA/UMAP) → then cluster.
- **Optimization**
  - Implement momentum, Nesterov, RMSProp, or Adam and compare with vanilla GD.
- **Perceptron**
  - Add polynomial features; compare to Logistic Regression and SVM.
- **Digits**
  - Swap the perceptron with an `SGDClassifier` or small MLP; report per-class F1.
- **RL**
  - Add epsilon decay schedules; plot Q-table heatmaps; try different environments.

---

## Repo Structure

```
.
├── Clustering Algorithms.ipynb
├── Gradient Descent and Perceptron Algorithm.ipynb
├── Learning about weights using Digits dataset and Perceptrons.ipynb
├── Reinforcement Learning.ipynb
└── README.md  <-- you are here
```

> Notebooks are intentionally independent. Feel free to duplicate a notebook to start a new experiment without breaking others.

---

## Contributing

Have a small improvement or a new learning mini-project? PRs are welcome!

**Guidelines**
1. Keep notebooks self-contained and runnable from a clean environment.
2. Prefer built-in datasets (e.g., `sklearn.datasets`) or provide a small download cell.
3. Add short markdown cells explaining *why* choices were made (learning-first approach).
4. If you add new deps, update `requirements.txt` or the **Environment Setup** section.

**Suggested Labels (for Issues/PRs)**
- `good first issue`, `documentation`, `notebook`, `enhancement`, `bug`

---

## License

No license file is present yet. If you’re open to it, consider adding an [MIT License](https://choosealicense.com/licenses/mit/) so others can use and learn from your work. If you prefer a different license, add `LICENSE` at the repo root and mention it here.

---

## Acknowledgements

- `scikit-learn` team and docs — for datasets and clean APIs.
- Open-source community examples that inspire these educational notebooks.

---

*Happy learning and tinkering!* 🧪📈
