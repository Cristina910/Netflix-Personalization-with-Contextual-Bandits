# 🎯 Netflix Personalization with Contextual Bandits
_A simulation-driven approach to personalized content ranking using reinforcement learning_

> Recruiter-friendly case study demonstrating contextual multi-armed bandits (ε-Greedy, LinUCB, Thompson Sampling) for Netflix-style personalization. Includes reproducible simulation, evaluation (CTR, regret), plots, and an interactive Streamlit dashboard.

## 📌 Overview
This project shows how **Contextual Bandits** can power **personalized recommendations**. We simulate user–item interactions and compare policies on click-through rate (CTR), cumulative reward, and regret.

## 🚀 Key Skills
- Reinforcement Learning (Contextual Bandits: ε-Greedy, LinUCB, Thompson Sampling)
- Python Data Stack (NumPy, Pandas, Scikit-learn, Matplotlib, Plotly, Streamlit)
- Experiment design, offline evaluation, and visualization
- Reproducible code with CLI and dashboard

## 🛠️ Tech Stack
**Languages:** Python (3.9+)  
**Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Plotly, Streamlit  
**Algorithms:** ε-Greedy, LinUCB, Thompson Sampling  
**Evaluation:** CTR, Cumulative Reward, Regret

## 🔄 Workflow
1) **Simulate** users/items & interactions  
2) **Run** bandit policies  
3) **Evaluate** CTR and regret  
4) **Visualize** learning curves (assets/)  
5) **Explore** results with Streamlit dashboard

## 📊 Results Snapshot (example)
| Policy | Avg. CTR | Cumulative Reward | Regret ↓ |
|---|---:|---:|---:|
| Epsilon-Greedy (0.1) | 0.19 | 1920 | 540 |
| LinUCB | 0.25 | 2470 | 280 |
| Thompson Sampling | 0.26 | 2580 | 260 |

> These are illustrative; run the simulation to regenerate.

## 🖥️ How to Run
```bash
# 1) Clone and enter
git clone https://github.com/YOUR_USERNAME/netflix-contextual-bandits.git
cd netflix-contextual-bandits

# 2) Install deps (recommend venv)
pip install -r requirements.txt

# 3) Run experiments (generates assets & results CSV)
python run_bandits.py --steps 5000 --epsilon 0.1

# 4) Launch interactive dashboard
streamlit run dashboard.py
```

## 📂 Structure
```
📦 netflix-contextual-bandits
 ┣ 📂 data               # (optional) synthetic CSVs if you export them
 ┣ 📂 notebooks          # Jupyter exploration (placeholders)
 ┣ 📂 src
 ┃ ┣ bandit_policies.py
 ┃ ┣ simulation.py
 ┃ ┣ evaluation.py
 ┃ ┗ utils.py
 ┣ 📂 assets             # Plots (learning_curves.png, regret_plot.png)
 ┣ dashboard.py
 ┣ run_bandits.py
 ┣ requirements.txt
 ┣ LICENSE
 ┗ README.md
```

## 📸 Visuals
- `assets/learning_curves.png` — CTR over time by policy  
- `assets/regret_plot.png` — Regret vs. oracle over time  
- `assets/architecture.png` — (optional) add your diagram

## 🧑‍💼 Contact
**Cristina Kalla** — [LinkedIn](https://linkedin.com/in/CristinaKalla) — kallaka@mail.uc.edu

---
**License:** MIT

---

## 🚀 Quickstart

Follow these steps to set up and run the project locally:

```bash
git clone https://github.com/Cristina910/Netflix-Personalization-with-Contextual-Bandits.git
cd netflix-contextual-bandits

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python run_bandits.py --steps 5000 --epsilon 0.1

streamlit run dashboard.py

cat << 'EOF' >> README.md

---

## 🚀 Quickstart

Follow these steps to set up and run the project locally:

```bash
git clone https://github.com/Cristina910/Netflix-Personalization-with-Contextual-Bandits.git
cd netflix-contextual-bandits

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python run_bandits.py --steps 5000 --epsilon 0.1

streamlit run dashboard.py

