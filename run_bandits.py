import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulation import Environment
from src.bandit_policies import EpsilonGreedy, LinUCB, ThompsonSamplingLinear
from src.evaluation import compute_ctr, compute_regret, to_results_df

def run_experiment(steps=5000, n_items=20, n_users=200, d=8, epsilon=0.1, alpha=1.0, v=0.1, seed=42, out_dir="assets"):
    env = Environment(user_dim=d, item_dim=d, n_users=n_users, n_items=n_items, seed=seed)
    policies = {
        f"epsilon_greedy_{epsilon}": EpsilonGreedy(n_items, epsilon=epsilon),
        f"linucb_{alpha}": LinUCB(n_items, d=d, alpha=alpha),
        f"thompson_{v}": ThompsonSamplingLinear(n_items, d=d, v=v),
    }

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for name, policy in policies.items():
        clicks = []
        taken_probs = []
        optimal_probs = []
        actions = []
        users = []

        for t in range(steps):
            user_idx = np.random.randint(env.n_users)
            # Context matrix per arm is just item features
            context = env.item_X
            a = policy.select(context)
            click, p = env.step(user_idx, a)
            # oracle best action for regret
            best_a = env.best_action(user_idx)
            _, best_p = env.step(user_idx, best_a)

            policy.update(a, click, context[a])
            clicks.append(click)
            taken_probs.append(p)
            optimal_probs.append(best_p)
            actions.append(a)
            users.append(user_idx)

        clicks = np.array(clicks, dtype=float)
        taken_probs = np.array(taken_probs, dtype=float)
        optimal_probs = np.array(optimal_probs, dtype=float)

        ctr_curve = compute_ctr(clicks)
        regret_curve = compute_regret(optimal_probs, taken_probs)

        df = pd.DataFrame({
            "step": np.arange(1, steps+1),
            "user": users,
            "action": actions,
            "click": clicks,
            "ctr": ctr_curve,
            "prob_taken": taken_probs,
            "prob_optimal": optimal_probs,
            "regret": regret_curve,
            "policy": name
        })
        results[name] = df
        df.to_csv(os.path.join(out_dir, f"results_{name}.csv"), index=False)

    # Combine for plotting
    combined = pd.concat(results.values(), ignore_index=True)

    # Plot CTR curves
    plt.figure()
    for name, df in combined.groupby("policy"):
        plt.plot(df["step"], df["ctr"], label=name)
    plt.title("CTR over time")
    plt.xlabel("Step")
    plt.ylabel("CTR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=200)
    plt.close()

    # Plot regret curves
    plt.figure()
    for name, df in combined.groupby("policy"):
        plt.plot(df["step"], df["regret"], label=name)
    plt.title("Cumulative Regret (lower is better)")
    plt.xlabel("Step")
    plt.ylabel("Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regret_plot.png"), dpi=200)
    plt.close()

    # Summary table
    summary = combined.groupby("policy").agg(
        avg_ctr=("click", "mean"),
        cumulative_reward=("click", "sum"),
        final_regret=("regret", "last")
    ).reset_index()

    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--n_items", type=int, default=20)
    parser.add_argument("--n_users", type=int, default=200)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--v", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="assets")
    args = parser.parse_args()

    run_experiment(**vars(args))
