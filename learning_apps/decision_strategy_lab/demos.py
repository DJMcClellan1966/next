"""Demos for Decision Support & Strategy Lab."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def ds_pareto_front():
    """Find the Pareto front of a bi-objective problem."""
    try:
        np.random.seed(42)
        n = 30
        cost = np.random.uniform(10, 100, n)
        quality = np.random.uniform(1, 10, n)

        # Find Pareto front (minimize cost, maximize quality)
        pareto = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j and cost[j] <= cost[i] and quality[j] >= quality[i]:
                    if cost[j] < cost[i] or quality[j] > quality[i]:
                        dominated = True
                        break
            if not dominated:
                pareto.append(i)

        out = "Pareto Front Discovery\n" + "=" * 50 + "\n"
        out += f"{n} alternatives, 2 objectives: minimize cost, maximize quality\n\n"
        out += f"Pareto-optimal solutions ({len(pareto)} of {n}):\n"
        for idx in sorted(pareto, key=lambda i: cost[i]):
            out += f"  Alt {idx:>2d}: cost={cost[idx]:>5.1f}, quality={quality[idx]:.1f}\n"
        out += f"\n{n - len(pareto)} dominated solutions eliminated."
        out += "\nDecision-maker chooses from Pareto front based on preferences."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_weighted_optimization():
    """Weighted-sum multi-objective optimization."""
    try:
        np.random.seed(42)
        alternatives = [
            {"name": "Plan A", "cost": 80, "speed": 3, "quality": 9},
            {"name": "Plan B", "cost": 40, "speed": 7, "quality": 5},
            {"name": "Plan C", "cost": 60, "speed": 5, "quality": 7},
            {"name": "Plan D", "cost": 90, "speed": 2, "quality": 10},
            {"name": "Plan E", "cost": 30, "speed": 8, "quality": 4},
        ]
        weight_sets = [
            {"label": "Cost-focused", "cost": 0.6, "speed": 0.2, "quality": 0.2},
            {"label": "Balanced", "cost": 0.33, "speed": 0.33, "quality": 0.34},
            {"label": "Quality-focused", "cost": 0.2, "speed": 0.2, "quality": 0.6},
        ]

        out = "Weighted Multi-Objective Optimization\n" + "=" * 50 + "\n\n"
        for ws in weight_sets:
            scores = []
            for alt in alternatives:
                # Normalize: lower cost is better → invert
                s = (1 - alt["cost"] / 100) * ws["cost"] + (alt["speed"] / 10) * ws["speed"] + (alt["quality"] / 10) * ws["quality"]
                scores.append((alt["name"], s))
            scores.sort(key=lambda x: -x[1])
            winner = scores[0]
            out += f"  {ws['label']:>16s} weights → Winner: {winner[0]} (score={winner[1]:.3f})\n"
            for name, s in scores:
                out += f"    {name}: {s:.3f}\n"
            out += "\n"
        out += "Different priorities lead to different optimal decisions!"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_nsga2():
    """Simplified NSGA-II on a bi-objective problem."""
    try:
        np.random.seed(42)
        pop_size = 20
        generations = 30

        # Bi-objective: f1 = x², f2 = (x-2)²
        pop = np.random.uniform(-2, 4, pop_size)

        def objectives(x):
            return x ** 2, (x - 2) ** 2

        for gen in range(generations):
            # Evaluate
            f_vals = np.array([objectives(x) for x in pop])

            # Non-dominated sorting (simplified: just rank 0)
            ranks = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j:
                        if np.all(f_vals[j] <= f_vals[i]) and np.any(f_vals[j] < f_vals[i]):
                            ranks[i] += 1

            # Select best half
            order = np.argsort(ranks)
            parents = pop[order[:pop_size // 2]]

            # Create offspring
            offspring = parents + np.random.normal(0, 0.2, len(parents))
            pop = np.concatenate([parents, offspring])[:pop_size]

        f_final = np.array([objectives(x) for x in pop])
        pareto_idx = []
        for i in range(len(pop)):
            dominated = False
            for j in range(len(pop)):
                if i != j and np.all(f_final[j] <= f_final[i]) and np.any(f_final[j] < f_final[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_idx.append(i)

        out = "NSGA-II (Simplified)\n" + "=" * 50 + "\n"
        out += f"Objectives: f1=x², f2=(x-2)²   |   Optimal: x ∈ [0, 2]\n"
        out += f"Population: {pop_size}, Generations: {generations}\n\n"
        out += f"Pareto front ({len(pareto_idx)} solutions):\n"
        for idx in sorted(pareto_idx, key=lambda i: pop[i]):
            x = pop[idx]
            f1, f2 = objectives(x)
            out += f"  x={x:.3f} → f1={f1:.3f}, f2={f2:.3f}\n"
        out += "\nAll solutions lie near x ∈ [0, 2] — the true Pareto front."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_nash_equilibrium():
    """Find Nash equilibrium of a 2-player game."""
    try:
        # Prisoner's Dilemma
        # Payoff matrices (row=P1 action, col=P2 action)
        # Actions: 0=Cooperate, 1=Defect
        P1 = np.array([[-1, -3], [0, -2]])  # Player 1 payoffs
        P2 = np.array([[-1, 0], [-3, -2]])   # Player 2 payoffs

        out = "Nash Equilibrium: Prisoner's Dilemma\n" + "=" * 50 + "\n"
        out += "Actions: C=Cooperate, D=Defect\n"
        out += "Payoff matrix (P1, P2):\n"
        out += "           P2: C    P2: D\n"
        out += f"  P1: C  ({P1[0,0]:>2d},{P2[0,0]:>2d})  ({P1[0,1]:>2d},{P2[0,1]:>2d})\n"
        out += f"  P1: D  ({P1[1,0]:>2d},{P2[1,0]:>2d})  ({P1[1,1]:>2d},{P2[1,1]:>2d})\n\n"

        # Find pure strategy Nash
        nash = []
        actions = ["Cooperate", "Defect"]
        for i in range(2):
            for j in range(2):
                p1_best = P1[i, j] >= max(P1[:, j])
                p2_best = P2[i, j] >= max(P2[i, :])
                if p1_best and p2_best:
                    nash.append((i, j))

        for i, j in nash:
            out += f"Nash Equilibrium: P1={actions[i]}, P2={actions[j]} → payoffs ({P1[i,j]}, {P2[i,j]})\n"
        out += "\nParadox: Both defect (Nash) is worse than both cooperate!"
        out += "\nThis illustrates why rational self-interest can lead to suboptimal outcomes."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_shapley_values():
    """Compute Shapley values for feature attribution."""
    try:
        from math import factorial

        players = ["Marketing", "Engineering", "Sales"]
        n = len(players)

        # Coalition value function (revenue in $M)
        def v(coalition):
            s = set(coalition)
            values = {
                frozenset(): 0,
                frozenset(["Marketing"]): 2,
                frozenset(["Engineering"]): 3,
                frozenset(["Sales"]): 1,
                frozenset(["Marketing", "Engineering"]): 8,
                frozenset(["Marketing", "Sales"]): 5,
                frozenset(["Engineering", "Sales"]): 6,
                frozenset(["Marketing", "Engineering", "Sales"]): 12,
            }
            return values.get(frozenset(s), 0)

        # Compute Shapley values
        shapley = {}
        for player in players:
            sv = 0
            others = [p for p in players if p != player]
            for size in range(n):
                # All subsets of others with given size
                from itertools import combinations
                for subset in combinations(others, size):
                    s = set(subset)
                    marginal = v(s | {player}) - v(s)
                    weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                    sv += weight * marginal
            shapley[player] = sv

        out = "Shapley Values: Fair Revenue Attribution\n" + "=" * 50 + "\n"
        out += f"Grand coalition value: ${v(players)}M\n\n"
        out += "Fair distribution (Shapley values):\n"
        total = sum(shapley.values())
        for player, sv in shapley.items():
            bar = "█" * int(sv * 3)
            out += f"  {player:>12s}: ${sv:.2f}M ({sv/total*100:.1f}%) {bar}\n"
        out += f"\n  Total: ${total:.2f}M (= grand coalition, as expected)"
        out += "\nShapley values are the ONLY attribution method satisfying efficiency,"
        out += "\nsymmetry, linearity, and null-player axioms."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_auction():
    """Second-price (Vickrey) auction simulation."""
    try:
        np.random.seed(42)
        n_bidders = 5
        true_values = np.random.uniform(50, 200, n_bidders)
        # In Vickrey auction, truthful bidding is optimal
        bids = true_values.copy()

        winner = np.argmax(bids)
        price = np.sort(bids)[-2]  # Second highest bid

        out = "Vickrey (Second-Price) Auction\n" + "=" * 50 + "\n\n"
        out += "Bidders (truthful bidding is dominant strategy):\n"
        for i in range(n_bidders):
            marker = " ← WINNER" if i == winner else ""
            out += f"  Bidder {i+1}: value=${true_values[i]:.0f}, bid=${bids[i]:.0f}{marker}\n"
        out += f"\nWinner: Bidder {winner+1} (bid ${bids[winner]:.0f})"
        out += f"\nPrice paid: ${price:.0f} (second-highest bid)"
        out += f"\nSurplus: ${true_values[winner] - price:.0f}"
        out += "\n\nKey insight: Truthful bidding is the dominant strategy"
        out += "\nbecause you pay the second price, not your own bid."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_scenario_matrix():
    """2x2 scenario planning matrix."""
    try:
        scenarios = {
            ("High Tech", "Loose Regulation"): {
                "name": "Wild West Innovation",
                "desc": "Rapid tech progress, minimal oversight → fast growth but ethical risks",
                "strategy": "Move fast, build trust, self-regulate",
            },
            ("High Tech", "Strict Regulation"): {
                "name": "Guided Progress",
                "desc": "Strong tech capabilities within clear guardrails → balanced growth",
                "strategy": "Comply early, shape standards, build moats",
            },
            ("Low Tech", "Loose Regulation"): {
                "name": "Stagnation",
                "desc": "Limited tech progress, no investment incentive → slow market",
                "strategy": "Focus on fundamentals, wait for catalyst",
            },
            ("Low Tech", "Strict Regulation"): {
                "name": "Locked Down",
                "desc": "Heavy regulation on limited tech → high barriers, niche markets",
                "strategy": "Find compliance advantages, lobby for reform",
            },
        }

        out = "Scenario Planning: 2×2 Matrix\n" + "=" * 50 + "\n"
        out += "Axes: Technology Progress × Regulatory Environment\n\n"
        for (tech, reg), s in scenarios.items():
            out += f"  [{tech:>8s} / {reg:>18s}]: {s['name']}\n"
            out += f"    {s['desc']}\n"
            out += f"    Strategy: {s['strategy']}\n\n"
        out += "Robust strategy: one that works across multiple scenarios."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_monte_carlo():
    """Monte Carlo simulation for investment decision."""
    try:
        np.random.seed(42)
        n_sims = 10000
        initial = 100000
        years = 5

        # Uncertain parameters
        returns = np.random.normal(0.08, 0.15, (n_sims, years))
        final_values = initial * np.prod(1 + returns, axis=1)

        mean_val = np.mean(final_values)
        p5 = np.percentile(final_values, 5)
        p95 = np.percentile(final_values, 95)
        prob_loss = np.mean(final_values < initial)

        out = "Monte Carlo Investment Simulation\n" + "=" * 50 + "\n"
        out += f"Initial: ${initial:,.0f} | Horizon: {years} years | Sims: {n_sims:,}\n"
        out += f"Return: μ=8%, σ=15% (annual)\n\n"
        out += f"Results:\n"
        out += f"  Expected value: ${mean_val:>12,.0f}\n"
        out += f"  5th percentile: ${p5:>12,.0f} (VaR 95%)\n"
        out += f"  95th percentile: ${p95:>12,.0f}\n"
        out += f"  P(loss): {prob_loss*100:.1f}%\n"

        # Histogram
        out += "\nDistribution:\n"
        bins = np.linspace(0, 300000, 11)
        counts, _ = np.histogram(final_values, bins)
        for i in range(len(counts)):
            bar = "█" * (counts[i] // 100)
            out += f"  ${bins[i]/1000:>5.0f}k-${bins[i+1]/1000:>5.0f}k: {bar} ({counts[i]})\n"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_decision_tree():
    """Decision tree analysis with expected value."""
    try:
        out = "Decision Tree: Product Launch\n" + "=" * 50 + "\n\n"
        out += "Decision: Launch new product or improve existing?\n\n"

        # Decision tree
        d1 = {"name": "Launch New", "branches": [
            {"event": "High demand (40%)", "prob": 0.4, "payoff": 500},
            {"event": "Medium demand (35%)", "prob": 0.35, "payoff": 200},
            {"event": "Low demand (25%)", "prob": 0.25, "payoff": -100},
        ]}
        d2 = {"name": "Improve Existing", "branches": [
            {"event": "Successful (60%)", "prob": 0.6, "payoff": 300},
            {"event": "Marginal (30%)", "prob": 0.3, "payoff": 100},
            {"event": "Failed (10%)", "prob": 0.1, "payoff": -50},
        ]}

        for d in [d1, d2]:
            ev = sum(b["prob"] * b["payoff"] for b in d["branches"])
            out += f"  Option: {d['name']} (EV = ${ev:,.0f}k)\n"
            for b in d["branches"]:
                out += f"    ├─ {b['event']}: ${b['payoff']:>+5d}k\n"
            out += "\n"

        ev1 = sum(b["prob"] * b["payoff"] for b in d1["branches"])
        ev2 = sum(b["prob"] * b["payoff"] for b in d2["branches"])
        best = d1["name"] if ev1 > ev2 else d2["name"]
        out += f"Recommendation: {best} (EV=${max(ev1,ev2):,.0f}k vs ${min(ev1,ev2):,.0f}k)"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_ethics_eval():
    """Evaluate a decision under three ethical frameworks."""
    try:
        decision = "Deploy AI system that improves efficiency 30% but eliminates 50 jobs"
        out = "Ethical Framework Analysis\n" + "=" * 50 + "\n"
        out += f"Decision: {decision}\n\n"

        frameworks = [
            ("Consequentialism", "MIXED",
             "Net utility depends on retraining programs. +30% efficiency helps customers,\n"
             "    but -50 livelihoods is significant negative. Need cost-benefit analysis."),
            ("Deontology", "CAUTION",
             "Duty to employees (implicit contract). Kant's categorical imperative:\n"
             "    'Would we universalize replacing all workers with AI?' If no, proceed carefully."),
            ("Virtue Ethics", "CAUTION",
             "Does this decision reflect compassion and justice? A virtuous company\n"
             "    would provide transition support. Character of the organization matters."),
        ]

        for name, verdict, analysis in frameworks:
            icon = "🟢" if verdict == "APPROVE" else ("🟡" if verdict == "MIXED" or verdict == "CAUTION" else "🔴")
            out += f"  {icon} {name}: [{verdict}]\n    {analysis}\n\n"

        out += "Synthesis: All frameworks recommend proceeding WITH safeguards\n"
        out += "(retraining programs, severance, gradual transition)."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_fairness_check():
    """Check fairness metrics for a binary classifier."""
    try:
        np.random.seed(42)
        n = 1000
        group = np.random.choice([0, 1], n, p=[0.6, 0.4])
        y_true = np.random.choice([0, 1], n, p=[0.7, 0.3])
        # Biased model: group 1 gets lower scores
        scores = np.random.uniform(0, 1, n) - 0.1 * group
        y_pred = (scores > 0.5).astype(int)

        def metrics(mask):
            tp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 1))
            fp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 0))
            fn = np.sum((y_pred[mask] == 0) & (y_true[mask] == 1))
            tn = np.sum((y_pred[mask] == 0) & (y_true[mask] == 0))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            pos_rate = np.mean(y_pred[mask])
            return {"tpr": tpr, "fpr": fpr, "pos_rate": pos_rate, "n": int(np.sum(mask))}

        g0 = metrics(group == 0)
        g1 = metrics(group == 1)

        out = "Fairness Audit\n" + "=" * 50 + "\n"
        out += f"{'Metric':>20s} {'Group 0':>10s} {'Group 1':>10s} {'Gap':>10s}\n"
        out += f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}\n"
        for key in ["pos_rate", "tpr", "fpr"]:
            gap = abs(g0[key] - g1[key])
            flag = " ⚠️" if gap > 0.05 else " ✅"
            out += f"  {key:>18s} {g0[key]:>10.3f} {g1[key]:>10.3f} {gap:>10.3f}{flag}\n"
        out += f"\nStatistical parity gap: {abs(g0['pos_rate'] - g1['pos_rate']):.3f}"
        out += "\nValues > 0.05 indicate potential unfairness."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def ds_stakeholder():
    """Stakeholder power-interest analysis."""
    try:
        stakeholders = [
            {"name": "CEO", "power": 9, "interest": 8, "strategy": "Manage closely"},
            {"name": "Employees", "power": 4, "interest": 9, "strategy": "Keep informed"},
            {"name": "Customers", "power": 7, "interest": 6, "strategy": "Keep satisfied"},
            {"name": "Regulators", "power": 8, "interest": 3, "strategy": "Keep satisfied"},
            {"name": "Community", "power": 3, "interest": 4, "strategy": "Monitor"},
            {"name": "Investors", "power": 8, "interest": 7, "strategy": "Manage closely"},
        ]

        out = "Stakeholder Analysis: Power × Interest\n" + "=" * 50 + "\n\n"
        out += f"{'Stakeholder':>12s} {'Power':>6s} {'Interest':>8s} {'Strategy':>18s}\n"
        out += f"{'-'*12} {'-'*6} {'-'*8} {'-'*18}\n"
        for s in stakeholders:
            out += f"  {s['name']:>10s} {s['power']:>6d} {s['interest']:>8d} {s['strategy']:>18s}\n"
        out += "\nQuadrants:\n"
        out += "  High Power + High Interest → Manage closely (CEO, Investors)\n"
        out += "  High Power + Low Interest  → Keep satisfied (Regulators)\n"
        out += "  Low Power + High Interest  → Keep informed (Employees)\n"
        out += "  Low Power + Low Interest   → Monitor (Community)"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


# --- Demo dispatcher ---
DEMO_HANDLERS = {
    "ds_pareto_front": ds_pareto_front,
    "ds_weighted_optimization": ds_weighted_optimization,
    "ds_nsga2": ds_nsga2,
    "ds_nash_equilibrium": ds_nash_equilibrium,
    "ds_shapley_values": ds_shapley_values,
    "ds_auction": ds_auction,
    "ds_scenario_matrix": ds_scenario_matrix,
    "ds_monte_carlo": ds_monte_carlo,
    "ds_decision_tree": ds_decision_tree,
    "ds_ethics_eval": ds_ethics_eval,
    "ds_fairness_check": ds_fairness_check,
    "ds_stakeholder": ds_stakeholder,
}

def run_demo(demo_id: str) -> dict:
    handler = DEMO_HANDLERS.get(demo_id)
    if handler:
        return handler()
    return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
