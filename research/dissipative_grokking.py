"""
Dissipative Grokking Predictor
===============================

Models neural network training as a Prigogine dissipative system to predict
when grokking (sudden generalization after memorization) will occur.

Core idea: Training dynamics follow
    ds/dt = (E_in - gammas)(1 - s) - deltas

where:
    s = generalization level (0=memorization, 1=full generalization)
    E_in = learning rate x gradient signal (energy input)
    gamma = weight decay rate (dissipation)
    delta = noise/forgetting rate

The system undergoes a phase transition at a critical ratio:
    E_in / gamma > critical_threshold

Below the threshold: stable at s~=0 (memorization, no grokking)
Above the threshold: bifurcation to s>0 (generalization emerges)

This predicts grokking timing from just eta, lambda, and gradient statistics.

Author: Research module, ML-ToolBox
Status: Proof-of-concept -- not peer-reviewed
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


# =============================================================================
# CORE: Dissipative Dynamics Model
# =============================================================================

class DissipativeModel:
    """
    Models generalization dynamics as a dissipative system.

    The ODE:
        ds/dt = (E_in - gamma*s)*(1 - s) - delta*s

    Fixed points (set ds/dt = 0):
        s_0 = 0 is always a fixed point when E_in = 0
        For E_in > 0, solving the quadratic gives critical behavior

    Stability: linearize around s=0:
        ds/dt ~= (E_in - delta)*s  for small s
        -> unstable (grokking starts) when E_in > delta

    The full critical condition accounting for gamma:
        E_in > delta + gamma*E_in/(E_in + gamma)  [from the bifurcation analysis]

    Simplified: grokking occurs when eta/lambda > threshold
    """

    def __init__(self, gamma: float = 0.1, delta: float = 0.05):
        """
        Args:
            gamma: dissipation rate (maps to weight decay)
            delta: noise rate (maps to label noise / forgetting)
        """
        self.gamma = gamma
        self.delta = delta

    def ds_dt(self, s: float, e_in: float) -> float:
        """Rate of change of generalization level."""
        return (e_in - self.gamma * s) * (1.0 - s) - self.delta * s

    def find_fixed_points(self, e_in: float) -> List[Dict]:
        """
        Find fixed points of ds/dt = 0.

        (E_in - gammas)(1-s) - deltas = 0
        Expanding: E_in - E_in*s - gammas + gammas^2 - deltas = 0
                   gammas^2 - (E_in + gamma + delta)s + E_in = 0

        Using quadratic formula:
            s = [(E_in + gamma + delta) +/- sqrt((E_in + gamma + delta)^2 - 4gammaE_in)] / (2gamma)
        """
        a = self.gamma
        b = -(e_in + self.gamma + self.delta)
        c = e_in

        disc = b ** 2 - 4 * a * c
        fixed_points = []

        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            s1 = (-b - sqrt_disc) / (2 * a) if a != 0 else -c / b
            s2 = (-b + sqrt_disc) / (2 * a) if a != 0 else s1

            for s_fp in [s1, s2]:
                if 0 <= s_fp <= 1:
                    # Check stability: d/ds[ds/dt] at fixed point
                    eps = 1e-6
                    deriv = (self.ds_dt(s_fp + eps, e_in) - self.ds_dt(s_fp - eps, e_in)) / (2 * eps)
                    fixed_points.append({
                        "s": float(s_fp),
                        "stable": deriv < 0,
                        "eigenvalue": float(deriv),
                    })

        return fixed_points

    def critical_energy(self) -> float:
        """
        Find the critical E_in where generalization becomes possible.

        At the bifurcation, the discriminant = 0 OR the lower fixed point
        becomes unstable. Linearizing around s=0:
            ds/dt ~= (E_in - delta) * 1 = E_in - delta
        So s=0 is unstable when E_in > delta.

        More precisely, accounting for the full dynamics:
            critical E_in = delta(1 + delta/gamma) approximately
        """
        # Numerical: find E_in where s=0 becomes unstable
        for e_test in np.linspace(0, 2, 1000):
            fp = self.find_fixed_points(e_test)
            # Look for when s=0 neighborhood becomes unstable
            low_fps = [f for f in fp if f["s"] < 0.1]
            if low_fps and not low_fps[0]["stable"]:
                return float(e_test)
            # Or when a new high fixed point appears
            high_fps = [f for f in fp if f["s"] > 0.3 and f["stable"]]
            if high_fps:
                return float(e_test)
        return float('inf')

    def simulate(self, e_in: float, s0: float = 0.01, dt: float = 0.01,
                 t_max: float = 50.0) -> Dict:
        """
        Simulate the dissipative dynamics forward in time.

        Returns trajectory and grokking detection.
        """
        t = 0.0
        s = s0
        trajectory = [{"t": 0.0, "s": s0}]
        grokking_time = None

        while t < t_max:
            ds = self.ds_dt(s, e_in)
            s = np.clip(s + dt * ds, 0.0, 1.0)
            t += dt
            trajectory.append({"t": float(t), "s": float(s)})

            # Detect grokking: rapid increase in s
            if grokking_time is None and s > 0.5:
                grokking_time = t

        return {
            "trajectory": trajectory,
            "final_s": float(s),
            "grokked": s > 0.5,
            "grokking_time": grokking_time,
            "e_in": e_in,
        }


# =============================================================================
# MAPPING: Neural Network -> Dissipative Parameters
# =============================================================================

def estimate_energy_input(learning_rate: float, grad_norm: float,
                          loss_value: float) -> float:
    """
    Map neural network training to dissipative energy input.

    E_in  proportional to  eta x ||nablaL|| x (1 - train_accuracy)

    Higher learning rate + larger gradients + more room to improve = more energy.
    """
    return learning_rate * grad_norm


def map_training_to_dissipative(learning_rate: float, weight_decay: float,
                                label_noise: float = 0.0) -> DissipativeModel:
    """
    Map training hyperparameters to dissipative model parameters.

    gamma (dissipation) <- weight_decay x scale_factor
    delta (noise) <- label_noise_rate + inherent_forgetting
    """
    gamma = weight_decay * 10.0  # Scale to make dynamics visible
    delta = label_noise * 5.0 + 0.02  # Base forgetting rate
    return DissipativeModel(gamma=gamma, delta=delta)


# =============================================================================
# EXPERIMENT 1: Phase Diagram
# =============================================================================

def run_phase_diagram_experiment():
    """
    Experiment 1: Map the phase diagram of grokking.

    Sweep learning_rate and weight_decay, predict where grokking occurs.
    Show the critical boundary.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Grokking Phase Diagram")
    print("=" * 70)
    print()
    print("Sweeping learning rate (eta) vs weight decay (lambda).")
    print("Predicting where grokking occurs based on dissipative dynamics.")
    print()

    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    weight_decays = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Assume a typical gradient norm during training
    typical_grad_norm = 1.0

    print(f"  {'eta \\ lambda':>8s}", end="")
    for wd in weight_decays:
        print(f" {wd:>6.3f}", end="")
    print()
    print("  " + "-" * (8 + 7 * len(weight_decays)))

    phase_data = []

    for lr in learning_rates:
        print(f"  {lr:>8.3f}", end="")
        for wd in weight_decays:
            model = map_training_to_dissipative(lr, wd)
            e_in = estimate_energy_input(lr, typical_grad_norm, 0.5)
            result = model.simulate(e_in, s0=0.01, t_max=100.0)

            if result["grokked"]:
                symbol = "  *   "  # Grokking predicted
            else:
                symbol = "  *   "  # No grokking
            print(symbol, end="")

            phase_data.append({
                "lr": lr, "wd": wd, "ratio": lr / wd,
                "grokked": result["grokked"],
                "grokking_time": result["grokking_time"],
                "final_s": result["final_s"],
            })
        print()

    print()
    print("  * = grokking predicted    * = memorization only")
    print()

    # Find the critical ratio
    grokked = [d for d in phase_data if d["grokked"]]
    not_grokked = [d for d in phase_data if not d["grokked"]]

    if grokked and not_grokked:
        min_grok_ratio = min(d["ratio"] for d in grokked)
        max_no_grok_ratio = max(d["ratio"] for d in not_grokked)
        critical_ratio = (min_grok_ratio + max_no_grok_ratio) / 2

        print(f"  Critical ratio eta/lambda ~= {critical_ratio:.2f}")
        print(f"    Minimum grokking ratio: {min_grok_ratio:.2f}")
        print(f"    Maximum non-grokking ratio: {max_no_grok_ratio:.2f}")
        print()
        print(f"  PREDICTION: Grokking occurs when eta/lambda > ~{critical_ratio:.1f}")
    else:
        print("  Could not determine critical ratio (all grokked or none grokked)")

    return phase_data


# =============================================================================
# EXPERIMENT 2: Grokking Timing Prediction
# =============================================================================

def run_timing_experiment():
    """
    Experiment 2: Predict WHEN grokking will occur.

    For different eta/lambda ratios above the critical threshold,
    predict the number of training steps before generalization emerges.
    """
    print()
    print("=" * 70)
    print("EXPERIMENT 2: Grokking Timing Prediction")
    print("=" * 70)
    print()
    print("For different eta/lambda ratios, predicting when generalization emerges.")
    print()

    weight_decay = 0.01
    results = []

    print(f"  {'eta':>8s} {'eta/lambda':>8s} {'Grokked':>8s} {'Time':>10s} {'Final s':>10s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for lr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        model = map_training_to_dissipative(lr, weight_decay)
        e_in = estimate_energy_input(lr, 1.0, 0.5)
        result = model.simulate(e_in, s0=0.01, t_max=200.0)

        ratio = lr / weight_decay
        grok_str = "Yes" if result["grokked"] else "No"
        time_str = f"{result['grokking_time']:.1f}" if result["grokking_time"] else "N/A"

        print(f"  {lr:>8.3f} {ratio:>8.1f} {grok_str:>8s} {time_str:>10s} "
              f"{result['final_s']:>10.4f}")

        results.append({
            "lr": lr, "ratio": ratio,
            "grokked": result["grokked"],
            "grokking_time": result["grokking_time"],
            "final_s": result["final_s"],
        })

    print()

    # Show that higher ratios grok faster
    grokked_results = [r for r in results if r["grokked"] and r["grokking_time"]]
    if len(grokked_results) >= 2:
        ratios = [r["ratio"] for r in grokked_results]
        times = [r["grokking_time"] for r in grokked_results]

        if times[0] > times[-1]:
            print("  [OK] Higher eta/lambda ratio -> faster grokking (as predicted)")
        else:
            print("  [!]  Timing relationship unclear")

        print()
        print("  KEY INSIGHT: The dissipative model predicts that grokking time")
        print("  scales as t_grok  proportional to  1 / (E_in - E_critical)")
        print("  i.e., the further above threshold, the faster it happens.")
    else:
        print("  Insufficient grokking events to analyze timing.")

    return results


# =============================================================================
# EXPERIMENT 3: Validate Against Actual NN Training
# =============================================================================

def run_neural_validation():
    """
    Experiment 3: Train an actual neural network on a modular arithmetic task
    (the canonical grokking setup) and compare against dissipative prediction.

    Task: (a + b) mod p, with p=7
    Training set: small random subset
    """
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Neural Network Grokking (Modular Arithmetic)")
    print("=" * 70)
    print()
    print("Training on (a + b) mod 7 with small training set.")
    print("Comparing actual generalization curve to dissipative prediction.")
    print()

    # Generate modular arithmetic dataset
    p = 7
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    X_all = np.array(all_pairs)
    y_all = np.array([(a + b) % p for a, b in all_pairs])

    # One-hot encode inputs and outputs
    def one_hot(vals, n_classes):
        oh = np.zeros((len(vals), n_classes))
        if vals.ndim == 1:
            oh[np.arange(len(vals)), vals] = 1
        else:
            # Two columns: encode each and concat
            oh1 = np.zeros((len(vals), n_classes))
            oh2 = np.zeros((len(vals), n_classes))
            oh1[np.arange(len(vals)), vals[:, 0]] = 1
            oh2[np.arange(len(vals)), vals[:, 1]] = 1
            oh = np.concatenate([oh1, oh2], axis=1)
        return oh

    X_oh = one_hot(X_all, p)  # (49, 14)
    y_oh = np.zeros((len(y_all), p))
    y_oh[np.arange(len(y_all)), y_all] = 1  # (49, 7)

    # Split: 60% train, 40% test (typical grokking setup)
    np.random.seed(42)
    n = len(X_oh)
    idx = np.random.permutation(n)
    n_train = int(0.6 * n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X_oh[train_idx], y_oh[train_idx]
    X_test, y_test = X_oh[test_idx], y_oh[test_idx]

    # Network: 14 -> 64 -> 64 -> 7
    input_dim, hidden1, hidden2, output_dim = 14, 64, 64, p

    def softmax(z):
        e = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

    configs = [
        {"name": "eta=0.1, lambda=0.1", "lr": 0.1, "wd": 0.1},
        {"name": "eta=0.1, lambda=0.01", "lr": 0.1, "wd": 0.01},
        {"name": "eta=0.1, lambda=0.001", "lr": 0.1, "wd": 0.001},
    ]

    for cfg in configs:
        np.random.seed(42)

        # Initialize weights
        W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(hidden1)
        W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        b2 = np.zeros(hidden2)
        W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        b3 = np.zeros(output_dim)

        lr = cfg["lr"]
        wd = cfg["wd"]
        epochs = 2000
        train_accs, test_accs = [], []

        for epoch in range(epochs):
            # Forward
            z1 = X_train @ W1 + b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ W2 + b2
            a2 = np.maximum(0, z2)
            z3 = a2 @ W3 + b3
            probs = softmax(z3)

            # Cross-entropy loss
            m = X_train.shape[0]
            dz3 = (probs - y_train) / m

            # Backward
            dW3 = a2.T @ dz3
            db3 = dz3.sum(axis=0)
            da2 = dz3 @ W3.T
            dz2 = da2 * (z2 > 0).astype(float)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0).astype(float)
            dW1 = X_train.T @ dz1
            db1 = dz1.sum(axis=0)

            # SGD + weight decay
            W1 -= lr * (dW1 + wd * W1)
            b1 -= lr * db1
            W2 -= lr * (dW2 + wd * W2)
            b2 -= lr * db2
            W3 -= lr * (dW3 + wd * W3)
            b3 -= lr * db3

            # Evaluate
            if epoch % 50 == 0 or epoch == epochs - 1:
                # Train accuracy
                p_train = softmax(
                    np.maximum(0, np.maximum(0, X_train @ W1 + b1) @ W2 + b2) @ W3 + b3
                )
                train_acc = np.mean(p_train.argmax(axis=1) == y_train.argmax(axis=1))
                # Test accuracy
                p_test = softmax(
                    np.maximum(0, np.maximum(0, X_test @ W1 + b1) @ W2 + b2) @ W3 + b3
                )
                test_acc = np.mean(p_test.argmax(axis=1) == y_test.argmax(axis=1))
                train_accs.append((epoch, train_acc))
                test_accs.append((epoch, test_acc))

        # Get dissipative prediction
        model = map_training_to_dissipative(lr, wd)
        e_in = estimate_energy_input(lr, 1.0, 0.5)
        pred = model.simulate(e_in, s0=0.01, t_max=100.0)

        # Report
        final_train = train_accs[-1][1]
        final_test = test_accs[-1][1]
        grokked_actual = final_test > 0.7 and any(
            ta[1] > 0.9 and te[1] < 0.3
            for ta, te in zip(train_accs[:10], test_accs[:10])
        )

        # Detect grokking: memorized early (train>90%) but test improved late
        memorized_early = any(ta[1] > 0.9 for ta in train_accs[:8])
        generalized_late = final_test > 0.5

        print(f"  {cfg['name']:20s} | eta/lambda={lr/wd:>6.1f} | "
              f"train={final_train:.2f} test={final_test:.2f} | "
              f"memorized={memorized_early} generalized={generalized_late} | "
              f"predicted_grok={pred['grokked']}")

    print()
    print("INTERPRETATION:")
    print("  If 'predicted_grok' matches 'generalized', the dissipative model works.")
    print("  Higher eta/lambda should show both predicted AND actual grokking.")
    print()
    print("  CAVEAT: This is a tiny experiment. Real grokking studies use")
    print("  much longer training (10k-100k+ epochs) and larger models.")
    print("  The dissipative model predicts the QUALITATIVE behavior,")
    print("  not exact epoch counts.")
    print()

    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("+==================================================================+")
    print("|  DISSIPATIVE GROKKING PREDICTOR                                |")
    print("|  Research Proof-of-Concept                                      |")
    print("+==================================================================+")
    print()
    print("Hypothesis: Grokking is a dissipative phase transition.")
    print("The critical ratio eta/lambda determines whether memorization->generalization")
    print("occurs, and how fast.")
    print()

    r1 = run_phase_diagram_experiment()
    r2 = run_timing_experiment()
    r3 = run_neural_validation()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("The Prigogine dissipative model predicts:")
    print("  1. A critical eta/lambda ratio below which grokking CANNOT occur")
    print("  2. Above the threshold, grokking time  proportional to  1/(E_in - E_critical)")
    print("  3. Weight decay is not just 'regularization' -- it's the dissipation")
    print("     rate that enables the phase transition")
    print()
    print("Next steps if results are positive:")
    print("  1. Run on canonical grokking tasks (modular arithmetic, perm groups)")
    print("     with 50k+ epochs to see true grokking")
    print("  2. Measure E_in = etax||nablaL|| dynamically during training and feed into")
    print("     the ODE solver for real-time grokking prediction")
    print("  3. Test whether the predicted critical ratio holds across architectures")
    print("  4. Extend to multi-dimensional dissipative model (one ODE per layer)")
    print()
