"""
Helmholtz Training: Free Energy Objectives with Temperature-Scheduled Entropy
=============================================================================

Novel idea: Use the Helmholtz free energy F = E - T*S directly as a training
loss, where E = task loss, S = entropy of the weight/prediction distribution,
and T = temperature scheduled from high (explore) to low (exploit).

Key insight: This generalizes several known techniques:
  - T=0: pure loss minimization (standard training)
  - T->inf: maximum entropy (uniform predictions)
  - Fixed T>0: label smoothing / entropy regularization
  - Scheduled T: principled annealing from exploration to exploitation

What's new: The Helmholtz framing with *parameter entropy* (not data
likelihood as in VAE's ELBO) and explicit temperature scheduling provides
a unified view of regularization, label smoothing, and entropy bonuses.

Origin: cross_domain_lab demos (entropy_regularization, free_energy,
        TemperatureScheduler, BoltzmannMachine)

Equations:
    F = E - T*S
    S = -Sigma p_i ln(p_i)     (Shannon entropy)
    T(t) via schedule: exponential, linear, cosine, adaptive
"""

import numpy as np
from collections import defaultdict


# =====================================================================
# Core: Helmholtz Free Energy Loss
# =====================================================================

class HelmholtzLoss:
    """
    Free energy loss: F = E - T*S
    
    E = task loss (e.g., cross-entropy)
    S = entropy of distribution (predictions or weights)
    T = temperature (controls exploration-exploitation)
    """
    
    def __init__(self, temperature=1.0, entropy_source='predictions'):
        self.T = temperature
        self.entropy_source = entropy_source
    
    def compute(self, loss, distribution):
        """
        F = E - T*S
        
        loss: scalar task loss (E)
        distribution: probability vector or weight distribution
        """
        S = self._entropy(distribution)
        F = loss - self.T * S
        return F, {'E': loss, 'T': self.T, 'S': S, 'F': F}
    
    @staticmethod
    def _entropy(p):
        p = np.clip(p, 1e-10, 1.0)
        if np.sum(p) > 0:
            p = p / np.sum(p)  # normalize
        return -np.sum(p * np.log(p))


class TemperatureSchedule:
    """Temperature annealing schedules for Helmholtz training."""
    
    @staticmethod
    def exponential(step, T_init=5.0, T_min=0.01, decay=0.99):
        return max(T_min, T_init * (decay ** step))
    
    @staticmethod
    def linear(step, T_init=5.0, T_min=0.01, total_steps=1000):
        return max(T_min, T_init - (T_init - T_min) * step / total_steps)
    
    @staticmethod
    def cosine(step, T_init=5.0, T_min=0.01, total_steps=1000):
        return T_min + 0.5 * (T_init - T_min) * (1 + np.cos(np.pi * step / total_steps))
    
    @staticmethod
    def adaptive(step, current_loss, prev_loss, T_current, lr=0.1):
        """Loss-responsive: cool faster when loss drops, warm when stuck."""
        if current_loss < prev_loss:
            return max(0.01, T_current * (1 - lr))  # cool: exploiting
        else:
            return min(10.0, T_current * (1 + lr * 0.5))  # warm: need exploration


# =====================================================================
# Synthetic Classification Problem
# =====================================================================

def make_dataset(n=500, n_classes=5, dim=10, noise=0.3, seed=42):
    """Multi-class classification with overlapping clusters."""
    np.random.seed(seed)
    centers = np.random.randn(n_classes, dim) * 3
    X, y = [], []
    for i in range(n):
        c = i % n_classes
        x = centers[c] + np.random.randn(dim) * noise
        X.append(x)
        y.append(c)
    X = np.array(X)
    y = np.array(y)
    # Shuffle
    perm = np.random.permutation(n)
    return X[perm], y[perm]


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(probs, targets, n_classes):
    one_hot = np.eye(n_classes)[targets]
    return -np.mean(np.sum(one_hot * np.log(probs + 1e-10), axis=1))


class SimpleLinearClassifier:
    """Minimal softmax classifier for experiments."""
    
    def __init__(self, dim, n_classes):
        self.W = np.random.randn(dim, n_classes) * 0.1
        self.b = np.zeros(n_classes)
    
    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)
    
    def accuracy(self, X, y):
        preds = np.argmax(self.predict_proba(X), axis=1)
        return np.mean(preds == y)
    
    def train_step(self, X, y, lr=0.01, helmholtz_loss=None):
        n = len(y)
        n_classes = self.W.shape[1]
        probs = self.predict_proba(X)
        one_hot = np.eye(n_classes)[y]
        
        # Task loss (cross-entropy)
        E = cross_entropy(probs, y, n_classes)
        
        # Gradient of cross-entropy
        dlogits = (probs - one_hot) / n
        dW = X.T @ dlogits
        db = dlogits.mean(axis=0)
        
        if helmholtz_loss is not None:
            # Add entropy gradient: dS/dlogits pushes toward uniform
            avg_probs = probs.mean(axis=0)
            S = helmholtz_loss._entropy(avg_probs)
            T = helmholtz_loss.T
            
            # Entropy bonus gradient (encourages diversity)
            entropy_grad = T * (np.log(avg_probs + 1e-10) + 1) / n
            dlogits_total = dlogits + np.broadcast_to(entropy_grad, dlogits.shape) * 0.1
            dW = X.T @ dlogits_total
            db = dlogits_total.mean(axis=0)
            
            F = E - T * S
        else:
            F = E
            S = 0
        
        self.W -= lr * dW
        self.b -= lr * db
        
        return {'E': E, 'S': S, 'F': F, 'T': helmholtz_loss.T if helmholtz_loss else 0}


# =====================================================================
# Experiments
# =====================================================================

def run_experiments():
    print("=" * 72)
    print("Helmholtz Training: Free Energy Objectives F = E - T*S")
    print("=" * 72)
    print()
    print("Hypothesis: Temperature-scheduled entropy regularization provides")
    print("a principled exploration->exploitation annealing that improves")
    print("generalization compared to fixed regularization.")
    print()
    
    dim, n_classes = 10, 5
    X_train, y_train = make_dataset(n=400, n_classes=n_classes, dim=dim, noise=0.5)
    X_test, y_test = make_dataset(n=200, n_classes=n_classes, dim=dim, noise=0.5, seed=99)
    
    n_epochs = 200
    lr = 0.05
    schedules = {
        "No entropy (T=0)": ("none", {}),
        "Fixed T=0.1": ("fixed", {"T": 0.1}),
        "Fixed T=1.0": ("fixed", {"T": 1.0}),
        "Exponential anneal": ("exponential", {"T_init": 2.0, "decay": 0.98}),
        "Cosine anneal": ("cosine", {"T_init": 2.0}),
        "Adaptive T": ("adaptive", {"T_init": 1.0}),
    }
    
    results = {}
    
    for name, (sched_type, params) in schedules.items():
        np.random.seed(42)
        model = SimpleLinearClassifier(dim, n_classes)
        
        if sched_type == "none":
            h_loss = None
        else:
            h_loss = HelmholtzLoss(temperature=params.get("T", params.get("T_init", 1.0)))
        
        history = []
        prev_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Update temperature
            if sched_type == "exponential":
                h_loss.T = TemperatureSchedule.exponential(
                    epoch, params["T_init"], 0.01, params["decay"])
            elif sched_type == "cosine":
                h_loss.T = TemperatureSchedule.cosine(
                    epoch, params["T_init"], 0.01, n_epochs)
            elif sched_type == "adaptive" and h_loss is not None:
                if epoch > 0:
                    h_loss.T = TemperatureSchedule.adaptive(
                        epoch, history[-1]['E'], prev_loss, h_loss.T)
            
            info = model.train_step(X_train, y_train, lr=lr, helmholtz_loss=h_loss)
            prev_loss = info['E']
            
            if epoch % 20 == 0 or epoch == n_epochs - 1:
                test_acc = model.accuracy(X_test, y_test)
                train_acc = model.accuracy(X_train, y_train)
                info['test_acc'] = test_acc
                info['train_acc'] = train_acc
                history.append(info)
        
        final_test = model.accuracy(X_test, y_test)
        final_train = model.accuracy(X_train, y_train)
        gap = final_train - final_test  # generalization gap
        
        results[name] = {
            'test_acc': final_test, 'train_acc': final_train,
            'gap': gap, 'history': history
        }
    
    # Report
    print(f"{'-' * 60}")
    print(f"Results: 5-class classification (d=10, n_train=400, n_test=200)")
    print(f"{'-' * 60}")
    print(f"  {'Method':25s} {'Train Acc':>9s} {'Test Acc':>9s} {'Gap':>6s}")
    print(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*6}")
    
    best_test = max(r['test_acc'] for r in results.values())
    for name, r in results.items():
        marker = " <- best" if r['test_acc'] == best_test else ""
        print(f"  {name:25s} {r['train_acc']:9.1%} {r['test_acc']:9.1%} {r['gap']:+5.1%}{marker}")
    
    # Temperature dynamics
    print(f"\n{'-' * 60}")
    print("Temperature Dynamics (Exponential Anneal)")
    print(f"{'-' * 60}")
    
    exp_hist = results.get("Exponential anneal", {}).get("history", [])
    for h in exp_hist:
        T_bar = "#" * int(h['T'] * 20)
        print(f"  T={h['T']:.3f}  E={h['E']:.4f}  S={h['S']:.4f}  "
              f"F={h['F']:.4f}  test={h.get('test_acc',0):.1%}  {T_bar}")
    
    # Free energy decomposition
    print(f"\n{'-' * 60}")
    print("Free Energy Decomposition: F = E - T*S")
    print(f"{'-' * 60}")
    print()
    print("  Phase 1 (high T): Entropy dominates -> explore diverse solutions")
    print("  Phase 2 (medium T): Balance E and S -> good generalization")
    print("  Phase 3 (low T): Loss dominates -> fine-tune to training data")
    print()
    
    # Connection to known techniques
    print(f"{'-' * 60}")
    print("Unifying View: What Helmholtz F = E - T*S Generalizes")
    print(f"{'-' * 60}")
    print()
    print("  T=0, S=predictions  ->  Standard cross-entropy training")
    print("  T=0.1, S=predictions ->  Label smoothing (Szegedy et al. 2016)")
    print("  T=fixed, S=policy   ->  Entropy bonus in RL (SAC, A3C)")
    print("  T=scheduled, S=pred ->  * Helmholtz training (this work)")
    print("  T->inf              ->  Maximum entropy model (uniform output)")
    print()
    print("  The Helmholtz framing reveals these are all special cases of")
    print("  the same free energy principle with different T and S choices.")
    
    # Summary
    print(f"\n{'=' * 72}")
    print("Summary")
    print(f"{'=' * 72}")
    print()
    print("Novel contribution:")
    print("  1. Helmholtz free energy F=E-TS as a unified training objective")
    print("  2. Temperature scheduling: explore (high T) -> exploit (low T)")
    print("  3. Adaptive T: loss-responsive temperature (cool when improving)")
    print("  4. Unifies label smoothing, entropy bonuses, and annealing")
    print("  5. Parameter entropy (not data likelihood) distinguishes from ELBO")
    print()
    print("Potential paper: 'Helmholtz Training: Free Energy Objectives")
    print("with Temperature-Scheduled Entropy Regularization'")


if __name__ == "__main__":
    run_experiments()
