"""Demos for the Simulation & Modeling Lab."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def sm_estimate_pi():
    """Estimate π using Monte Carlo dart throwing."""
    try:
        np.random.seed(42)
        samples = [100, 1_000, 10_000, 100_000]
        out = "Monte Carlo π Estimation\n" + "=" * 50 + "\n\n"
        for n in samples:
            pts = np.random.rand(n, 2)
            inside = np.sum(pts[:, 0]**2 + pts[:, 1]**2 <= 1)
            pi_est = 4 * inside / n
            error = abs(pi_est - np.pi)
            out += f"  n={n:>7,d}  →  π ≈ {pi_est:.6f}  (error={error:.6f})\n"
        out += f"\n  True π = {np.pi:.6f}"
        out += "\n  Convergence rate: O(1/√n) — need 100× samples for 10× accuracy."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_mc_integration():
    """Monte Carlo integration with variance reduction."""
    try:
        np.random.seed(42)
        n = 50_000
        # Basic MC: ∫sin(x) dx from 0 to π
        x_basic = np.random.uniform(0, np.pi, n)
        basic_est = np.pi * np.mean(np.sin(x_basic))
        basic_var = np.var(np.pi * np.sin(x_basic))

        # Antithetic variates
        u = np.random.uniform(0, np.pi, n // 2)
        anti = np.pi * np.mean((np.sin(u) + np.sin(np.pi - u)) / 2)
        anti_var = np.var(np.pi * (np.sin(u) + np.sin(np.pi - u)) / 2)

        # Stratified
        strata = 10
        strat_est = 0
        for i in range(strata):
            lo, hi = i * np.pi / strata, (i + 1) * np.pi / strata
            xs = np.random.uniform(lo, hi, n // strata)
            strat_est += (hi - lo) * np.mean(np.sin(xs))

        exact = 2.0
        out = "Monte Carlo Integration: ∫sin(x)dx from 0 to π\n" + "=" * 50 + "\n\n"
        out += f"  Exact answer: {exact:.6f}\n\n"
        out += f"  Basic MC        : {basic_est:.6f}  (var={basic_var:.4f})\n"
        out += f"  Antithetic      : {anti:.6f}  (var={anti_var:.4f})\n"
        out += f"  Stratified (k=10): {strat_est:.6f}\n\n"
        out += "  Variance reduction techniques improve estimate quality."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_mcmc():
    """Metropolis-Hastings MCMC sampling."""
    try:
        np.random.seed(42)
        # Target: mixture of two Gaussians
        def target_log_prob(x):
            p1 = 0.4 * np.exp(-0.5 * ((x - 2) / 0.8) ** 2)
            p2 = 0.6 * np.exp(-0.5 * ((x + 1) / 1.2) ** 2)
            return np.log(p1 + p2 + 1e-300)

        n_samples = 10_000
        burn_in = 1_000
        samples = []
        x = 0.0
        accepted = 0

        for i in range(n_samples + burn_in):
            x_prop = x + np.random.normal(0, 1.5)
            log_alpha = target_log_prob(x_prop) - target_log_prob(x)
            if np.log(np.random.rand()) < log_alpha:
                x = x_prop
                if i >= burn_in:
                    accepted += 1
            if i >= burn_in:
                samples.append(x)

        samples = np.array(samples)
        out = "MCMC: Metropolis-Hastings\n" + "=" * 50 + "\n"
        out += "Target: Mixture of Gaussians (40% at μ=2, 60% at μ=-1)\n\n"
        out += f"  Samples collected: {len(samples)}\n"
        out += f"  Acceptance rate: {accepted / n_samples:.1%}\n"
        out += f"  Sample mean: {np.mean(samples):.3f}\n"
        out += f"  Sample std: {np.std(samples):.3f}\n"
        out += f"  Mode region 1 (x<0.5): mean={np.mean(samples[samples<0.5]):.2f}\n"
        out += f"  Mode region 2 (x>0.5): mean={np.mean(samples[samples>0.5]):.2f}\n"
        out += "\n  MCMC correctly finds both modes of the bimodal distribution."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_logistic_map():
    """Explore the logistic map and route to chaos."""
    try:
        out = "Logistic Map: x(n+1) = r·x(n)·(1-x(n))\n" + "=" * 50 + "\n\n"
        r_values = [2.5, 3.2, 3.5, 3.9]
        labels = ["Fixed point", "Period-2 cycle", "Period-4 cycle", "Chaos"]

        for r, label in zip(r_values, labels):
            x = 0.5
            # Skip transients
            for _ in range(200):
                x = r * x * (1 - x)
            # Collect orbit
            orbit = []
            for _ in range(16):
                x = r * x * (1 - x)
                orbit.append(round(x, 4))
            unique = sorted(set(round(v, 2) for v in orbit))
            out += f"  r={r:.1f} ({label}):\n"
            out += f"    Last 8: {orbit[:8]}\n"
            out += f"    Unique values: ~{len(unique)}\n\n"
        out += "As r increases: fixed point → period-2 → period-4 → ... → chaos."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_lorenz():
    """Simulate the Lorenz attractor."""
    try:
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        dt = 0.01
        steps = 5000

        # Two trajectories with tiny initial difference
        x1, y1, z1 = 1.0, 1.0, 1.0
        x2, y2, z2 = 1.0 + 1e-10, 1.0, 1.0

        divergence = []
        for i in range(steps):
            dx1 = sigma * (y1 - x1) * dt
            dy1 = (x1 * (rho - z1) - y1) * dt
            dz1 = (x1 * y1 - beta * z1) * dt
            x1 += dx1; y1 += dy1; z1 += dz1

            dx2 = sigma * (y2 - x2) * dt
            dy2 = (x2 * (rho - z2) - y2) * dt
            dz2 = (x2 * y2 - beta * z2) * dt
            x2 += dx2; y2 += dy2; z2 += dz2

            if i % 500 == 0:
                d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                divergence.append((i * dt, d))

        out = "Lorenz Attractor (σ=10, ρ=28, β=8/3)\n" + "=" * 50 + "\n\n"
        out += "Two trajectories starting 1e-10 apart:\n"
        for t, d in divergence:
            bar = "█" * min(50, int(np.log10(d + 1e-15) + 15))
            out += f"  t={t:5.1f}  divergence={d:.2e}  {bar}\n"
        out += f"\nFinal state 1: ({x1:.2f}, {y1:.2f}, {z1:.2f})"
        out += f"\nFinal state 2: ({x2:.2f}, {y2:.2f}, {z2:.2f})"
        out += "\n\nButterfly effect: tiny differences → completely different trajectories."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_predator_prey():
    """Lotka-Volterra predator-prey simulation."""
    try:
        alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
        dt = 0.01
        steps = 5000
        prey, pred = 10.0, 5.0
        history = []

        for i in range(steps):
            dp = (alpha * prey - beta * prey * pred) * dt
            dd = (delta * prey * pred - gamma * pred) * dt
            prey += dp
            pred += dd
            prey = max(0, prey)
            pred = max(0, pred)
            if i % 500 == 0:
                history.append((i * dt, prey, pred))

        out = "Lotka-Volterra Predator-Prey Model\n" + "=" * 50 + "\n"
        out += f"α={alpha}, β={beta}, δ={delta}, γ={gamma}\n\n"
        out += f"  {'Time':>6s}  {'Prey':>8s}  {'Predator':>8s}\n"
        out += f"  {'-'*6}  {'-'*8}  {'-'*8}\n"
        for t, pr, pd in history:
            prey_bar = "🐇" * min(10, int(pr))
            pred_bar = "🐺" * min(10, int(pd))
            out += f"  t={t:5.1f}  {pr:8.2f}  {pd:8.2f}  {prey_bar}{pred_bar}\n"
        out += "\nOscillatory dynamics: prey up → predators up → prey down → predators down → repeat."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_game_of_life():
    """Conway's Game of Life cellular automaton."""
    try:
        np.random.seed(42)
        size = 10
        grid = np.random.choice([0, 1], (size, size), p=[0.65, 0.35])

        def step(g):
            n = np.zeros_like(g)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    total = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = (i + di) % size, (j + dj) % size
                            total += g[ni, nj]
                    if g[i, j] == 1:
                        n[i, j] = 1 if total in (2, 3) else 0
                    else:
                        n[i, j] = 1 if total == 3 else 0
            return n

        out = "Conway's Game of Life (10×10 toroidal grid)\n" + "=" * 50 + "\n\n"
        for gen in range(5):
            alive = int(np.sum(grid))
            out += f"  Generation {gen} ({alive} alive):\n"
            for row in grid:
                out += "    " + " ".join("█" if c else "·" for c in row) + "\n"
            out += "\n"
            grid = step(grid)
        out += "Rules: Born with 3 neighbors, survive with 2-3, else die."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_flocking():
    """Boids flocking simulation."""
    try:
        np.random.seed(42)
        n_boids = 15
        pos = np.random.rand(n_boids, 2) * 100
        vel = np.random.randn(n_boids, 2) * 2

        out = "Boids Flocking Simulation\n" + "=" * 50 + "\n"
        out += "Rules: Separation + Alignment + Cohesion\n\n"

        for step_num in range(5):
            new_vel = vel.copy()
            for i in range(n_boids):
                dists = np.linalg.norm(pos - pos[i], axis=1)
                neighbors = (dists > 0) & (dists < 30)
                if np.any(neighbors):
                    center = pos[neighbors].mean(axis=0)
                    cohesion = (center - pos[i]) * 0.01
                    avg_vel = vel[neighbors].mean(axis=0)
                    alignment = (avg_vel - vel[i]) * 0.05
                    close = (dists > 0) & (dists < 10)
                    if np.any(close):
                        separation = -(pos[close] - pos[i]).mean(axis=0) * 0.1
                    else:
                        separation = np.zeros(2)
                    new_vel[i] += cohesion + alignment + separation

            speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
            new_vel = np.where(speed > 5, new_vel / speed * 5, new_vel)
            vel = new_vel
            pos = (pos + vel) % 100

            spread = np.std(pos, axis=0).mean()
            avg_heading = np.mean(np.arctan2(vel[:, 1], vel[:, 0]))
            heading_std = np.std(np.arctan2(vel[:, 1], vel[:, 0]))
            out += f"  Step {step_num}: spread={spread:.1f}, heading_alignment={1-heading_std/np.pi:.2f}\n"

        out += f"\n  Final positions (first 5 boids):\n"
        for i in range(5):
            out += f"    Boid {i}: pos=({pos[i,0]:.1f},{pos[i,1]:.1f}) vel=({vel[i,0]:.1f},{vel[i,1]:.1f})\n"
        out += "\nEmergent flocking from simple local rules."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_genetic_algorithm():
    """Genetic algorithm solving a simple optimization."""
    try:
        np.random.seed(42)
        # Maximize f(x) = -x² + 10x (optimum at x=5, f=25)
        pop_size, n_genes, n_gens = 20, 8, 15
        pop = np.random.randint(0, 2, (pop_size, n_genes))

        def decode(chrom):
            return sum(b * 2**i for i, b in enumerate(chrom)) * 10 / 255

        def fitness(x):
            return -(x ** 2) + 10 * x

        out = "Genetic Algorithm: Maximize f(x) = -x² + 10x\n" + "=" * 50 + "\n"
        out += "Optimum: x=5.0, f(5)=25.0\n\n"

        for gen in range(n_gens):
            decoded = np.array([decode(p) for p in pop])
            fits = np.array([fitness(x) for x in decoded])
            best_idx = np.argmax(fits)
            best_x, best_f = decoded[best_idx], fits[best_idx]

            if gen % 3 == 0 or gen == n_gens - 1:
                out += f"  Gen {gen:2d}: best x={best_x:.3f}, f(x)={best_f:.3f}, avg_fitness={np.mean(fits):.3f}\n"

            # Selection (tournament)
            new_pop = []
            for _ in range(pop_size):
                i, j = np.random.randint(0, pop_size, 2)
                winner = i if fits[i] > fits[j] else j
                new_pop.append(pop[winner].copy())

            # Crossover
            for i in range(0, pop_size - 1, 2):
                if np.random.rand() < 0.8:
                    cx = np.random.randint(1, n_genes)
                    new_pop[i][cx:], new_pop[i+1][cx:] = new_pop[i+1][cx:].copy(), new_pop[i][cx:].copy()

            # Mutation
            for i in range(pop_size):
                for j in range(n_genes):
                    if np.random.rand() < 0.05:
                        new_pop[i][j] = 1 - new_pop[i][j]

            pop = np.array(new_pop)

        out += f"\nEvolution converges toward x ≈ 5.0, f(x) ≈ 25.0"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_annealing():
    """Simulated annealing for function minimization."""
    try:
        np.random.seed(42)
        # Minimize Rastrigin-like: f(x) = x² - 10cos(2πx) + 10
        def f(x):
            return x**2 - 10 * np.cos(2 * np.pi * x) + 10

        x = np.random.uniform(-5, 5)
        best_x, best_f = x, f(x)
        T = 10.0
        cooling = 0.995
        history = []

        for i in range(2000):
            x_new = x + np.random.normal(0, 0.5)
            x_new = np.clip(x_new, -5, 5)
            delta = f(x_new) - f(x)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                x = x_new
            if f(x) < best_f:
                best_x, best_f = x, f(x)
            T *= cooling
            if i % 400 == 0:
                history.append((i, T, x, f(x), best_x, best_f))

        out = "Simulated Annealing: Rastrigin-like function\n" + "=" * 50 + "\n"
        out += "f(x) = x² - 10cos(2πx) + 10  (global min at x=0, f=0)\n\n"
        for step, temp, cx, cf, bx, bf in history:
            out += f"  Step {step:4d}: T={temp:6.2f}  current=({cx:+.3f}, {cf:.3f})  best=({bx:+.3f}, {bf:.3f})\n"
        out += f"\n  Final best: x={best_x:+.4f}, f(x)={best_f:.4f}"
        out += "\n  High T = explore (accept worse), Low T = exploit (only accept better)."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_quantum_walk():
    """Compare classical vs quantum-inspired random walk."""
    try:
        np.random.seed(42)
        n_steps = 100
        n_walkers = 5000

        # Classical random walk
        classical = np.zeros(n_walkers)
        for _ in range(n_steps):
            classical += np.random.choice([-1, 1], n_walkers)

        # Quantum-inspired: use superposition-like state vector
        positions = np.arange(-n_steps, n_steps + 1)
        state = np.zeros(len(positions), dtype=complex)
        state[n_steps] = 1.0  # start at 0

        coin = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        full_state = np.zeros((2, len(positions)), dtype=complex)
        full_state[0, n_steps] = 1 / np.sqrt(2)
        full_state[1, n_steps] = 1j / np.sqrt(2)

        for _ in range(n_steps):
            new = np.zeros_like(full_state)
            new[0, 1:] += coin[0, 0] * full_state[0, :-1] + coin[0, 1] * full_state[1, :-1]
            new[1, :-1] += coin[1, 0] * full_state[0, 1:] + coin[1, 1] * full_state[1, 1:]
            full_state = new

        prob = np.abs(full_state[0])**2 + np.abs(full_state[1])**2
        q_mean = np.sum(positions * prob)
        q_std = np.sqrt(np.sum(positions**2 * prob) - q_mean**2)

        out = f"Classical vs Quantum-Inspired Walk ({n_steps} steps)\n" + "=" * 50 + "\n\n"
        out += f"  Classical ({n_walkers} walkers):\n"
        out += f"    Mean position: {np.mean(classical):.2f}\n"
        out += f"    Spread (std): {np.std(classical):.2f}\n"
        out += f"    Expected spread: √n = {np.sqrt(n_steps):.2f}\n\n"
        out += f"  Quantum-Inspired:\n"
        out += f"    Mean position: {q_mean:.2f}\n"
        out += f"    Spread (std): {q_std:.2f}\n"
        out += f"    Expected spread: ~n = {n_steps}\n\n"
        out += f"  Speedup factor: {q_std / np.std(classical):.1f}×"
        out += "\n  Quantum walks spread linearly (ballistic) vs classically (diffusive)."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def sm_multiverse():
    """Run parallel simulations with different parameters."""
    try:
        np.random.seed(42)
        # Simulate a simple model under different "universes"
        universes = [
            {"name": "Conservative", "growth": 0.02, "volatility": 0.05},
            {"name": "Moderate", "growth": 0.05, "volatility": 0.15},
            {"name": "Aggressive", "growth": 0.10, "volatility": 0.30},
            {"name": "Crisis", "growth": -0.03, "volatility": 0.40},
            {"name": "Boom", "growth": 0.15, "volatility": 0.20},
        ]

        n_steps = 100
        out = "Multiverse Simulation\n" + "=" * 50 + "\n"
        out += f"Running {len(universes)} parallel universes, {n_steps} timesteps each\n\n"

        results = []
        for u in universes:
            value = 100.0
            path = [value]
            for _ in range(n_steps):
                ret = u["growth"] / n_steps + u["volatility"] / np.sqrt(n_steps) * np.random.randn()
                value *= (1 + ret)
                path.append(value)
            results.append({**u, "final": value, "max": max(path), "min": min(path)})

        for r in results:
            bar_len = int(max(0, min(30, r["final"] / 10)))
            bar = "█" * bar_len
            out += f"  [{r['name']:>12s}] Final={r['final']:7.1f}  Min={r['min']:7.1f}  Max={r['max']:7.1f}  {bar}\n"

        best = max(results, key=lambda r: r["final"])
        worst = min(results, key=lambda r: r["final"])
        out += f"\n  Best universe: {best['name']} (final={best['final']:.1f})"
        out += f"\n  Worst universe: {worst['name']} (final={worst['final']:.1f})"
        out += "\n\n  Multiverse thinking: explore parameter space in parallel."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


# --- Demo dispatcher ---
DEMO_HANDLERS = {
    "sm_estimate_pi": sm_estimate_pi,
    "sm_mc_integration": sm_mc_integration,
    "sm_mcmc": sm_mcmc,
    "sm_logistic_map": sm_logistic_map,
    "sm_lorenz": sm_lorenz,
    "sm_predator_prey": sm_predator_prey,
    "sm_game_of_life": sm_game_of_life,
    "sm_flocking": sm_flocking,
    "sm_genetic_algorithm": sm_genetic_algorithm,
    "sm_annealing": sm_annealing,
    "sm_quantum_walk": sm_quantum_walk,
    "sm_multiverse": sm_multiverse,
}

def run_demo(demo_id: str) -> dict:
    handler = DEMO_HANDLERS.get(demo_id)
    if handler:
        return handler()
    return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
