"""Curriculum for the Simulation & Modeling Lab."""

LEVELS = ["beginner", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "montecarlo", "name": "Monte Carlo & Stochastic Methods",
     "short": "MC", "color": "#e74c3c"},
    {"id": "dynamical", "name": "Dynamical Systems & Chaos",
     "short": "DS", "color": "#3498db"},
    {"id": "agentbased", "name": "Agent-Based & Evolutionary Models",
     "short": "AB", "color": "#2ecc71"},
    {"id": "quantum", "name": "Quantum-Inspired & Advanced Simulation",
     "short": "QI", "color": "#9b59b6"},
]

CURRICULUM = [
    # --- Monte Carlo & Stochastic Methods ---
    {
        "id": "mc_basics",
        "book_id": "montecarlo",
        "level": "beginner",
        "title": "Monte Carlo Estimation",
        "learn": "Use random sampling to estimate quantities like π. Understand the law of large numbers and convergence.",
        "try_code": "import numpy as np\\nnp.random.seed(42)\\npts = np.random.rand(100000, 2)\\npi_est = 4 * np.mean(np.sum(pts**2, axis=1) <= 1)\\nprint(f'π ≈ {pi_est:.4f}')",
        "try_demo": "sm_estimate_pi",
    },
    {
        "id": "mc_integration",
        "book_id": "montecarlo",
        "level": "intermediate",
        "title": "Monte Carlo Integration",
        "learn": "Compute integrals by random sampling. Compare variance reduction techniques: importance sampling, stratified sampling.",
        "try_code": "import numpy as np\\nnp.random.seed(42)\\nx = np.random.uniform(0, np.pi, 10000)\\nI = np.pi * np.mean(np.sin(x))\\nprint(f'∫sin(x)dx from 0 to π ≈ {I:.4f} (exact=2.0)')",
        "try_demo": "sm_mc_integration",
    },
    {
        "id": "mc_markov_chain",
        "book_id": "montecarlo",
        "level": "advanced",
        "title": "Markov Chain Monte Carlo (MCMC)",
        "learn": "Sample from complex distributions using Metropolis-Hastings. Understand burn-in, mixing, and convergence diagnostics.",
        "try_code": "# MCMC samples from a target distribution\\nimport numpy as np\\nprint('Metropolis-Hastings for Bayesian inference')",
        "try_demo": "sm_mcmc",
    },
    {
        "id": "mc_rare_events",
        "book_id": "montecarlo",
        "level": "expert",
        "title": "Rare Event Simulation",
        "learn": "Estimate extremely low probabilities using importance sampling, cross-entropy method, and splitting techniques.",
        "try_code": None,
        "try_demo": None,
    },
    # --- Dynamical Systems & Chaos ---
    {
        "id": "ds_logistic_map",
        "book_id": "dynamical",
        "level": "beginner",
        "title": "The Logistic Map",
        "learn": "Explore how a simple equation x(n+1) = r·x(n)·(1−x(n)) produces fixed points, oscillations, and chaos.",
        "try_code": "r, x = 3.9, 0.5\\nfor i in range(20):\\n    x = r * x * (1 - x)\\n    print(f'  Step {i+1}: x = {x:.4f}')",
        "try_demo": "sm_logistic_map",
    },
    {
        "id": "ds_lorenz",
        "book_id": "dynamical",
        "level": "intermediate",
        "title": "Lorenz Attractor & Chaos",
        "learn": "Simulate the Lorenz system — the original chaos model. Visualize sensitive dependence on initial conditions.",
        "try_code": "# Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz\\nprint('σ=10, ρ=28, β=8/3 → chaotic attractor')",
        "try_demo": "sm_lorenz",
    },
    {
        "id": "ds_predator_prey",
        "book_id": "dynamical",
        "level": "advanced",
        "title": "Predator-Prey Dynamics",
        "learn": "Simulate Lotka-Volterra equations. Observe oscillatory population dynamics and phase portraits.",
        "try_code": "print('Lotka-Volterra: dx/dt = αx - βxy, dy/dt = δxy - γy')",
        "try_demo": "sm_predator_prey",
    },
    {
        "id": "ds_bifurcation",
        "book_id": "dynamical",
        "level": "expert",
        "title": "Bifurcation Analysis",
        "learn": "Map how system behavior changes as parameters vary. Identify period-doubling routes to chaos.",
        "try_code": None,
        "try_demo": None,
    },
    # --- Agent-Based & Evolutionary Models ---
    {
        "id": "ab_conway",
        "book_id": "agentbased",
        "level": "beginner",
        "title": "Conway's Game of Life",
        "learn": "Simple rules create emergent complexity. Explore gliders, oscillators, and still lifes in a cellular automaton.",
        "try_code": "import numpy as np\\ngrid = np.random.choice([0,1], (5,5), p=[0.7,0.3])\\nprint('Random 5×5 grid:')\\nfor row in grid:\\n    print(' '.join('█' if c else '·' for c in row))",
        "try_demo": "sm_game_of_life",
    },
    {
        "id": "ab_flocking",
        "book_id": "agentbased",
        "level": "intermediate",
        "title": "Flocking & Swarm Behavior",
        "learn": "Implement Boids: separation, alignment, cohesion. Simple local rules produce global swarm behavior.",
        "try_code": "print('Boids rules: Separation + Alignment + Cohesion = Flocking')",
        "try_demo": "sm_flocking",
    },
    {
        "id": "ab_genetic",
        "book_id": "agentbased",
        "level": "advanced",
        "title": "Genetic Algorithm Optimization",
        "learn": "Evolve solutions through selection, crossover, and mutation. Solve optimization problems nature's way.",
        "try_code": "import numpy as np\\nprint('GA: Initialize → Fitness → Select → Crossover → Mutate → Repeat')",
        "try_demo": "sm_genetic_algorithm",
    },
    {
        "id": "ab_coevolution",
        "book_id": "agentbased",
        "level": "expert",
        "title": "Co-evolutionary Arms Races",
        "learn": "Two populations evolve in response to each other (Red Queen dynamics). Predator-prey, host-parasite co-evolution.",
        "try_code": None,
        "try_demo": None,
    },
    # --- Quantum-Inspired & Advanced Simulation ---
    {
        "id": "qi_annealing",
        "book_id": "quantum",
        "level": "beginner",
        "title": "Simulated Annealing",
        "learn": "Escape local optima by 'cooling' a temperature parameter. Inspired by metallurgical annealing.",
        "try_code": "import numpy as np\\nnp.random.seed(42)\\nprint('Simulated Annealing: high temp = explore, low temp = exploit')",
        "try_demo": "sm_annealing",
    },
    {
        "id": "qi_quantum_walk",
        "book_id": "quantum",
        "level": "intermediate",
        "title": "Quantum-Inspired Random Walks",
        "learn": "Compare classical random walks with quantum-inspired walks that use superposition-like state vectors.",
        "try_code": "import numpy as np\\nprint('Quantum walk: spread ∝ t  vs  Classical walk: spread ∝ √t')",
        "try_demo": "sm_quantum_walk",
    },
    {
        "id": "qi_multiverse",
        "book_id": "quantum",
        "level": "advanced",
        "title": "Multiverse Simulation",
        "learn": "Run parallel simulations exploring different parameter spaces simultaneously. Bayesian parallel worlds.",
        "try_code": "print('Multiverse: run N parallel simulations, compare outcomes')",
        "try_demo": "sm_multiverse",
    },
    {
        "id": "qi_quantum_optimization",
        "book_id": "quantum",
        "level": "expert",
        "title": "Quantum-Inspired Optimization",
        "learn": "QAOA-inspired classical algorithms, quantum tunneling heuristics, and variational optimization.",
        "try_code": None,
        "try_demo": None,
    },
]


def get_books():
    return BOOKS

def get_levels():
    return LEVELS

def get_curriculum():
    return CURRICULUM

def get_curriculum_by_book(book_id: str):
    return [c for c in CURRICULUM if c["book_id"] == book_id]

def get_curriculum_by_level(level: str):
    return [c for c in CURRICULUM if c["level"] == level]

def get_item(item_id: str):
    return next((c for c in CURRICULUM if c["id"] == item_id), None)
