# Learning Apps — What Was Done

Remaining books and areas in ML-ToolBox were turned into **separate learning apps**, with related topics **combined where it made sense**. Each app lives in its **own named folder** under `learning_apps/` and exposes a slim Flask UI (By Book / By Level, curriculum, optional demos).

---

## Overview

| App | Port | Books / sources | Run command |
|-----|------|-----------------|-------------|
| **ml_learning_lab** | 5001 | TAOCP (Knuth), Skiena & Bentley, Sedgewick, Bishop/Goodfellow/R&N, Info theory, Algorithm design | (at repo root) `python ml_learning_lab/app.py` |
| **clrs_algorithms_lab** | 5002 | CLRS (Cormen et al.) | `python learning_apps/clrs_algorithms_lab/app.py` |
| **deep_learning_lab** | 5003 | Goodfellow, Bishop, ESL, Burkov | `python learning_apps/deep_learning_lab/app.py` |
| **ai_concepts_lab** | 5004 | Russell & Norvig style (ml_toolbox.ai_concepts) | `python learning_apps/ai_concepts_lab/app.py` |
| **cross_domain_lab** | 5005 | textbook_concepts: quantum, stat mech, linguistics, precognition, self-organization | `python learning_apps/cross_domain_lab/app.py` |
| **python_practice_lab** | 5006 | Reed & Zelle (reed_zelle_patterns.py) | `python learning_apps/python_practice_lab/app.py` |

All run commands assume you are in the **repo root** (`ML-ToolBox`).

---

## What each app contains

- **ml_learning_lab** (existing): Compass (Learn/Decide/Practice/Think), By Book, By Level, Build Knuth Machine; covers Knuth, Skiena, Sedgewick, textbook concepts, information/communication theory, algorithm design.
- **clrs_algorithms_lab**: CLRS — DP (Optimal BST, LIS, Coin Change, Rod Cutting), Greedy (Prim’s MST, Activity Selection), Graph (Bellman-Ford). Curriculum + runnable demos using `clrs_complete_algorithms`.
- **deep_learning_lab**: Goodfellow (regularization, optimization), Bishop (Gaussian processes, EM), ESL (SVM, boosting), Burkov (workflow, ensembles). Curriculum + ESL SVM demo via `three_books_methods`.
- **ai_concepts_lab**: Game theory (Nash, cooperative games), search & planning, reinforcement learning, probabilistic reasoning, clustering; uses `ml_toolbox.ai_concepts`. Curriculum only (demos stub).
- **cross_domain_lab**: “Unusual” cross-domain: quantum mechanics, statistical mechanics (e.g. simulated annealing), linguistics (parsing, grammar features), precognition (forecaster), self-organization (SOM, dissipative structures); uses `ml_toolbox.textbook_concepts`. Curriculum only (demos stub).
- **python_practice_lab**: Reed & Zelle — problem decomposition, algorithm patterns (divide-and-conquer, greedy, recursive, iterative), data structure optimizer, code organizer; uses `reed_zelle_patterns.py`. Curriculum only (demos stub).

---

## Conventions

- **Paths**: From each lab’s `app.py`, repo root is `Path(__file__).resolve().parents[2]`; lab root is `Path(__file__).parent`. Same in each lab’s `demos.py` for REPO.
- **Structure**: Each lab has `curriculum.py` (LEVELS, BOOKS, CURRICULUM + getters), `demos.py` (`run_demo`, DEMO_HANDLERS or stub), `app.py` (Flask, `/api/health`, `/api/curriculum`, `/api/curriculum/book/<id>`, `/api/curriculum/level/<level>`, `/api/try/<demo_id>`, `/`), and its own `README.md`.
- **Ports**: 5001 (ml_learning_lab), 5002–5006 for the five labs under `learning_apps/`.

---

## Adding demos

Labs that currently have a demos stub (ai_concepts_lab, cross_domain_lab, python_practice_lab) can add runnable demos in their `demos.py`: implement `run_demo(demo_id)` and, if desired, wire `try_demo` in curriculum items to those demo IDs.
