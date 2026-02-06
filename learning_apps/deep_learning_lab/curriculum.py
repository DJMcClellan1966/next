"""
Curriculum: Deep Learning & Statistical ML — Goodfellow/Bengio/Courville, Bishop, ESL (Hastie et al.), Burkov.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "goodfellow", "name": "Deep Learning (Goodfellow et al.)", "short": "Deep Learning", "color": "#2563eb"},
    {"id": "bishop", "name": "Pattern Recognition & ML (Bishop)", "short": "Bishop", "color": "#059669"},
    {"id": "esl", "name": "Elements of Statistical Learning", "short": "ESL", "color": "#7c3aed"},
    {"id": "burkov", "name": "Hundred-Page ML (Burkov)", "short": "Burkov", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    {"id": "dl_regularization", "book_id": "goodfellow", "level": "intermediate", "title": "Regularization (Dropout, L2)",
     "learn": "Reduce overfitting: L2 weight decay, dropout (randomly zero activations), early stopping. Goodfellow Ch 7.",
     "try_code": "from three_books_methods import DeepLearningMethods\nm=DeepLearningMethods()\n# regularization in training",
     "try_demo": None},
    {"id": "dl_optimization", "book_id": "goodfellow", "level": "intermediate", "title": "Optimization (Adam, RMSprop)",
     "learn": "Adaptive learning rates: Adam, RMSprop. Momentum and Nesterov. Goodfellow Ch 8.",
     "try_code": "from three_books_methods import DeepLearningMethods\n# optimizer choice in fit()",
     "try_demo": None},
    {"id": "bishop_gaussian", "book_id": "bishop", "level": "intermediate", "title": "Gaussian Processes (Bishop)",
     "learn": "Non-parametric: prior over functions, kernel covariance. Posterior predictive distribution.",
     "try_code": "from three_books_methods import BishopMethods\nb=BishopMethods()\n# Gaussian process regression",
     "try_demo": None},
    {"id": "bishop_em", "book_id": "bishop", "level": "advanced", "title": "EM Algorithm (Bishop)",
     "learn": "Expectation-Maximization for latent variable models. E-step: q(z); M-step: maximize bound.",
     "try_code": "from ml_toolbox.textbook_concepts.probabilistic_ml import EMAlgorithm",
     "try_demo": None},
    {"id": "esl_svm", "book_id": "esl", "level": "intermediate", "title": "Support Vector Machines (ESL)",
     "learn": "Max-margin classifier. Kernel trick for nonlinearity. C and gamma. ESL Ch 12.",
     "try_code": "from three_books_methods import ESLMethods\ne=ESLMethods()\n# e.support_vector_machine(X,y,kernel='rbf')",
     "try_demo": "esl_svm"},
    {"id": "esl_boosting", "book_id": "esl", "level": "intermediate", "title": "Gradient Boosting (ESL)",
     "learn": "Additive model: fit residuals sequentially. XGBoost, LightGBM use this idea.",
     "try_code": "from three_books_methods import ESLMethods\ne=ESLMethods()\n# gradient boosting classifier/regressor",
     "try_demo": None},
    {"id": "burkov_workflow", "book_id": "burkov", "level": "basics", "title": "ML Project Workflow (Burkov)",
     "learn": "Define problem → get data → train/val/test → iterate. Keep it simple first.",
     "try_code": "# Define metric, baseline, then improve",
     "try_demo": None},
    {"id": "burkov_ensemble", "book_id": "burkov", "level": "basics", "title": "Ensembles (Burkov)",
     "learn": "Bagging (e.g. Random Forest), boosting (e.g. AdaBoost), stacking. Combine weak learners.",
     "try_code": "from ml_toolbox.textbook_concepts.practical_ml import EnsembleMethods",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
