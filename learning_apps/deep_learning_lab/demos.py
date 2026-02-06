"""Demos for Deep Learning Lab."""
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def esl_svm():
    try:
        from three_books_methods import ESLMethods
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        m = ESLMethods()
        out = m.support_vector_machine(X, y, kernel="rbf", C=1.0, task_type="classification")
        return {"ok": True, "output": str(out.get("model", "trained"))[:200] + " ..." if len(str(out)) > 200 else str(out)}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


DEMO_HANDLERS = {"esl_svm": esl_svm}


def run_demo(demo_id: str):
    if demo_id not in DEMO_HANDLERS:
        return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
    try:
        return DEMO_HANDLERS[demo_id]()
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}
