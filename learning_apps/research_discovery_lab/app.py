"""Research & Knowledge Discovery Lab — Semantic search, knowledge graphs, multi-agent research, forecasting. Run from repo root: python learning_apps/research_discovery_lab/app.py"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(LAB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from curriculum import get_curriculum, get_books, get_levels, get_by_book, get_by_level, get_item
    import curriculum as curriculum_module
except:
    curriculum_module = None

try:
    import demos as demos_module
except:
    demos_module = None

from learning_apps.app_factory import create_lab_app

app = create_lab_app(
    title="Research & Knowledge Discovery Lab",
    description="Semantic search, knowledge graphs, multi-agent research, trend forecasting, and ethical review.",
    port=5014,
    lab_id="research_discovery_lab",
    curriculum_module=curriculum_module,
    demos_module=demos_module
)

if __name__ == "__main__":
    print(f"Research & Knowledge Discovery Lab  http://127.0.0.1:5014/")
    app.run(host="127.0.0.1", port=5014, debug=False)
