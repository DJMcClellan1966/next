"""Simulation & Modeling Lab – standalone app."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from learning_apps.app_factory import create_lab_app

app = create_lab_app(
    lab_id="simulation_modeling_lab",
    lab_title="Advanced Simulation & Modeling Lab",
    curriculum_module="learning_apps.simulation_modeling_lab.curriculum",
    demos_module="learning_apps.simulation_modeling_lab.demos",
)

if __name__ == "__main__":
    app.run(debug=True, port=5018)
