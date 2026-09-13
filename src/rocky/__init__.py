"""Rocky: a five-legged (pentapod) robot -- CAD-derived sim model, gait, and RL task.

Importing this package pulls in nothing heavier than numpy. The mjlab task lives
in `rocky.tasks`, which is what the `mjlab.tasks` entry point loads.
"""

from pathlib import Path

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
ROCKY_XML = MODEL_DIR / "rocky.xml"
ROCKY_STANDALONE_XML = MODEL_DIR / "rocky_standalone.xml"
ROCKY_URDF = MODEL_DIR / "rocky.urdf"

__all__ = ["REPO_ROOT", "MODEL_DIR", "ROCKY_XML", "ROCKY_STANDALONE_XML", "ROCKY_URDF"]
