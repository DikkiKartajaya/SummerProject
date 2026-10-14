"""Put the project root on sys.path so `import src...` works from any cwd.

Every script starts with `import _bootstrap  # noqa: F401`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
