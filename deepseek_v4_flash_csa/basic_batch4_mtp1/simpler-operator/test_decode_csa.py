"""Compile and run the shipped Simpler C++ without PyPTO code generation.

PyPTO's replay loader and the bundled golden harness bind the generated ABI.
No external pypto-lib checkout is required; PyPTO and its pinned runtime are required.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pypto-lib-operator"))
from run_benchmark import main  # noqa: E402

if __name__ == "__main__":
    main(runtime_dir=HERE)
