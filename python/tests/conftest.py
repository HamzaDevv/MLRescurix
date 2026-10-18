"""Global Pytest Configuration for MLRescurix.

Configures environment and runtimes before any tests are imported.
"""

import os
import platform

# Suppress macOS duplicate OpenMP runtime conflict (SIGSEGV / EXC_BAD_ACCESS in libomp)
# when both LightGBM and PyTorch native C++ runtimes are loaded in the same process:
if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
