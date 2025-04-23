"""
Python module serving as a project/extension template.
"""

import os
import toml

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

# Humarconoid root directory
HUMARCONOID_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
