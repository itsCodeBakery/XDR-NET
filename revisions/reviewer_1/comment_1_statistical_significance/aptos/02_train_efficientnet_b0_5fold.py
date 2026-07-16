# APTOS 2019: five-fold EfficientNet-B0 baseline training
# Saves checkpoints, fold predictions, out-of-fold probabilities, metrics,
# confusion matrix, ROC curves, and training-history figures.

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
