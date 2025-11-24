import os
os.environ["FORCE_CPU"] = "1"

import constants  # type: ignore
constants.USE_GPU = False
constants.GPU_AVAILABLE = False

import importlib
import part3_complete_pipeline as pipeline  # type: ignore

if __name__ == "__main__":
    pipeline.process_all_sites()
