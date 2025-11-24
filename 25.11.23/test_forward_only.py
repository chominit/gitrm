# -*- coding: utf-8 -*-
"""Test forward pass only"""
import sys
sys.path.insert(0, r'C:\Users\jscool\uav_pipeline_codes\25.11.21\footprint_based_forward')

from pathlib import Path
import numpy as np
import constants as C
from part3_complete_pipeline import load_point_cloud

def test():
    site_name = "Zenmuse_AI_Site_B"
    las_dir = C.PART2_DIR / site_name

    print(f"Loading from: {las_dir}")
    try:
        coords, colors = load_point_cloud(las_dir)
        print(f"Loaded: {len(coords):,} points")

        # Now import and call forward_scanline
        from part3_complete_pipeline import forward_scanline
        print("Calling forward_scanline...")
        votes = forward_scanline(site_name, coords, colors)
        print(f"Total votes: {votes.sum()}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
