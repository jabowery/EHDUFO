#!/usr/bin/env python3
"""
Sample script to visualize EHD simulation data saved with FieldOutputManager.
"""

import sys
import os
from field_visualizer import FieldVisualizer

if __name__ == "__main__":
    # If a directory is provided as a command-line argument, use it
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        data_dir = sys.argv[1]
        visualizer = FieldVisualizer(data_dir)
    else:
        # Otherwise, open a file dialog
        visualizer = FieldVisualizer()
    
    # You can adjust visualization parameters before showing
    visualizer.vector_scale = 0.01  # Adjust for your specific data scale
    visualizer.vector_density = 15  # Show vectors every 15 points (higher = sparser)
    visualizer.colormap = 'viridis'  # Alternative: 'plasma', 'jet', 'Blues', etc.
    visualizer.grid_resolution = 120  # Higher = smoother but slower interpolation
    
    # Show the visualization
    visualizer.show()
