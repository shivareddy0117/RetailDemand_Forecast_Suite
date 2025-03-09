#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to run the retail forecasting demo with detailed error information.
"""

import traceback
import sys

try:
    import notebooks.retail_forecasting_demo
except Exception as e:
    print("ERROR: An exception occurred while running the demo script:")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()
    
    print("\nModule information:")
    for name, module in sys.modules.items():
        if name.startswith('pandas') or name.startswith('numpy') or name.startswith('matplotlib'):
            print(f"{name}: {module}") 