"""Export and verify the ONNX start rule. Works from any folder, without `pip install`:

    python scripts/export_onnx.py --out qaoa_start_rule.onnx
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qaoa_start_rule.onnx_export import main

if __name__ == "__main__":
    main()
