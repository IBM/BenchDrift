"""
BenchDrift Interactive App — Entry Point

Analyze any problem, generate meaning-preserving variations with live streaming,
and detect drift across target models — all in one flow.

Usage:
    python app.py              # Launch locally
    python app.py --share      # Public share link
    python app.py --port 7861  # Custom port
"""

import argparse
import os
import sys

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from app.main import build_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BenchDrift Interactive App")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
