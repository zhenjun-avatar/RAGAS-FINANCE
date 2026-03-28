#!/usr/bin/env python
"""启动 FastAPI（RAG 文档 API）。"""

import argparse
import os
import sys

_agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "agent"))
sys.path.insert(0, _agent_dir)

from api.server import run_server  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
