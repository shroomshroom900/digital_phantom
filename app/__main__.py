"""python -m app [--port 8000] [--host 127.0.0.1]"""
import argparse
import os
import sys

# the numerical modules (config, mesh, fem, fibre, muscle) live in the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--reload", action="store_true")
a = p.parse_args()
uvicorn.run("app.server:api", host=a.host, port=a.port, reload=a.reload)
