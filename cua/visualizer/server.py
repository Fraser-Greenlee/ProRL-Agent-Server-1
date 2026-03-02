#!/usr/bin/env python3
"""Trajectory visualizer server. Serves a web UI for browsing Kimi trajectories.

Usage:
    python visualizer/server.py [--port 8888] [--trajectories trajectories/kimi]

Then SSH port-forward:
    ssh -L 8888:localhost:8888 <remote>

Open http://localhost:8888 in your local browser.
"""

import argparse
import json
import os
import base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

TRAJECTORIES_DIR = None


def scan_trajectories(base_dir: str) -> list[dict]:
    """Walk the trajectories directory and return a list of trajectory summaries."""
    results = []
    base = Path(base_dir)
    for collection_dir in sorted(base.iterdir()):
        if not collection_dir.is_dir():
            continue
        for traj_dir in sorted(collection_dir.iterdir()):
            if not traj_dir.is_dir():
                continue
            traj_json = traj_dir / "trajectory.json"
            if not traj_json.exists():
                continue
            try:
                with open(traj_json) as f:
                    data = json.load(f)
                n_actions = sum(len(s.get("actions", [])) for s in data.get("steps", []))
                results.append({
                    "id": f"{collection_dir.name}/{traj_dir.name}",
                    "collection": collection_dir.name,
                    "trajectory_id": data.get("trajectory_id", traj_dir.name),
                    "goal": data.get("goal", ""),
                    "n_actions": n_actions,
                    "pipeline": data.get("metadata", {}).get("pipeline", "unknown"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def load_trajectory(base_dir: str, traj_path: str) -> dict | None:
    """Load a single trajectory JSON."""
    traj_json = Path(base_dir) / traj_path / "trajectory.json"
    if not traj_json.exists():
        return None
    with open(traj_json) as f:
        return json.load(f)


def load_image_base64(img_path: str) -> str | None:
    """Load an image file and return as base64 data URI."""
    p = Path(img_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    suffix = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
    return f"data:{mime};base64,{data}"


class TrajectoryHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/list":
            self._json_response(scan_trajectories(TRAJECTORIES_DIR))
        elif path == "/api/trajectory":
            traj_path = params.get("id", [None])[0]
            if not traj_path:
                self._json_response({"error": "missing id"}, 400)
                return
            data = load_trajectory(TRAJECTORIES_DIR, traj_path)
            if data is None:
                self._json_response({"error": "not found"}, 404)
                return
            self._json_response(data)
        elif path == "/api/image":
            img_path = params.get("path", [None])[0]
            if not img_path:
                self._json_response({"error": "missing path"}, 400)
                return
            # Allow loading by relative path (from trajectory dir) or absolute
            if not os.path.isabs(img_path):
                img_path = os.path.join(TRAJECTORIES_DIR, img_path)
            b64 = load_image_base64(img_path)
            if b64 is None:
                self._json_response({"error": "image not found"}, 404)
                return
            self._json_response({"data": b64})
        elif path == "/api/image_raw":
            img_path = params.get("path", [None])[0]
            if not img_path:
                self.send_error(400)
                return
            if not os.path.isabs(img_path):
                img_path = os.path.join(TRAJECTORIES_DIR, img_path)
            p = Path(img_path)
            if not p.exists():
                self.send_error(404)
                return
            suffix = p.suffix.lower().lstrip(".")
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            with open(p, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html_path = Path(__file__).parent / "index.html"
        with open(html_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if "/api/image" not in str(args):
            super().log_message(format, *args)


def main():
    global TRAJECTORIES_DIR
    parser = argparse.ArgumentParser(description="Trajectory Visualizer")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--trajectories", default="../trajectories/kimi",
                        help="Path to trajectories directory")
    args = parser.parse_args()

    TRAJECTORIES_DIR = os.path.abspath(args.trajectories)
    if not os.path.isdir(TRAJECTORIES_DIR):
        print(f"Error: {TRAJECTORIES_DIR} is not a directory")
        return

    server = HTTPServer(("0.0.0.0", args.port), TrajectoryHandler)
    print(f"Serving trajectories from: {TRAJECTORIES_DIR}")
    print(f"Visualizer running at http://0.0.0.0:{args.port}")
    print(f"SSH forward: ssh -L {args.port}:localhost:{args.port} <remote>")
    print(f"Then open: http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()

# ssh -L 8888:localhost:8888 jaehunj@oci-nrt-cs-001-login-02.nvidia.com

