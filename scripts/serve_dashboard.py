#!/usr/bin/env python3
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main():
    web_root = Path(__file__).resolve().parents[1] / "web"
    handler = lambda *args, **kwargs: NoCacheHandler(*args, directory=str(web_root), **kwargs)
    server = ThreadingHTTPServer(("0.0.0.0", 8765), handler)
    print(f"Serving Statbirt dashboard from {web_root} at http://localhost:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
