"""Start a temporary local server, verify HTTP health, and always stop it."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    # Bypass proxy environment variables for this loopback-only request.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with client.open(
                        f"http://127.0.0.1:{port}/_stcore/health", timeout=1
                    ) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            print(f"PASS: Streamlit health returned 200/ok on port {port}.")
                            return
                except OSError, urllib.error.URLError:
                    time.sleep(0.25)
            log.seek(0)
            raise RuntimeError(f"Streamlit did not become healthy:\n{log.read()}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            print("Verification server stopped.")


if __name__ == "__main__":
    main()
