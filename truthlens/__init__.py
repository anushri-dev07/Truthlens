from __future__ import annotations

from .api.main import app

__all__ = ["app"]

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
