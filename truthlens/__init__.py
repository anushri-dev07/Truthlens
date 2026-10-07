from .api.main import app

def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

__all__ = ["app", "main"]
