"""
Main entry point for API server
"""
from api.server import app
import uvicorn

if __name__ == "__main__":
    import os
    # Allow port to be configured via environment variable
    port = int(os.getenv("PORT", 8001))  # Default to 8001 if 8000 is busy
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
