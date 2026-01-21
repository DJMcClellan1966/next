#!/usr/bin/env python3
"""
Start the API server with automatic port selection
"""
import socket
import sys
import os

def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port starting from {start_port}")

def main():
    """Start the API server"""
    import uvicorn
    from api.server import app
    
    # Try to find free port
    try:
        port = find_free_port()
        if port != 8000:
            print(f"⚠️  Port 8000 is busy. Using port {port} instead.")
            print(f"📡 API will be available at: http://localhost:{port}")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print(f"🚀 Starting Quantum AI Platform API server...")
    print(f"📡 API: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🔍 ReDoc: http://localhost:{port}/redoc")
    print(f"\nPress Ctrl+C to stop\n")
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
