"""
Entry point for Docker deployment.
This creates the FastAPI app for uvicorn.
"""

from src.api.routes_2d import create_app

# Create app instance
app = create_app("./models/model.keras")