"""Used by the Docker dashboard service. Same as: algolab dashboard (the FastAPI dashboard, served by uvicorn)."""
from algolab.dashboard.api import main

if __name__ == "__main__":
    main()
