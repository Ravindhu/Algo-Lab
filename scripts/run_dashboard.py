"""Used by the Docker dashboard service. Same as: algolab dashboard"""
from algolab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["dashboard"]))
