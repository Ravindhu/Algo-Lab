"""Used by deploy/systemd/algolab-optimize.service. Same as: algolab optimize"""
from algolab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["optimize"]))
