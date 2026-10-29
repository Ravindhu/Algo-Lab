"""Called by systemd (OnFailure=) when a unit crashes, so a dead timer still sends you an alert."""
import sys

from algolab.ops import alerts


def main() -> None:
    unit = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    alerts.send(f"systemd unit failed: {unit}. Check: journalctl -u {unit} -n 50",
                "CRITICAL", key=f"unit:{unit}", cooldown_min=30)


if __name__ == "__main__":
    main()
