import sys

from xencode.cli import cli


def main() -> None:
    """Delegate to the canonical Click CLI agentic command."""
    cli.main(args=["agentic", *sys.argv[1:]], prog_name="xencode")


if __name__ == "__main__":
    main()
