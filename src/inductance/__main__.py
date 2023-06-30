"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Inductance."""


if __name__ == "__main__":
    main(prog_name="inductance")  # pragma: no cover
