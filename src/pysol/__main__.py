"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """PySol."""


if __name__ == "__main__":
    main(prog_name="pysol")  # pragma: no cover
