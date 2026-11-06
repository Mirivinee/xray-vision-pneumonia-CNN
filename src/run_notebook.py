"""Execute a project notebook and save its outputs without replacing existing files."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def execute_notebook(
    source: Path,
    output: Path,
    *,
    kernel_name: str = "python3",
    timeout: int | None = None,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Notebook does not exist: {source}")
    if source == output:
        raise ValueError("The executed notebook output must not overwrite its source.")

    failed_output = output.with_name(f"{output.stem}.failed{output.suffix}")
    for path in (output, failed_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite notebook output: {path}")

    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    except CellExecutionError:
        with failed_output.open("xb") as stream:
            nbformat.write(notebook, stream)
        print(f"Execution failed; partial notebook output saved to {failed_output}")
        raise

    with output.open("xb") as stream:
        nbformat.write(notebook, stream)
    print(f"Executed notebook saved to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source .ipynb file.")
    parser.add_argument("--output", type=Path, required=True, help="New path for the executed notebook.")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name (default: python3).")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-cell timeout in seconds; 0 means no timeout (default: 0).",
    )
    arguments = parser.parse_args()
    if arguments.timeout < 0:
        parser.error("--timeout must be zero or a positive number.")
    execute_notebook(
        arguments.input,
        arguments.output,
        kernel_name=arguments.kernel,
        timeout=arguments.timeout or None,
    )


if __name__ == "__main__":
    main()
