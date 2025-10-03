import logging
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, Literal, Optional, Tuple, cast

from tap import Tap

from nightjarpy.utils import NJ_BUILD_DIR, enable_nj_logging
from nightjarpy.utils.utils import NJ_CACHE

logger = logging.getLogger(__name__)


class Main(Tap):
    clear_cache: bool = False
    run: Optional[str] = None
    verbose: bool = False


def run_file(filename: str, verbose: bool = False) -> None:
    """
    Run a Nightjar Python file.

    Args:
        filename: Path to the Python file to execute
        verbose: Whether to enable verbose logging
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

    if not filename.endswith(".py"):
        print(f"Error: File '{filename}' is not a Python file")
        sys.exit(1)

    # Enable logging if verbose
    if verbose:
        enable_nj_logging()

    try:
        # Read and execute the file
        with open(filename, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Compile and execute the file
        compiled_code = compile(source_code, filename, "exec")
        exec(compiled_code, {"__name__": "__main__"})

    except Exception as e:
        print(f"Error executing file '{filename}': {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    args = Main().parse_args()

    if args.clear_cache:
        NJ_CACHE.clear_cache()
        print("Cleared cache!")
    elif args.run:
        run_file(args.run, verbose=args.verbose)
    else:
        print("Usage: python -m nightjarpy [--clear-cache] [--run <filename>] [--verbose]")
        print("  --clear-cache: Clear the Nightjar cache")
        print("  --run <filename>: Run a Nightjar Python file")
        print("  --verbose: Enable verbose logging")
        sys.exit(1)
