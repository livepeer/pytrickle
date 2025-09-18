from pathlib import Path
from typing import Optional

import argparse

TEMPLATE_HANDLERS = """
import logging
from pytrickle.decorators import model_loader, video_handler, param_updater, on_stream_stop
from pytrickle.stream_processor import StreamProcessor
from pytrickle.frame_skipper import FrameSkipConfig

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Handlers:
    def __init__(self):
        self.state = {"intensity": 0.8}

    @model_loader
    async def load(self):
        # TODO: Load your model/resources here
        pass

    @video_handler
    async def process_frame(self, frame):
        # TODO: Implement your processing; return None for pass-through
        return None

    @param_updater
    async def update(self, params: dict):
        # Update internal state based on params
        if not params:
            return
        self.state.update(params)

    @on_stream_stop
    async def on_stop(self):
        # Cleanup if needed
        pass


def main():
    handlers = Handlers()
    processor = StreamProcessor.from_handlers(
        handlers,
        name="pytrickle-app",
        port=8000,
        frame_skip_config=FrameSkipConfig(),
    )
    processor.run()


if __name__ == "__main__":
    main()
""".lstrip()

TEMPLATE_RUNNER = """
from {package}.handlers import main

if __name__ == "__main__":
    main()
""".lstrip()

README_SNIPPET = """
# PyTrickle App

This project was bootstrapped with `pytrickle init`.

## Quick start

1. Install dependencies
2. Run the app

```bash
python -m {package}
```

Then POST to /api/stream/start on localhost:8000.
""".lstrip()


def _write_file(path: Path, content: str, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.path).resolve()
    package = args.package or target.name.replace("-", "_")

    # Layout:
    # <target>/
    #   {package}/
    #     __init__.py
    #     handlers.py
    #   README.md
    #   pyproject.toml (optional)

    handlers_py = TEMPLATE_HANDLERS
    runner_py = TEMPLATE_RUNNER.format(package=package)
    readme_md = README_SNIPPET.format(package=package)

    created = []
    created.append(_write_file(target / package / "__init__.py", "", overwrite=args.force))
    created.append(_write_file(target / package / "handlers.py", handlers_py, overwrite=args.force))
    created.append(_write_file(target / package / "__main__.py", runner_py, overwrite=args.force))
    created.append(_write_file(target / "README.md", readme_md, overwrite=args.force))

    print(f"Initialized PyTrickle app in {target}")
    print(f"Package: {package}")
    print("Files:")
    for p in [target / package / "handlers.py", target / package / "__main__.py", target / "README.md"]:
        print(f"  - {p.relative_to(target)}")

    print("\nTry it:")
    print("  python -m", package)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Allow running a local package quickly: pytrickle run --module my_app
    module = args.module
    if not module:
        print("--module is required (e.g., my_app)")
        return 2
    try:
        __import__(module)
        # Delegate to module's __main__
        run_code = f"import runpy; runpy.run_module('{module}', run_name='__main__')"
        exec(run_code, {})
        return 0
    except Exception as e:
        print(f"Failed to run module '{module}': {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pytrickle", description="PyTrickle CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Scaffold a new PyTrickle app")
    p_init.add_argument("path", nargs="?", default=".", help="Target directory")
    p_init.add_argument("--package", help="Package name (default: folder name)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="Run a local app module (python -m)")
    p_run.add_argument("--module", required=True, help="Module/package to run")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
