# Thin shim so `python main.py` keeps working alongside `uv run mirror`. The
# real CLI lives in `src/main.py` because that file is part of the editable
# package surface (`[tool.hatch.build.targets.wheel] packages = ["src"]`), so
# edits there are picked up live by `uv run mirror`. Keeping a force-included
# copy of the CLI at the project root would snapshot a stale version into
# site-packages and shadow source edits — that footgun is what this shim is
# avoiding.
from src.main import main

if __name__ == "__main__":
    main()
