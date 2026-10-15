import importlib
from pathlib import Path

import pytest

from src.utils.config import SRC_DIR


def module_names(folder: Path) -> list[str]:
    """Dotted names of every Python file under `folder`, packages or not."""
    return [".".join(path.relative_to(folder.parent).with_suffix("").parts)
            for path in sorted(folder.rglob("*.py")) if path.name != "__init__.py"]


@pytest.mark.parametrize("module", module_names(Path(SRC_DIR)))
def test_module_imports(module):
    importlib.import_module(module)
