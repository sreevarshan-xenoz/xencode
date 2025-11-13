"""Project-level pytest fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_workspace_with_code():
    """Provide a temporary workspace populated with sample Python code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)

        (workspace_path / "main.py").write_text(
            "import os\n"
            "import sys\n\n"
            "def main():\n"
            '    print(\"Hello, World!\")\n'
            "    return 0\n\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n"
        )

        (workspace_path / "utils.py").write_text(
            "import json\n"
            "from typing import Dict, Any\n\n"
            "def load_config() -> Dict[str, Any]:\n"
            '    with open(\"config.json\") as f:\n'
            "        return json.load(f)\n"
        )

        (workspace_path / "tests").mkdir()
        (workspace_path / "tests" / "test_main.py").write_text(
            "import pytest\n"
            "from main import main\n\n"
            "def test_main():\n"
            "    assert main() == 0\n"
        )

        yield workspace_path

