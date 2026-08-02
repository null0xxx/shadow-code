from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PROJECT_PYTHON_FLOOR = Version("3.10")
ROOT = Path(__file__).parents[2]


def test_locked_distributions_support_declared_python_floor() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in pyproject

    requirements = [
        Requirement(line)
        for line in (ROOT / "requirements-dev.lock").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    for requirement in requirements:
        installed = distribution(requirement.name)
        assert Version(installed.version) in requirement.specifier
        requires_python = installed.metadata.get("Requires-Python")
        if requires_python:
            assert PROJECT_PYTHON_FLOOR in SpecifierSet(requires_python), (
                f"{requirement} requires Python {requires_python}"
            )
