from pathlib import Path

from setuptools import setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="evoloza",
    version="0.1.0",
    description="An improvement-loop harness for git repositories with Codex and Ollama backends.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Evoloza",
    python_requires=">=3.9",
    install_requires=["tomli>=2.0.1; python_version < '3.11'"],
    py_modules=["run"],
    entry_points={"console_scripts": ["evoloza=run:main"]},
)
