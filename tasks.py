import shutil
from pathlib import Path

from invoke import task

project_root = Path(__file__).parent


@task
def clean(c):
    cache_directories = [".mypy_cache", "dist"]
    for directory in cache_directories:
        directory_path = Path(directory)
        if directory_path.exists():
            shutil.rmtree(directory_path)


@task
def black(c):
    c.run(f"black --check {project_root}")


@task
def isort(c):
    c.run(f"isort --check {project_root}")


@task
def flake(c):
    c.run(f"flake8 {project_root}")


@task
def mypy(c):
    print(f"mypy {project_root}")
    c.run(f"mypy {project_root}")


@task(black, isort, flake, mypy)
def check(c):
    pass


@task
def black_fix(c):
    c.run(f"black {project_root}")


@task
def isort_fix(c):
    c.run(f"isort {project_root}")


@task(black_fix, isort_fix, black_fix)
def fix(c):
    pass
