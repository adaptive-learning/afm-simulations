import shutil
from pathlib import Path

from invoke import task

project_root = Path(__file__).parent


@task
def clean(c):
    cache_directories = [project_root / ".mypy_cache", project_root / "dist"]
    for directory in cache_directories:
        if directory.exists():
            shutil.rmtree(directory)


@task
def clean_results(c):
    analysis_root = project_root / "src/afm_simulations/analysis"
    for root in (project_root, analysis_root):
        results_directories = [
            root / "cache",
            root / "data",
            root / "fig",
        ]
        for directory in results_directories:
            if directory.exists():
                shutil.rmtree(directory)


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
