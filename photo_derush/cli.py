import typer
from .engine import run_engine

def main():
    typer.run(run_engine)

