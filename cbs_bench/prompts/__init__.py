import os
from enum import Enum


class Prompt(Enum):
    GENERATE_PUZZLE = 1
    SOLVE_PUZZLE = 2

_PROMPT_TO_FILENAME = {
        Prompt.GENERATE_PUZZLE: "generate_puzzle.md",
        Prompt.SOLVE_PUZZLE: "solve_puzzle.md",
}

def load(prompt: Prompt) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = _PROMPT_TO_FILENAME[prompt]

    with open(os.path.join(dir_path, file_name), "r") as f:
        return f.read()

SOLVE_PUZZLE_SYSTEM_PROMPT = load(Prompt.SOLVE_PUZZLE)
GENERATE_PUZZLE_SYSTEM_PROMPT = load(Prompt.GENERATE_PUZZLE)