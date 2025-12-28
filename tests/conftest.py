import pytest
from z3 import (
    BoolRef,
    Bool,
)
from models import PuzzleCell, Status


"""
Fixtures
"""
def simple_grid(width, height) -> list[list[BoolRef]]:
    rows = height
    cols = width
    grid = []
    for r in range(rows):
        row_refs = []
        for c in range(cols):
            row_refs.append(Bool(f"{r}_{c}"))
        grid.append(row_refs)
    return grid

@pytest.fixture
def simple_3x3_constraint_grid():
    """Create a simple 3x3 constraint grid with Z3 Bool refs"""
    return simple_grid(3, 3)

@pytest.fixture
def simple_4x5_constraint_grid():
    """Create a simple 4x5 constraint grid with Z3 Bool refs"""
    return simple_grid(4, 5)


@pytest.fixture
def simple_3x3_puzzle_state():
    """
    Create a simple 3x3 puzzle state.
    Layout:
      0 1 2
    0 C I C
    1 I C I
    2 C I C

    Where C=Criminal, I=Innocent
    """
    state = []
    pattern = [
        # (Name, is_criminal, profession)
        [("Sally", True, "builder"), ("Carol", False, "builder"), ("Ned", True, "pilot")],
        [("Alice", False, "clerk"), ("Bob", True, "coder"), ("Charlie", False, "cook")],
        [("Dana", True, "cop"), ("Emma", False, "doctor"), ("Frank", True, "farmer")],
    ]

    for r in range(3):
        row = []
        for c in range(3):
            name, is_criminal, profession = pattern[r][c]
            cell = PuzzleCell(
                status=Status.CRIMINAL if is_criminal else Status.INNOCENT,
                is_criminal=is_criminal,
                profession=profession,
                name=name,
                gender="male",
                orig_hint="",
                paths=[],
                clue=""
            )
            row.append(cell)
        state.append(row)

    return state


@pytest.fixture
def simple_4x5_puzzle_state():
    """
    Create a simple 5x5 puzzle state with varied professions.
    Layout (C=Criminal, I=Innocent):
      0 1 2 3 4
    0 C I C I C
    1 I C I C I
    2 C I C I C
    3 I C I C I
    4 C I C I C

    Professions: alternating between builder, clerk, coder, cook, cop
    """
    state = []
    # (Name, is_criminal, profession)
    pattern = [
        [("Grace", True, "builder"), ("Henry", False, "clerk"), ("Iris", True, "coder"), ("Jack", False, "cook")],
        [("Liam", False, "clerk"), ("Maya", True, "coder"), ("Noah", False, "cook"), ("Olivia", True, "cop")],
        [("Quinn", True, "coder"), ("Rachel", False, "cook"), ("Sam", True, "cop"), ("Tara", True, "builder")],
        [("Victor", False, "cook"), ("Wendy", True, "cop"), ("Xavier", False, "builder"), ("Yara", False, "clerk")],
        [("Amy", True, "cop"), ("Ben", False, "builder"), ("Claire", True, "clerk"), ("David", False, "coder")],
    ]

    for r in range(5):
        row = []
        for c in range(4):
            name, is_criminal, profession = pattern[r][c]
            cell = PuzzleCell(
                status=Status.CRIMINAL if is_criminal else Status.INNOCENT,
                is_criminal=is_criminal,
                profession=profession,
                name=name,
                gender="male" if c % 2 == 0 else "female",
                orig_hint="",
                paths=[],
                clue=""
            )
            row.append(cell)
        state.append(row)

    return state

