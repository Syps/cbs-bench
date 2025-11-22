from models import PuzzleState, PuzzleCell

from z3 import (
    Int,
    Bool,
    solve,
    SortRef,
    Solver,
    sat,
)


def is_solvable(puzzle_state: PuzzleState) -> bool:
    solver = Solver()
    for row in puzzle_state:
        for cell in row:
            constraint = cell_to_constraint(cell, puzzle_state)
            solver.add(constraint)
    soln = solver.check()
    return soln == sat


def cell_to_constraint(cell: PuzzleCell, state: PuzzleState) -> SortRef:
    """
    - one clue might contain more than 1 constraint
        - Ex: "Both innocents in column D are connected".
    """
    orig_hint = cell.orig_hint
    raw_clue = cell.clue




