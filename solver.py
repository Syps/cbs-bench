import dataclasses
import math
from models import PuzzleState, Status, PuzzleCell
import hint_dsl_functions
from hint_dsl_functions import DSLContraintAdapter, DSLEvalResult
import ast
from z3 import (
    Int,
    Bool,
    solve,
    SortRef,
    BoolRef,
    Solver,
    sat,
)

from queue import Queue


@dataclasses.dataclass
class PuzzleValidationResult:
    valid: bool
    invalid_message: str | None = None

adapter = DSLContraintAdapter()

def eval_cell_hint_dsl(
        cell: PuzzleCell,
        state: PuzzleState,
        constraint_grid: list[list[BoolRef]]
) -> DSLEvalResult:
    orig_hint = cell.orig_hint
    _ast = ast.parse(orig_hint)
    result = eval_expr(next(ast.walk(_ast)), state, constraint_grid)

    return result


"""
Raw Form

  {
    "criminal": false,
    "profession": "farmer",
    "name": "aaron",
    "gender": "male",
    "orig_hint": "more_traits_in_unit_than_unit(unit(neighbor,0),unit(neighbor,16),criminal)",
    "paths": [
      [
        8,
        9,
        16,
        5,
        1
      ]
    ],
    "hint": "#NAME:0 has more criminal neighbors than #NAME:16"
  }
"""





class PuzzleValidator:
    def __init__(self, puzzle_state: PuzzleState):
        self.puzzle_state = puzzle_state
        self.dimens = len(puzzle_state), len(puzzle_state[0])
        self.solver = Solver()
        self.solver.push()
        self.constraint_grid = self._init_constraint_grid()


    def _init_constraint_grid(self) -> list[list[BoolRef]]:
        rows, cols = self.dimens
        grid = []

        for r in range(rows):
            row = []
            grid.append(row)
            for c in range(cols):
                ref = Bool(f"{r}_{c}")
                row.append(ref)
                cell = self.puzzle_state[r][c]
                if cell.status != Status.UNKNOWN:
                    self.solver.add(ref == (cell.status == Status.CRIMINAL))

        return grid

    def _init_cell_queue(self) -> Queue[PuzzleCell]:
        queue = Queue()
        single_reveal_found = False
        for row in self.puzzle_state:
            for cell in row:
                if cell.status != Status.UNKNOWN:
                    if single_reveal_found:
                        raise ValueError("Multiple revealed cells found when initializing cell queue")
                    queue.put(cell)

        return queue

    def validate(self) -> PuzzleValidationResult:
        queue = self._init_cell_queue()
        has_new_clue = False

        while not queue.empty() or has_new_clue:
            has_new_clue = False
            cell = queue.get()
            result = eval_cell_hint_dsl(cell, self.puzzle_state, self.constraint_grid)
            self.solver.add(result.constraint)

            if self.solver.check() != sat:
                message = "\n  - ".join([str(c) for c in self.solver.unsat_core()])
                return PuzzleValidationResult(
                    valid=False,
                    invalid_message=f"Unsolvable. Conflicting constraints:\n{message}"
                )

            model = self.solver.model()
            for ref in model:
                row, col = [int(c) for c in ref.name().split("_")]
                model_value = model[ref]
                puzzle_cell = self.puzzle_state[row][col]

                if puzzle_cell.status == Status.UNKNOWN:
                    has_new_clue = True
                    puzzle_cell.status = Status.INNOCENT if model_value else Status.CRIMINAL
                    queue.put(puzzle_cell)

                elif (puzzle_cell.status == Status.CRIMINAL and not model_value) \
                    or (puzzle_cell.status == Status.INNOCENT and model_value):
                    raise ValueError("Shouldn't happen")

        if len(self.solver.model()) != math.prod(self.dimens):
            return PuzzleValidationResult(
                valid=False,
                invalid_message=f"Not all cells were revealed by end of validation"
            )

        return PuzzleValidationResult(valid=True)


def eval_expr(expr, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
    if isinstance(expr, ast.Constant):
        return expr.value
    elif isinstance(expr, ast.Name):
        return str(expr.id)
    elif isinstance(expr, ast.Expr):
        return eval_expr(expr.value, puzzle_state, constraint_grid)
    elif isinstance(expr, ast.Call):
        clazz = adapter.from_string(expr.func.id)
        args = [eval_expr(arg, puzzle_state, constraint_grid) for arg in expr.args]
        return clazz(*args).eval(puzzle_state, constraint_grid)
    elif isinstance(expr, ast.Module):
        # just return the first thing since our module is always just a single expression
        return eval_expr(expr.body[0], puzzle_state, constraint_grid)
    else:
        raise TypeError(f"Unexpected type {type(expr)}")




