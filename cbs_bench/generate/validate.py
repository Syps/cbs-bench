import dataclasses
import math
from cbs_bench.models import PuzzleState, Status, PuzzleCell
from . import hint_dsl_functions
from .hint_dsl_functions import DSLContraintAdapter, DSLEvalResult
import ast
from z3 import (
    Int,
    Bool,
    solve,
    SortRef,
FuncDeclRef,
    BoolRef,
    Solver,
    sat,
    Not,
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

def is_indexed_bool_ref(ref_name: str | None) -> bool:
    if not ref_name:
        return False

    try:
        _, _ = [int(val) for val in ref_name.split("_")]
    except ValueError:
        return False

    return True

def is_determined_cell_value(solver, decl, model) -> bool:
    """Check if var's value in model is the only possible value."""
    var = decl()          # Call it to get the Z3 expression
    val = model[decl]         # Get the value
    if val is None or not is_indexed_bool_ref(decl.name()):
        # Only check cell values the model has deduced
        return False

    solver.push()
    solver.add(var != val)
    result = solver.check()
    solver.pop()

    return result != sat

class PuzzleValidator:

    VALIDATION_ERROR_CELL_LOGIC = "Validation error: Cell at {cell_index} is marked {cell_label} but prover logic says it should be {model_label}."
    VALIDATION_ERROR_UNREACHABLE_CELLS = "Not all cells are reachable with the provided hints."

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
                label = f"{r}_{c}"
                ref = Bool(label)
                row.append(ref)
                cell = self.puzzle_state[r][c]
                if cell.status != Status.UNKNOWN:
                    self.solver.assert_and_track(ref == (cell.status == Status.CRIMINAL), cell.label)

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

        determined_refs = []

        while not queue.empty() or has_new_clue:
            has_new_clue = False
            cell = queue.get()
            result = eval_cell_hint_dsl(cell, self.puzzle_state, self.constraint_grid)
            self.solver.assert_and_track(result.constraint, result.hint_text)

            if self.solver.check() != sat:
                message = " AND ".join([str(c) for c in self.solver.unsat_core()])
                return PuzzleValidationResult(
                    valid=False,
                    invalid_message=f"Conflicting constraints: {message}"
                )

            model = self.solver.model()
            determined_refs = [ref for ref in model.decls() if is_determined_cell_value(self.solver, ref, model)]
            for ref in determined_refs:
                row, col = [int(c) for c in ref.name().split("_")]
                model_value = bool(model[ref])
                puzzle_cell = self.puzzle_state[row][col]

                if puzzle_cell.status == Status.UNKNOWN:
                    if puzzle_cell.is_criminal is not model_value:
                            return PuzzleValidationResult(
                                valid=False,
                                invalid_message=self.VALIDATION_ERROR_CELL_LOGIC.format(
                                    cell_index=f'({row},{col})',
                                    cell_label='criminal' if puzzle_cell.is_criminal else 'innocent',
                                    model_label='criminal' if model_value else 'innocent',
                                )
                            )
                    puzzle_cell.status = Status.CRIMINAL if model_value else Status.INNOCENT

                    if puzzle_cell.orig_hint is not None:
                        has_new_clue = True
                        queue.put(puzzle_cell)

                elif (puzzle_cell.status == Status.CRIMINAL and not model_value) \
                    or (puzzle_cell.status == Status.INNOCENT and model_value):
                    raise ValueError("Shouldn't happen")



        check = self.solver.check()
        if len(determined_refs) != math.prod(self.dimens):
            return PuzzleValidationResult(
                valid=False,
                invalid_message=self.VALIDATION_ERROR_UNREACHABLE_CELLS
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




