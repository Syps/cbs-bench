import dataclasses
import math
from models import PuzzleState, Status, PuzzleCell
import hint_dsl_functions
from enum import Enum
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


# DSL Enum Classes




def cell_to_constraint(
        cell: PuzzleCell,
        state: PuzzleState,
        constraint_grid: list[list[BoolRef]]
) -> SortRef:
    """
    - one clue might contain more than 1 constraint
        - Ex: "Both innocents in column D are connected".
    """
    orig_hint = cell.orig_hint
    raw_clue = cell.clue
    _ast = ast.parse(orig_hint)
    constraint = eval_expr(next(ast.walk(_ast)))

    return constraint


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


"""
Steps
-----

1. Model generates puzzle JSON
2. JSON -> PuzzleState (or throw error)
3. Init constraint grid
4. Init Solver
5. For each puzzle cell in puzzle state:
    - orig_hint -> list[constraint]
    - add constraints via solver.add(constraint)
    - check if satisfiable
6. Check if solvable
7. Print result
    
"""


"""
Questions
---------

- How do we know that the generated orig_hint correctly maps to the hint?
- If the hint is what the human/competitor sees, then we need to make sure it's accurate.


Ways to parse a regular hint
- LLM text-to-JSON (take string, give back JSON repr)
- LLM text-to-DSL (take string, give back DSL expr)
    - would need to do an evaluation of this to check accuracy
- regex (check for keywords like "both", "between", "neighbors")
- have LLM generate DSL, use LLM to generate hint from DSL
    - can we use regex to make sure generated text matches?
    - can we generate text ourselves?
        - Yes, I think so...
        - LLM generates DSL expression
        - We parse the expression
        - We validate it (is it a real function name, are the args being passed correct per the func signature, etc)
        - We construct an object that represents it (e.g AllTraitsAreNeighborsInUnit, etc)
        - All expression objects have a plain_english() method which return the "hint" representation of the DSL expr
        - That's the hint we use in the final puzzle
        
        TODO:
        -----
        - unit tests for all hint_text methods
"""

@dataclasses.dataclass
class PuzzleValidationResult:
    valid: bool
    invalid_message: str | None = None



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
            constraint = cell_to_constraint(cell, self.puzzle_state, self.constraint_grid)
            self.solver.add(constraint)

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












        for cell in ordered:
            constraint = cell_to_constraint(cell, self.puzzle_state)
            self.solver.add(constraint)
            if self.solver.check() != sat:
                raise ValueError("Unsatsifiable")

            model = self.solver.model()


    def add_constraints_for_hints(self) -> None:
        for row in self.puzzle_state:
            for cell in row:
                constraint = cell_to_constraint(cell, self.puzzle_state)
                self.solver.add(constraint)





class ConstraintChecker:
    def __init__(self, ref_array: list[list[BoolRef]]):
        self.ref_array = ref_array


    def add(self, orig_hint: str):
        pass

    def update_refs(self, x: list[int]):
        pass



def eval_expr(expr):
    if isinstance(expr, ast.Constant):
        return expr.value
    elif isinstance(expr, ast.Name):
        return str(expr.id)
    elif isinstance(expr, ast.Expr):
        return expr.value
    elif isinstance(expr, ast.Call):
        func_name = expr.func.id
        func = getattr(hint_dsl_functions, func_name)
        args = [eval_expr(arg) for arg in expr.args]
        return func(*args)
    elif isinstance(expr, ast.Module):
        # just return the first thing since our module is always just a single expression
        return [e for e in ast.walk(expr)][0]
    else:
        raise TypeError(f"Unexpected type {type(expr)}")




