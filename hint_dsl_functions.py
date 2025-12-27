
import dataclasses
import math
import pdb

from models import PuzzleState, Status, PuzzleCell
from enum import Enum
from abc import ABC, abstractmethod
import ast
from z3 import (
    Int,
    Bool,
    Not,
    SortRef,
    Sum,
    And,
    BoolVal,
    Implies,
    BoolRef,
    Solver,
    sat,
)

from queue import Queue


# DSL Enum Classes

def number_to_string(value: int) -> str:
    strings = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
     "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
     "seventeen", "eighteen", "nineteen"]

    return strings[value-1]


class Trait(Enum):
    """Represents whether a cell is criminal or innocent"""
    CRIMINAL = "criminal"
    INNOCENT = "innocent"

    def hint_text(self):
        return str(self.value) + "s"


class UnitType(Enum):
    """Represents the type of unit/group being referenced"""
    BETWEEN = "between"  # unit(between, pair(a,b))
    COL = "col"  # unit(col, n)
    CORNER = "corner"  # unit(corner, void)
    EDGE = "edge"  # unit(edge, void)
    NEIGHBOR = "neighbor"  # unit(neighbor, n)
    PROFESSION = "profession"  # unit(profession, prof)
    ROW = "row"  # unit(row, n)

    def hint_text(self):
        return str(self.value)


class Profession(Enum):
    """Represents the profession of a cell"""
    BUILDER = "builder"
    CLERK = "clerk"
    CODER = "coder"
    COOK = "cook"
    COP = "cop"
    DOCTOR = "doctor"
    FARMER = "farmer"
    GUARD = "guard"
    JUDGE = "judge"
    MECH = "mech"
    PAINTER = "painter"
    PILOT = "pilot"
    SINGER = "singer"
    SLEUTH = "sleuth"
    TEACHER = "teacher"


class Pair:
    """Represents a pair of cells"""
    def __init__(self, a_index: int, b_index: int):
        self.a_index = a_index
        self.b_index = b_index


def idx_to_coord(idx: int, dimens: tuple[int, int]) -> tuple[int, int]:
    rows, cols = dimens
    return idx // cols, idx % cols

# despite the name, between seems to be inclusive of the given indexes, depending on vert or horiz
def between(pair: Pair, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
    dimens = len(constraint_grid), len(constraint_grid[0])
    r1, c1 = idx_to_coord(pair.a_index, dimens)
    r2, c2 = idx_to_coord(pair.b_index, dimens)

    if r1 != r2 and c1 != c2:
        raise ValueError(f"Pair indices must share same column or row. Got ({r1},{c1}) and ({r2},{c2}).")

    r_asc = sorted([r1, r2])
    c_asc = sorted([c1, c2])

    # if r1 == r2:
    c_asc[1] += 1
    # else:
    r_asc[1] += 1

    cells = []

    for r in range(*r_asc):
        for c in range(*c_asc):
            cells.append(constraint_grid[r][c])

    return cells

def col(index: int, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
    return [r[index] for r in constraint_grid]

def row(index: int, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
    return constraint_grid[index]

def corners(puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
    return [
        constraint_grid[0][0],
        constraint_grid[0][-1],
        constraint_grid[-1][0],
        constraint_grid[-1][-1],
    ]

def edges(puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
    return [
        # top row
        *constraint_grid[0],
        # bottom row
        *constraint_grid[-1],
        # left edge
        *[r[0] for r in constraint_grid[1:-1]],
        # right edge
        *[r[-1] for r in constraint_grid[1:-1]],
    ]

def profession_bools(profession: Profession, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
    bools = []
    for i_r, _row in enumerate(puzzle_state):
        for i_c, cell in enumerate(_row):
            if cell.profession == profession.value:
                bools.append(constraint_grid[i_r][i_c])
    return bools


def neighbor_indexes(index: int, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:

    rows, cols = len(constraint_grid), len(constraint_grid[0])
    _row, _col = idx_to_coord(index, (rows, cols))
    bools = []
    for r in [_row-1, _row, _row+1]:
        for c in [_col-1, _col, _col+1]:
            if r == _row and c == _col:
                continue

            if 0 <= r < rows and 0 <= c < cols:
                bools.append(constraint_grid[r][c])

    return bools



@dataclasses.dataclass(frozen=True)
class DSLEvalResult:
    constraint: SortRef
    hint_text: str


class DSLFuncEvaluator(ABC):
    """Abstract base class for DSL function evaluators"""

    @abstractmethod
    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        """Evaluate the DSL function and return a constraint and hint text"""
        pass

    @abstractmethod
    def hint_text(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> str:
        """Return the human-readable hint text for this constraint"""
        pass



class Pair:

    def __init__(self, a_index: int, b_index: int):
        self.a_index = a_index
        self.b_index = b_index

class Unit:
    # Example: unit_shares_n_out_of_n_traits_with_unit(unit(edge,void),unit(neighbor,2),criminal,1,10)
    # Example: is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
    # Example: both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
    def __init__(self, unit_type, selector):
        self.unit_type = unit_type
        self.selector = selector

    def count_trait(self, trait: Trait, puzzle_state: PuzzleState) -> int:
        cells = self.cells(puzzle_state)
        trait_is_criminal = trait == Trait.CRIMINAL

        return sum(
            1 for cell in cells if
            (trait_is_criminal and cell.is_criminal) or (not trait_is_criminal and not cell.is_criminal)
        )

    def cells(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> list[BoolRef]:
        if self.unit_type == UnitType.BETWEEN.value:
            return between(self.selector, puzzle_state, constraint_grid)

        if self.unit_type == UnitType.ROW.value:
            return row(self.selector, puzzle_state, constraint_grid)

        if self.unit_type == UnitType.COL.value:
            return col(self.selector, puzzle_state, constraint_grid)

        if self.unit_type == UnitType.CORNER.value:
            return corners(puzzle_state, constraint_grid)

        if self.unit_type == UnitType.EDGE.value:
            return edges(puzzle_state, constraint_grid)

        if self.unit_type == UnitType.PROFESSION.value:
            return profession_bools(self.selector, puzzle_state, constraint_grid)

        if self.unit_type == UnitType.NEIGHBOR.value:
            return neighbor_indexes(self.selector, puzzle_state, constraint_grid)

        raise ValueError(f"Unknown unit type: {self.unit_type}")

    def hint_text(self):
        if self.unit_type == UnitType.BETWEEN:
            return f"between {self.selector.hint_text()}"

        if self.unit_type == UnitType.COL:
            return f"in column {self.selector.hint_text()}"

        if self.unit_type == UnitType.CORNER:
            return "in the corners"

        if self.unit_type == UnitType.EDGE:
            return "on the edges"

        if self.unit_type == UnitType.PROFESSION:
            return self.selector.hint_text()

        if self.unit_type == UnitType.NEIGHBOR:
            return "neighbors"

        if self.unit_type == UnitType.ROW:
            return "in row "


# Special constant for void value used with edge/corner units
VOID = "void"


def all_units(unit_type: UnitType, constraint_grid: list[list[BoolRef]]) -> list[Unit]:
    if unit_type == UnitType.ROW.value:
        return [Unit(unit_type, i) for i in range(len(constraint_grid))]

    if unit_type == UnitType.COL.value:
        return [Unit(unit_type, i) for i in range(len(constraint_grid[0]))]

    if unit_type == UnitType.NEIGHBOR.value:
        units = []
        n_rows = len(constraint_grid)
        n_cols = len(constraint_grid[0])
        for i in range(n_rows):
            for j in range(n_cols):
                units.append(Unit(unit_type, i*n_cols + j))
        return units


    raise ValueError(f"Unexpected unit type passed to all_units: {unit_type}")


# DSL Constraint Classes

class AllTraitsAreNeighborsInUnit(DSLFuncEvaluator):
    # Example: all_traits_are_neighbors_in_unit(unit(between,pair(0,16)),criminal)
    # Hint: "All criminals #BETWEEN:pair(0,16) are connected"
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> str:
        label = "criminals" if self.trait == Trait.CRIMINAL else "innocents"

        if not hasattr(self.unit.selector, 'a_index') or \
            not hasattr(self.unit.selector, 'b_index'):
            raise ValueError(f"Unexpected unit type: {self.unit}")

        a_index = self.unit.selector.a_index
        b_index = self.unit.selector.b_index

        dimens = len(constraint_grid), len(constraint_grid[0])
        r1, c1 = idx_to_coord(a_index, dimens)
        r2, c2 = idx_to_coord(b_index, dimens)

        if r1 == r2:
            if sorted([c1, c2])[0] == 0:
                position = "to the left of"
                name = puzzle_state[r1][-1].name
            else:
                position = "to the right of"
                name = puzzle_state[r1][0].name
        else:
            if sorted([r1, r2])[0] == 0:
                position = "above"
                name = puzzle_state[-1][c1].name
            else:
                position = "below"
                name = puzzle_state[0][c1].name

        return f"All {label} {position} {name} are connected"

    def connected(self, bools: list[BoolRef], is_criminal: bool):
        if len(bools) <= 2:
            return BoolVal(True)

        # Normalize: if checking for False connectivity, invert the perspective
        check = [(b if is_criminal else Not(b)) for b in bools]

        constraints = []
        for i in range(len(check) - 2):
            constraints.append(Implies(And(check[i], check[i + 2]), check[i + 1]))

        return And(constraints)

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        is_criminal = self.trait == Trait.CRIMINAL
        constraint = self.connected(self.unit.cells(puzzle_state, constraint_grid), is_criminal)

        return DSLEvalResult(
            constraint=constraint,
            hint_text=self.hint_text(puzzle_state, constraint_grid),
        )


class AllUnitsHaveAtLeastNTraits(DSLFuncEvaluator):
    # Example: all_units_have_at_least_n_traits(row,innocent,2)
    # Hint: Each row has at least 2 innocents
    def __init__(self, unit_type: UnitType, trait: Trait, n: int):
        self.unit_type = unit_type
        self.trait = Trait(trait)
        self.n = n

    def hint_text(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> str:
        if self.unit_type == "neighbor":
            return f"Everyone has at least {self.n} {self.trait.hint_text()} neighbors"
        else:
            if self.unit_type == UnitType.ROW.value:
                unit_type_hint = None
            return f"All {self.unit_type}s have at least {number_to_string(self.n)} {self.trait.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        units = all_units(self.unit_type, constraint_grid)
        and_constraints = []
        for unit in units:
            cells = unit.cells(puzzle_state, constraint_grid)
            if self.trait == Trait.CRIMINAL:
                and_constraints.append(
                    Sum([c for c in cells]) == self.n
                )
            else:
                and_constraints.append(
                    Sum([c for c in cells]) == len(cells) - self.n
                )

        constraint = And(and_constraints)

        return DSLEvalResult(
            constraint=constraint,
            hint_text=self.hint_text(puzzle_state, constraint_grid),
        )


class BothTraitsAreNeighborsInUnit(DSLFuncEvaluator):
    # Example: both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
    # Hint: "Both innocents #BETWEEN:pair(3,15) are connected"
    # The unit will ALWAYS be a between pair and it's implied they are connected
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        # needs to be fixed to be "to the right of" or "above", etc.
        # See https://cluesbysam.com/s/user/671c185070c51ea6/pack-1/47/
        # (hash: b4a2be0c47822689baa6d1ff5749c4e8)
        return f"Both {self.trait.hint_text()} {self.unit.hint_text()} are connected"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        cells = self.unit.cells(puzzle_state, constraint_grid)
        is_criminal = self.trait == Trait.CRIMINAL
        or_constraints = [

        ]
        # iterate over cells with a window of 2
        for c1, c2 in zip(cells, cells[1:]):
            or_constraints.append(And(
                c1 == is_criminal,
                c2 == is_criminal
            ))

        return And(Sum([c == is_criminal for c in cells]) == 2, or_constraints)



class BothTraitsInUnitAreInUnit(DSLFuncEvaluator):
    # Example: both_traits_in_unit_are_in_unit(unit(col,3),unit(neighbor,11),innocent)
    # Hint: Both innocents in column #C:3 are #NAMES:11 neighbors
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"Both {self.trait.hint_text()} {self.unit_1.hint_text()} are {self.unit_2.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class EqualNumberOfTraitsInUnits(DSLFuncEvaluator):
    # Example: equal_number_of_traits_in_units(unit(neighbor,0),unit(neighbor,7),innocent)
    # Hint: #NAME:0 and #NAME:7 have an equal number of innocent neighbors
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"{self.unit_1.hint_text()} and {self.unit_2.hint_text()} have an equal number of {self.trait.value} {self.unit_1.unit_type.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class EqualTraitsAndTraitsInUnit(DSLFuncEvaluator):
    # Example: equal_traits_and_traits_in_unit(unit(between,pair(15,19)),criminal,innocent)
    # Hint: "There are as many criminals as innocents #BETWEEN:pair(15,19)"
    def __init__(self, unit: Unit, trait_1: Trait, trait_2: Trait):
        self.unit = unit
        self.trait_1 = trait_1
        self.trait_2 = trait_2

    def hint_text(self) -> str:
        return f"There are as many {self.trait_1.hint_text()} as {self.trait_2.hint_text()} {self.unit.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class EveryProfessionHasATraitInDir(DSLFuncEvaluator):
    # Example: every_profession_has_a_trait_in_dir(guard,criminal,-1,0)
    # Hint: Every #PROF:guard has a criminal directly to the left of them
    def __init__(self, profession: Profession, trait: Trait, dx: int, dy: int):
        self.profession = profession
        self.trait = trait
        self.dx = dx
        self.dy = dy

    def hint_text(self) -> str:
        direction = self._direction_text(self.dx, self.dy)
        return f"Every {self.profession.value} has a {self.trait.value} directly {direction} of them"

    def _direction_text(self, dx: int, dy: int) -> str:
        if dx == -1 and dy == 0:
            return "to the left"
        elif dx == 1 and dy == 0:
            return "to the right"
        elif dx == 0 and dy == -1:
            return "above"
        elif dx == 0 and dy == 1:
            return "below"
        else:
            return f"at ({dx},{dy})"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class HasMostTraits(DSLFuncEvaluator):
    # Example: has_most_traits(unit(col,4),innocent)
    # Hint: Column #C:4 has more innocents than any other column
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        unit_type_name = self.unit.unit_type.hint_text()
        return f"{self.unit.hint_text().capitalize()} has more {self.trait.hint_text()} than any other {unit_type_name}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class HasTrait(DSLFuncEvaluator):
    # Example: has_trait(19,innocent)
    # Hint: Each #PROF:cook neighboring me is innocent
    def __init__(self, cell_id: int, trait: Trait):
        self.cell_id = cell_id
        self.trait = trait

    def hint_text(self) -> str:
        # Note: This hint seems context-dependent based on the example
        return f"Cell {self.cell_id} is {self.trait.value}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class IsNotOnlyTraitInUnit(DSLFuncEvaluator):
    # Example: is_not_only_trait_in_unit(unit(between,pair(5,7)),5,criminal)
    # Hint: "#NAME:5 is one of two or more criminals #BETWEEN:pair(5,7)"
    def __init__(self, unit: Unit, cell_id: int, trait: Trait):
        self.unit = unit
        self.cell_id = cell_id
        self.trait = trait

    def hint_text(self) -> str:
        return f"Cell {self.cell_id} is one of two or more {self.trait.hint_text()} {self.unit.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class IsOneOfNTraitsInUnit(DSLFuncEvaluator):
    # Example: is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
    # Hint: #NAME:5 is one of #NAMES:6 5 criminal neighbors
    def __init__(self, unit: Unit, cell_id: int, trait: Trait, n: int):
        self.unit = unit
        self.cell_id = cell_id
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        return f"Cell {self.cell_id} is one of {self.n} {self.trait.value} {self.unit.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class MinNumberOfTraitsInUnit(DSLFuncEvaluator):
    # Example: min_number_of_traits_in_unit(unit(edge,void),criminal,7)
    # Hint: There are at least 7 criminals on the edges
    def __init__(self, unit: Unit, trait: Trait, min_count: int):
        self.unit = unit
        self.trait = trait
        self.min_count = min_count

    def hint_text(self) -> str:
        return f"There are at least {self.min_count} {self.trait.hint_text()} {self.unit.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class MoreTraitsInUnitThanUnit(DSLFuncEvaluator):
    # Example: more_traits_in_unit_than_unit(unit(neighbor,2),unit(neighbor,19),innocent)
    # Hint: #NAME:2 has more innocent neighbors than #NAME:19
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"{self.unit_1.hint_text()} has more {self.trait.value} {self.unit_1.unit_type.hint_text()} than {self.unit_2.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class MoreTraitsThanTraitsInUnit(DSLFuncEvaluator):
    # Example: more_traits_than_traits_in_unit(unit(neighbor,17),criminal,innocent)
    # Hint: #NAME:17 has more criminal than innocent neighbors
    def __init__(self, unit: Unit, trait_1: Trait, trait_2: Trait):
        self.unit = unit
        self.trait_1 = trait_1
        self.trait_2 = trait_2

    def hint_text(self) -> str:
        return f"{self.unit.hint_text()} has more {self.trait_1.value} than {self.trait_2.value} {self.unit.unit_type.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class NProfessionsHaveTraitInDir(DSLFuncEvaluator):
    # Example: n_professions_have_trait_in_dir(singer,criminal,0,-1,0)
    # Hint: No #PROF:singer has a criminal directly above them
    def __init__(self, profession: Profession, trait: Trait, dx: int, dy: int, n: int):
        self.profession = profession
        self.trait = trait
        self.dx = dx
        self.dy = dy
        self.n = n

    def hint_text(self) -> str:
        direction = self._direction_text(self.dx, self.dy)
        if self.n == 0:
            return f"No {self.profession.value} has a {self.trait.value} directly {direction} them"
        elif self.n == 1:
            return f"1 {self.profession.value} has a {self.trait.value} directly {direction} them"
        else:
            return f"{self.n} {self.profession.value}s have a {self.trait.value} directly {direction} them"

    def _direction_text(self, dx: int, dy: int) -> str:
        if dx == -1 and dy == 0:
            return "to the left of"
        elif dx == 1 and dy == 0:
            return "to the right of"
        elif dx == 0 and dy == -1:
            return "above"
        elif dx == 0 and dy == 1:
            return "below"
        else:
            return f"at ({dx},{dy}) from"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class NumberOfTraits(DSLFuncEvaluator):
    # Example: number_of_traits(criminal,16)
    # Hint: There are 16 criminals in total
    def __init__(self, trait: Trait, count: int):
        self.trait = trait
        self.count = count

    def hint_text(self) -> str:
        return f"There are {self.count} {self.trait.hint_text()} in total"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class NumberOfTraitsInUnit(DSLFuncEvaluator):
    # Example: number_of_traits_in_unit(unit(between,pair(1,5)),innocent,1)
    # Hint: "There is only one innocent #BETWEEN:pair(1,5)"
    def __init__(self, unit: Unit, trait: Trait, count: int):
        self.unit = unit
        self.trait = trait
        self.count = count

    def hint_text(self) -> str:
        if self.count == 1:
            return f"There is only one {self.trait.value} {self.unit.hint_text()}"
        else:
            return f"There are {self.count} {self.trait.hint_text()} {self.unit.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class OddNumberOfTraitsInUnit(DSLFuncEvaluator):
    # Example: odd_number_of_traits_in_unit(unit(profession,guard),innocent)
    # Hint: There's an odd number of innocent #PROFS:guard
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        return f"There's an odd number of {self.trait.value} {self.unit.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class OnlyOneUnitHasExactlyNTraits(DSLFuncEvaluator):
    # Example: only_one_unit_has_exactly_n_traits(col,innocent,1)
    # Hint: Only one column has exactly 1 innocent
    def __init__(self, unit_type: UnitType, trait: Trait, n: int):
        self.unit_type = unit_type
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        return f"Only one {self.unit_type.hint_text()} has exactly {self.n} {self.trait.value if self.n == 1 else self.trait.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class OnlyTraitInUnitIsInUnit(DSLFuncEvaluator):
    # Example: only_trait_in_unit_is_in_unit(unit(between,pair(0,2)),unit(between,pair(1,3)),innocent)
    # Hint: "The only innocent #BETWEEN:pair(0,2) is #BETWEEN:pair(1,3)"
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"The only {self.trait.value} {self.unit_1.hint_text()} is {self.unit_2.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class OnlyUnitHasExactlyNTraits(DSLFuncEvaluator):
    # Example: only_unit_has_exactly_n_traits(unit(neighbor,19),innocent,1)
    # Hint: #NAME:19 is the only one with only 1 innocent neighbor
    def __init__(self, unit: Unit, trait: Trait, n: int):
        self.unit = unit
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        return f"{self.unit.hint_text()} is the only one with only {self.n} {self.trait.value} {self.unit.unit_type.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class TotalNumberOfTraitsInUnits(DSLFuncEvaluator):
    # Example: total_number_of_traits_in_units(unit(neighbor,11),unit(neighbor,16),innocent,4)
    # Hint: #NAME:11 and #NAME:16 have 4 innocent neighbors in total
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait, total: int):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait
        self.total = total

    def hint_text(self) -> str:
        return f"{self.unit_1.hint_text()} and {self.unit_2.hint_text()} have {self.total} {self.trait.value} {self.unit_1.unit_type.hint_text()} in total"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class UnitSharesNOutOfNTraitsWithUnit(DSLFuncEvaluator):
    # Example: unit_shares_n_out_of_n_traits_with_unit(unit(edge,void),unit(neighbor,2),criminal,1,10)
    # Hint: Only 1 of the 10 criminals on the edges is #NAMES:2 neighbor
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait, n: int, total: int):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait
        self.n = n
        self.total = total

    def hint_text(self) -> str:
        return f"Only {self.n} of the {self.total} {self.trait.hint_text()} {self.unit_1.hint_text()} is {self.unit_2.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class UnitsShareNTraits(DSLFuncEvaluator):
    # Example: units_share_n_traits(unit(between,pair(2,14)),unit(neighbor,13),innocent,0)
    # Hint: "There are no innocents #BETWEEN:pair(2,14) who neighbor #NAME:13"
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait, n: int):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        if self.n == 0:
            return f"There are no {self.trait.hint_text()} {self.unit_1.hint_text()} who are {self.unit_2.hint_text()}"
        else:
            return f"There are {self.n} {self.trait.hint_text()} {self.unit_1.hint_text()} who are {self.unit_2.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass


class UnitsShareOddNTraits(DSLFuncEvaluator):
    # Example: units_share_odd_n_traits(unit(edge,void),unit(neighbor,7),innocent)
    # Hint: An odd number of innocents on the edges neighbor #NAME:7
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"An odd number of {self.trait.hint_text()} {self.unit_1.hint_text()} are {self.unit_2.hint_text()}"

    def eval(self, puzzle_state: PuzzleState, constraint_grid: list[list[BoolRef]]) -> DSLEvalResult:
        pass



class DSLContraintAdapter:
    def __init__(self):
        self.dsl_to_class_map = {
            # Example: all_traits_are_neighbors_in_unit(unit(between,pair(0,16)),criminal)
            "all_traits_are_neighbors_in_unit": AllTraitsAreNeighborsInUnit,
            # Example: all_units_have_at_least_n_traits(row,innocent,2)
            "all_units_have_at_least_n_traits": AllUnitsHaveAtLeastNTraits,
            # Example: both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
            "both_traits_are_neighbors_in_unit": BothTraitsAreNeighborsInUnit,
            # Example: both_traits_in_unit_are_in_unit(unit(col,3),unit(neighbor,11),innocent)
            "both_traits_in_unit_are_in_unit": BothTraitsInUnitAreInUnit,
            # Example: equal_number_of_traits_in_units(unit(neighbor,0),unit(neighbor,7),innocent)
            "equal_number_of_traits_in_units": EqualNumberOfTraitsInUnits,
            # Example: equal_traits_and_traits_in_unit(unit(between,pair(15,19)),criminal,innocent)
            "equal_traits_and_traits_in_unit": EqualTraitsAndTraitsInUnit,
            # Example: every_profession_has_a_trait_in_dir(guard,criminal,-1,0)
            "every_profession_has_a_trait_in_dir": EveryProfessionHasATraitInDir,
            # Example: has_most_traits(unit(col,4),innocent)
            "has_most_traits": HasMostTraits,
            # Example: has_trait(19,innocent)
            "has_trait": HasTrait,
            # Example: is_not_only_trait_in_unit(unit(between,pair(5,7)),5,criminal)
            "is_not_only_trait_in_unit": IsNotOnlyTraitInUnit,
            # Example: is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
            "is_one_of_n_traits_in_unit": IsOneOfNTraitsInUnit,
            # Example: min_number_of_traits_in_unit(unit(edge,void),criminal,7)
            "min_number_of_traits_in_unit": MinNumberOfTraitsInUnit,
            # Example: more_traits_in_unit_than_unit(unit(neighbor,2),unit(neighbor,19),innocent)
            "more_traits_in_unit_than_unit": MoreTraitsInUnitThanUnit,
            # Example: more_traits_than_traits_in_unit(unit(neighbor,17),criminal,innocent)
            "more_traits_than_traits_in_unit": MoreTraitsThanTraitsInUnit,
            # Example: n_professions_have_trait_in_dir(singer,criminal,0,-1,0)
            "n_professions_have_trait_in_dir": NProfessionsHaveTraitInDir,
            # Example: number_of_traits(criminal,16)
            "number_of_traits": NumberOfTraits,
            # Example: number_of_traits_in_unit(unit(between,pair(1,5)),innocent,1)
            "number_of_traits_in_unit": NumberOfTraitsInUnit,
            # Example: odd_number_of_traits_in_unit(unit(profession,guard),innocent)
            "odd_number_of_traits_in_unit": OddNumberOfTraitsInUnit,
            # Example: only_one_unit_has_exactly_n_traits(col,innocent,1)
            "only_one_unit_has_exactly_n_traits": OnlyOneUnitHasExactlyNTraits,
            # Example: only_trait_in_unit_is_in_unit(unit(between,pair(0,2)),unit(between,pair(1,3)),innocent)
            "only_trait_in_unit_is_in_unit": OnlyTraitInUnitIsInUnit,
            # Example: only_unit_has_exactly_n_traits(unit(neighbor,19),innocent,1)
            "only_unit_has_exactly_n_traits": OnlyUnitHasExactlyNTraits,
            # Example: total_number_of_traits_in_units(unit(neighbor,11),unit(neighbor,16),innocent,4)
            "total_number_of_traits_in_units": TotalNumberOfTraitsInUnits,
            # Example: unit_shares_n_out_of_n_traits_with_unit(unit(edge,void),unit(neighbor,2),criminal,1,10)
            "unit_shares_n_out_of_n_traits_with_unit": UnitSharesNOutOfNTraitsWithUnit,
            # Example: units_share_n_traits(unit(between,pair(2,14)),unit(neighbor,13),innocent,0)
            "units_share_n_traits": UnitsShareNTraits,
            # Example: units_share_odd_n_traits(unit(edge,void),unit(neighbor,7),innocent)
            "units_share_odd_n_traits": UnitsShareOddNTraits,
            "unit": Unit,
            "pair": Pair,
        }

    def from_string(self, value: str):
        if value not in self.dsl_to_class_map:
            raise KeyError(f"String value passed, `{value}` was not found in DSL map")

        return self.dsl_to_class_map[value]