
import dataclasses
import math
from models import PuzzleState, Status, PuzzleCell
from enum import Enum
import ast
from z3 import (
    Int,
    Bool,
    solve,
    SortRef,
    And,
    Or,
    BoolRef,
    Solver,
    sat,
)

from queue import Queue


# DSL Enum Classes

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




def between(pair: Pair) -> list[int]:
    pass

def col(index: int) -> list[int]:
    pass

def row(index: int) -> list[int]:
    pass

def corners() -> list[int]:
    pass

def edges() -> list[int]:
    pass

def profession_indexes(profession: Profession) -> list[int]:
    pass

def neighbor_indexes(index: int) -> list[int]:
    pass


class Unit:
    # Example: unit_shares_n_out_of_n_traits_with_unit(unit(edge,void),unit(neighbor,2),criminal,1,10)
    # Example: is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
    # Example: both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
    def __init__(self, unit_type, selector):
        self.unit_type = unit_type
        self.selector = selector

    def cells(self) -> list[int]:
        if self.unit_type == UnitType.BETWEEN:
            return between(self.selector)

        if self.unit_type == UnitType.COL:
            return col(self.selector)

        if self.unit_type == UnitType.CORNER:
            return corners()

        if self.unit_type == UnitType.EDGE:
            return edges()

        if self.unit_type == UnitType.PROFESSION:
            return profession_indexes(self.selector)

        if self.unit_type == UnitType.NEIGHBOR:
            return neighbor_indexes(self.selector)

        if self.unit_type == UnitType.ROW:
            return row(self.selector)

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

# DSL FUNCTIONS
def all_traits_are_neighbors_in_unit(unit: Unit, trait: Trait):
    # Example: all_traits_are_neighbors_in_unit(unit(between,pair(0,16)),criminal)
    pass

def all_units_have_at_least_n_traits(unit_type: UnitType, trait: Trait, n: int):
    # Example: all_units_have_at_least_n_traits(row,innocent,2)
    pass

def both_traits_are_neighbors_in_unit(unit: Unit, trait: Trait):
    # Example: both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
    pass

def both_traits_in_unit_are_in_unit(unit_1: Unit, unit_2: Unit, trait: Trait):
    # Example: both_traits_in_unit_are_in_unit(unit(col,3),unit(neighbor,11),innocent)
    pass

def equal_number_of_traits_in_units(unit_1: Unit, unit_2: Unit, trait: Trait):
    # Example: equal_number_of_traits_in_units(unit(neighbor,0),unit(neighbor,7),innocent)
    pass

def equal_traits_and_traits_in_unit(unit: Unit, trait_1: Trait, trait_2: Trait):
    # Example: equal_traits_and_traits_in_unit(unit(between,pair(15,19)),criminal,innocent)
    pass

def every_profession_has_a_trait_in_dir(profession: Profession, trait: Trait, dx: int, dy: int):
    # Example: every_profession_has_a_trait_in_dir(guard,criminal,-1,0)
    pass

def has_most_traits(unit: Unit, trait: Trait):
    # Example: has_most_traits(unit(col,4),innocent)
    pass

def has_trait(cell_id: int, trait: Trait):
    # Example: has_trait(19,innocent)
    pass

def is_not_only_trait_in_unit(unit: Unit, cell_id: int, trait: Trait):
    # Example: is_not_only_trait_in_unit(unit(between,pair(5,7)),5,criminal)
    pass

def is_one_of_n_traits_in_unit(unit: Unit, cell_id: int, trait: Trait, n: int):
    # Example: is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
    pass

def min_number_of_traits_in_unit(unit: Unit, trait: Trait, min_count: int):
    # Example: min_number_of_traits_in_unit(unit(edge,void),criminal,7)
    pass

def more_traits_in_unit_than_unit(unit_1: Unit, unit_2: Unit, trait: Trait):
    # Example: more_traits_in_unit_than_unit(unit(neighbor,2),unit(neighbor,19),innocent)
    pass

def more_traits_than_traits_in_unit(unit: Unit, trait_1: Trait, trait_2: Trait):
    # Example: more_traits_than_traits_in_unit(unit(neighbor,17),criminal,innocent)
    pass

def n_professions_have_trait_in_dir(profession: Profession, trait: Trait, dx: int, dy: int, n: int):
    # Example: n_professions_have_trait_in_dir(singer,criminal,0,-1,0)
    pass

def number_of_traits(trait: Trait, count: int):
    # Example: number_of_traits(criminal,16)
    pass

def number_of_traits_in_unit(unit: Unit, trait: Trait, count: int):
    # Example: number_of_traits_in_unit(unit(between,pair(1,5)),innocent,1)
    pass

def odd_number_of_traits_in_unit(unit: Unit, trait: Trait):
    # Example: odd_number_of_traits_in_unit(unit(profession,guard),innocent)
    pass

def only_one_unit_has_exactly_n_traits(unit_type: UnitType, trait: Trait, n: int):
    # Example: only_one_unit_has_exactly_n_traits(col,innocent,1)
    pass

def only_trait_in_unit_is_in_unit(unit_1: Unit, unit_2: Unit, trait: Trait):
    # Example: only_trait_in_unit_is_in_unit(unit(between,pair(0,2)),unit(between,pair(1,3)),innocent)
    pass

def only_unit_has_exactly_n_traits(unit: Unit, trait: Trait, n: int):
    # Example: only_unit_has_exactly_n_traits(unit(neighbor,19),innocent,1)
    pass

def total_number_of_traits_in_units(unit_1: Unit, unit_2: Unit, trait: Trait, total: int):
    # Example: total_number_of_traits_in_units(unit(neighbor,11),unit(neighbor,16),innocent,4)
    pass

def unit_shares_n_out_of_n_traits_with_unit(unit_1: Unit, unit_2: Unit, trait: Trait, n: int, total: int):
    # Example: unit_shares_n_out_of_n_traits_with_unit(unit(edge,void),unit(neighbor,2),criminal,1,10)
    pass

def units_share_n_traits(unit_1: Unit, unit_2: Unit, trait: Trait, n: int):
    # Example: units_share_n_traits(unit(between,pair(2,14)),unit(neighbor,13),innocent,0)
    pass

def units_share_odd_n_traits(unit_1: Unit, unit_2: Unit, trait: Trait):
    # Example: units_share_odd_n_traits(unit(edge,void),unit(neighbor,7),innocent)
    pass


# DSL Constraint Classes

class AllTraitsAreNeighborsInUnit:
    # Example: all_traits_are_neighbors_in_unit(unit(between,pair(0,16)),criminal)
    # Hint: "All criminals #BETWEEN:pair(0,16) are connected"
    def __init__(self, unit: Unit, trait: Trait, constraint_grid):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        return f"All {self.trait.hint_text()} {self.unit.hint_text()} are connected"

    def eval(self) -> SortRef:
        is_innocent = self.trait == Trait.INNOCENT
        return And([c == is_innocent for c in self.unit.cells()])




class AllUnitsHaveAtLeastNTraits:
    # Example: all_units_have_at_least_n_traits(row,innocent,2)
    # Hint: Each row has at least 2 innocents
    def __init__(self, unit_type: UnitType, trait: Trait, n: int):
        self.unit_type = unit_type
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        if self.unit_type == "neighbor":
            return f"Everyone has at least {self.n} {self.trait.hint_text()} neighbors"
        else:
            return f"Each {self.unit_type.hint_text()} has at least {self.n} {self.trait.hint_text()}"


class BothTraitsAreNeighborsInUnit:
    # Example: both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
    # Hint: "Both innocents #BETWEEN:pair(3,15) are connected"
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        return f"Both {self.trait.hint_text()} {self.unit.hint_text()} are connected"


class BothTraitsInUnitAreInUnit:
    # Example: both_traits_in_unit_are_in_unit(unit(col,3),unit(neighbor,11),innocent)
    # Hint: Both innocents in column #C:3 are #NAMES:11 neighbors
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"Both {self.trait.hint_text()} {self.unit_1.hint_text()} are {self.unit_2.hint_text()}"


class EqualNumberOfTraitsInUnits:
    # Example: equal_number_of_traits_in_units(unit(neighbor,0),unit(neighbor,7),innocent)
    # Hint: #NAME:0 and #NAME:7 have an equal number of innocent neighbors
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"{self.unit_1.hint_text()} and {self.unit_2.hint_text()} have an equal number of {self.trait.value} {self.unit_1.unit_type.hint_text()}"


class EqualTraitsAndTraitsInUnit:
    # Example: equal_traits_and_traits_in_unit(unit(between,pair(15,19)),criminal,innocent)
    # Hint: "There are as many criminals as innocents #BETWEEN:pair(15,19)"
    def __init__(self, unit: Unit, trait_1: Trait, trait_2: Trait):
        self.unit = unit
        self.trait_1 = trait_1
        self.trait_2 = trait_2

    def hint_text(self) -> str:
        return f"There are as many {self.trait_1.hint_text()} as {self.trait_2.hint_text()} {self.unit.hint_text()}"


class EveryProfessionHasATraitInDir:
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


class HasMostTraits:
    # Example: has_most_traits(unit(col,4),innocent)
    # Hint: Column #C:4 has more innocents than any other column
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        unit_type_name = self.unit.unit_type.hint_text()
        return f"{self.unit.hint_text().capitalize()} has more {self.trait.hint_text()} than any other {unit_type_name}"


class HasTrait:
    # Example: has_trait(19,innocent)
    # Hint: Each #PROF:cook neighboring me is innocent
    def __init__(self, cell_id: int, trait: Trait):
        self.cell_id = cell_id
        self.trait = trait

    def hint_text(self) -> str:
        # Note: This hint seems context-dependent based on the example
        return f"Cell {self.cell_id} is {self.trait.value}"


class IsNotOnlyTraitInUnit:
    # Example: is_not_only_trait_in_unit(unit(between,pair(5,7)),5,criminal)
    # Hint: "#NAME:5 is one of two or more criminals #BETWEEN:pair(5,7)"
    def __init__(self, unit: Unit, cell_id: int, trait: Trait):
        self.unit = unit
        self.cell_id = cell_id
        self.trait = trait

    def hint_text(self) -> str:
        return f"Cell {self.cell_id} is one of two or more {self.trait.hint_text()} {self.unit.hint_text()}"


class IsOneOfNTraitsInUnit:
    # Example: is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
    # Hint: #NAME:5 is one of #NAMES:6 5 criminal neighbors
    def __init__(self, unit: Unit, cell_id: int, trait: Trait, n: int):
        self.unit = unit
        self.cell_id = cell_id
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        return f"Cell {self.cell_id} is one of {self.n} {self.trait.value} {self.unit.hint_text()}"


class MinNumberOfTraitsInUnit:
    # Example: min_number_of_traits_in_unit(unit(edge,void),criminal,7)
    # Hint: There are at least 7 criminals on the edges
    def __init__(self, unit: Unit, trait: Trait, min_count: int):
        self.unit = unit
        self.trait = trait
        self.min_count = min_count

    def hint_text(self) -> str:
        return f"There are at least {self.min_count} {self.trait.hint_text()} {self.unit.hint_text()}"


class MoreTraitsInUnitThanUnit:
    # Example: more_traits_in_unit_than_unit(unit(neighbor,2),unit(neighbor,19),innocent)
    # Hint: #NAME:2 has more innocent neighbors than #NAME:19
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"{self.unit_1.hint_text()} has more {self.trait.value} {self.unit_1.unit_type.hint_text()} than {self.unit_2.hint_text()}"


class MoreTraitsThanTraitsInUnit:
    # Example: more_traits_than_traits_in_unit(unit(neighbor,17),criminal,innocent)
    # Hint: #NAME:17 has more criminal than innocent neighbors
    def __init__(self, unit: Unit, trait_1: Trait, trait_2: Trait):
        self.unit = unit
        self.trait_1 = trait_1
        self.trait_2 = trait_2

    def hint_text(self) -> str:
        return f"{self.unit.hint_text()} has more {self.trait_1.value} than {self.trait_2.value} {self.unit.unit_type.hint_text()}"


class NProfessionsHaveTraitInDir:
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


class NumberOfTraits:
    # Example: number_of_traits(criminal,16)
    # Hint: There are 16 criminals in total
    def __init__(self, trait: Trait, count: int):
        self.trait = trait
        self.count = count

    def hint_text(self) -> str:
        return f"There are {self.count} {self.trait.hint_text()} in total"


class NumberOfTraitsInUnit:
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


class OddNumberOfTraitsInUnit:
    # Example: odd_number_of_traits_in_unit(unit(profession,guard),innocent)
    # Hint: There's an odd number of innocent #PROFS:guard
    def __init__(self, unit: Unit, trait: Trait):
        self.unit = unit
        self.trait = trait

    def hint_text(self) -> str:
        return f"There's an odd number of {self.trait.value} {self.unit.hint_text()}"


class OnlyOneUnitHasExactlyNTraits:
    # Example: only_one_unit_has_exactly_n_traits(col,innocent,1)
    # Hint: Only one column has exactly 1 innocent
    def __init__(self, unit_type: UnitType, trait: Trait, n: int):
        self.unit_type = unit_type
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        return f"Only one {self.unit_type.hint_text()} has exactly {self.n} {self.trait.value if self.n == 1 else self.trait.hint_text()}"


class OnlyTraitInUnitIsInUnit:
    # Example: only_trait_in_unit_is_in_unit(unit(between,pair(0,2)),unit(between,pair(1,3)),innocent)
    # Hint: "The only innocent #BETWEEN:pair(0,2) is #BETWEEN:pair(1,3)"
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"The only {self.trait.value} {self.unit_1.hint_text()} is {self.unit_2.hint_text()}"


class OnlyUnitHasExactlyNTraits:
    # Example: only_unit_has_exactly_n_traits(unit(neighbor,19),innocent,1)
    # Hint: #NAME:19 is the only one with only 1 innocent neighbor
    def __init__(self, unit: Unit, trait: Trait, n: int):
        self.unit = unit
        self.trait = trait
        self.n = n

    def hint_text(self) -> str:
        return f"{self.unit.hint_text()} is the only one with only {self.n} {self.trait.value} {self.unit.unit_type.hint_text()}"


class TotalNumberOfTraitsInUnits:
    # Example: total_number_of_traits_in_units(unit(neighbor,11),unit(neighbor,16),innocent,4)
    # Hint: #NAME:11 and #NAME:16 have 4 innocent neighbors in total
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait, total: int):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait
        self.total = total

    def hint_text(self) -> str:
        return f"{self.unit_1.hint_text()} and {self.unit_2.hint_text()} have {self.total} {self.trait.value} {self.unit_1.unit_type.hint_text()} in total"


class UnitSharesNOutOfNTraitsWithUnit:
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


class UnitsShareNTraits:
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


class UnitsShareOddNTraits:
    # Example: units_share_odd_n_traits(unit(edge,void),unit(neighbor,7),innocent)
    # Hint: An odd number of innocents on the edges neighbor #NAME:7
    def __init__(self, unit_1: Unit, unit_2: Unit, trait: Trait):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.trait = trait

    def hint_text(self) -> str:
        return f"An odd number of {self.trait.hint_text()} {self.unit_1.hint_text()} are {self.unit_2.hint_text()}"


@dataclasses.dataclass(frozen=True)
class DSLEvalResult:
    constraint: SortRef
    hint_text: str


class DSLContraintAdapter:
    def __init__(self, dimens: tuple[int, int]):
        self.rows, self.cols = dimens
        self.dsl_to_class_map = {
            # Claude fill in
        }