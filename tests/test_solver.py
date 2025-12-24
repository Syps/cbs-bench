import pytest
from z3 import *
from hint_dsl_functions import (
    Profession,
    Pair,
    between,
    col,
    row,
    corners,
    edges,
    profession_bools,
    neighbor_indexes,
    idx_to_coord,
    AllTraitsAreNeighborsInUnit,
    DSLEvalResult,
    Unit,
    Trait,
)
from hint_dsl_functions import UnitType
from models import PuzzleCell, Status





class TestSolver:
    pass