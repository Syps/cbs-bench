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
from solver import eval_cell_hint_dsl
from models import PuzzleCell, Status

from fixtures import (
    simple_3x3_puzzle_state,
    simple_3x3_constraint_grid,
    simple_4x5_puzzle_state,
    simple_4x5_constraint_grid,
)


class TestEvalCellHintDSL:

    def test_all_rows_have_at_least_two_innocents(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        dsl_str = "all_units_have_at_least_n_traits(row,innocent,2)"
        cell = simple_3x3_puzzle_state[0][0]
        cell.orig_hint = dsl_str
        actual = eval_cell_hint_dsl(
            cell,
            simple_3x3_puzzle_state,
            simple_3x3_constraint_grid
        )
        expected = DSLEvalResult(
            hint_text="All rows have at least two innocents",
            constraint=And([
                Sum([c for c in simple_3x3_constraint_grid[0]]) == 1,
                Sum([c for c in simple_3x3_constraint_grid[1]]) == 1,
                Sum([c for c in simple_3x3_constraint_grid[2]]) == 1,
            ])
        )

        assert actual == expected

    def test_all_rows_have_at_least_two_criminals(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        dsl_str = "all_units_have_at_least_n_traits(row,criminal,2)"
        cell = simple_3x3_puzzle_state[0][0]
        cell.orig_hint = dsl_str
        actual = eval_cell_hint_dsl(
            cell,
            simple_3x3_puzzle_state,
            simple_3x3_constraint_grid
        )
        expected = DSLEvalResult(
            hint_text="All rows have at least two criminals",
            constraint=And([
                Sum([c for c in simple_3x3_constraint_grid[0]]) == 2,
                Sum([c for c in simple_3x3_constraint_grid[1]]) == 2,
                Sum([c for c in simple_3x3_constraint_grid[2]]) == 2,
            ])
        )

        assert actual == expected

    # def test_all_innocents_are_neighbors_in_unit(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
    #     dsl_str = "all_traits_are_neighbors_in_unit(unit(between, pair(0, 1)), innocent)"
    #     cell = simple_3x3_puzzle_state[0][0]
    #     cell.orig_hint = dsl_str
    #     actual = eval_cell_hint_dsl(
    #         cell,
    #         simple_3x3_puzzle_state,
    #         simple_3x3_constraint_grid
    #     )
    #     expected = DSLEvalResult(
    #
    #     )
