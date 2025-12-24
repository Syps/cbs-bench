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

from fixtures import (
    simple_4x5_puzzle_state,
    simple_4x5_constraint_grid,
    simple_3x3_puzzle_state,
    simple_3x3_constraint_grid,
)


class TestRowHelper:
    def test_row_returns_correct_row(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = row(1, simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert len(result) == 3
        assert result[0].sexpr() == simple_3x3_constraint_grid[1][0].sexpr()
        assert result[1].sexpr() == simple_3x3_constraint_grid[1][1].sexpr()
        assert result[2].sexpr() == simple_3x3_constraint_grid[1][2].sexpr()

    def test_row_first_row(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        result = row(0, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 4
        for i in range(4):
            assert result[i].sexpr() == simple_4x5_constraint_grid[0][i].sexpr()

    def test_row_last_row(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        result = row(3, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 4
        for i in range(4):
            assert result[i].sexpr() == simple_4x5_constraint_grid[3][i].sexpr()


class TestColHelper:
    def test_col_returns_correct_column(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = col(1, simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert len(result) == 3
        assert result[0].sexpr() == simple_3x3_constraint_grid[0][1].sexpr()
        assert result[1].sexpr() == simple_3x3_constraint_grid[1][1].sexpr()
        assert result[2].sexpr() == simple_3x3_constraint_grid[2][1].sexpr()

    def test_col_first_column(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        result = col(0, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 5
        for i in range(5):
            assert result[i].sexpr() == simple_4x5_constraint_grid[i][0].sexpr()

    def test_col_last_column(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        result = col(3, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 5
        for i in range(5):
            assert result[i].sexpr() == simple_4x5_constraint_grid[i][3].sexpr()


def test_idx_to_coord():
    assert idx_to_coord(1, dimens=(3,3)) == (0, 1)
    assert idx_to_coord(4, dimens=(3, 3)) == (1, 1)
    assert idx_to_coord(8, dimens=(3, 3)) == (2, 2)



class TestBetweenHelper:
    def test_to_the_right_of(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        pair = Pair(1, 3)
        result = between(pair, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 3
        assert result[0].sexpr() == simple_4x5_constraint_grid[0][1].sexpr()
        assert result[1].sexpr() == simple_4x5_constraint_grid[0][2].sexpr()
        assert result[2].sexpr() == simple_4x5_constraint_grid[0][3].sexpr()

    def test_to_the_left_of(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        pair = Pair(0, 2)
        result = between(pair, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 3
        assert result[0].sexpr() == simple_4x5_constraint_grid[0][0].sexpr()
        assert result[1].sexpr() == simple_4x5_constraint_grid[0][1].sexpr()
        assert result[2].sexpr() == simple_4x5_constraint_grid[0][2].sexpr()

    def test_above_end(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        pair = Pair(3, 15)
        result = between(pair, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 4
        assert result[0].sexpr() == simple_4x5_constraint_grid[0][3].sexpr()
        assert result[1].sexpr() == simple_4x5_constraint_grid[1][3].sexpr()
        assert result[2].sexpr() == simple_4x5_constraint_grid[2][3].sexpr()
        assert result[3].sexpr() == simple_4x5_constraint_grid[3][3].sexpr()

    def test_below_start(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        pair = Pair(7, 19)
        result = between(pair, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 4
        assert result[0].sexpr() == simple_4x5_constraint_grid[1][3].sexpr()
        assert result[1].sexpr() == simple_4x5_constraint_grid[2][3].sexpr()
        assert result[2].sexpr() == simple_4x5_constraint_grid[3][3].sexpr()
        assert result[3].sexpr() == simple_4x5_constraint_grid[4][3].sexpr()


class TestNeighborIndexesHelper:
    def test_neighbor_indexes_corner_cell(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = neighbor_indexes(0, simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert sorted(map(str, result)) == ['0_1', '1_0', '1_1']

    def test_neighbor_indexes_edge_cell(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = neighbor_indexes(3, simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert sorted(map(str, result)) == ['0_0', '0_1', '1_1', '2_0', '2_1']

    def test_neighbor_indexes_center_cell(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = neighbor_indexes(4, simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert sorted(map(str, result)) == ['0_0', '0_1', '0_2', '1_0', '1_2', '2_0', '2_1', '2_2']


class TestCornersHelper:
    def test_corners_returns_four_corners_3x3(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = corners(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert len(result) == 4
        expected_corners = [
            simple_3x3_constraint_grid[0][0],  # Top-left
            simple_3x3_constraint_grid[0][2],  # Top-right
            simple_3x3_constraint_grid[2][0],  # Bottom-left
            simple_3x3_constraint_grid[2][2],  # Bottom-right
        ]

        for corner in expected_corners:
            assert any(r.sexpr() == corner.sexpr() for r in result)


class TestEdgesHelper:
    def test_edges_returns_edge_cells_3x3(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        result = edges(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        # In a 3x3 grid, all cells except (1,1) are on the edge
        assert len(result) == 8

        # Center cell should NOT be in result
        center_ref = simple_3x3_constraint_grid[1][1]
        assert not any(r.sexpr() == center_ref.sexpr() for r in result)


class TestProfessionIndexesHelper:
    def test_profession_indexes_finds_all_builders(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        result = profession_bools(Profession.BUILDER, simple_4x5_puzzle_state, simple_4x5_constraint_grid)
        assert sorted(map(str, result)) == ['0_0', '2_3', '3_2', '4_1']

    def test_profession_indexes_returns_empty_for_missing_profession(self, simple_4x5_puzzle_state,
                                                                     simple_4x5_constraint_grid):
        result = profession_bools(Profession.PAINTER, simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert len(result) == 0

class TestAllTraitsAreNeighborsInUnit:
    def test_all_criminals_above_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all criminals in a vertical unit (column) are connected"""
        unit = Unit(UnitType.BETWEEN, Pair(3, 15))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        expected_constraint = And(
            Implies(
                And(
                    simple_4x5_constraint_grid[0][-1],
                    simple_4x5_constraint_grid[2][-1]
                ),
                simple_4x5_constraint_grid[1][-1]
            ),
            Implies(
                And(
                    simple_4x5_constraint_grid[1][-1],
                    simple_4x5_constraint_grid[3][-1],
                ),
                simple_4x5_constraint_grid[2][-1]
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)
        assert result.hint_text == "All criminals above David are connected"

    def test_all_innocents_above_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all innocents in a vertical unit (column) are connected"""
        # Pair(3, 15) is col 3, rows 0-3 (indices 3, 7, 11, 15)
        unit = Unit(UnitType.BETWEEN, Pair(3, 15))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.INNOCENT)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # For innocents, we check NOT criminal
        expected_constraint = And(
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[0][-1]),
                    Not(simple_4x5_constraint_grid[2][-1])
                ),
                Not(simple_4x5_constraint_grid[1][-1])
            ),
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[1][-1]),
                    Not(simple_4x5_constraint_grid[3][-1]),
                ),
                Not(simple_4x5_constraint_grid[2][-1])
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)
        assert result.hint_text == "All innocents above David are connected"

    def test_all_criminals_to_right_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all criminals in a horizontal unit (row) are connected"""
        # Pair(4, 6) is row 1, cols 0-2 (indices 4, 5, 6)
        unit = Unit(UnitType.BETWEEN, Pair(4, 6))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        expected_constraint = And(
            Implies(
                And(
                    simple_4x5_constraint_grid[1][0],
                    simple_4x5_constraint_grid[1][2]
                ),
                simple_4x5_constraint_grid[1][1]
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)
        assert result.hint_text == "All criminals to the left of Olivia are connected"

    def test_all_innocents_to_right_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all innocents in a horizontal unit (row) are connected"""
        # Pair(0, 2) is row 0, cols 0-2 (indices 0, 1, 2)
        unit = Unit(UnitType.BETWEEN, Pair(0, 2))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.INNOCENT)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # For innocents, we check NOT criminal
        expected_constraint = And(
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[0][0]),
                    Not(simple_4x5_constraint_grid[0][2])
                ),
                Not(simple_4x5_constraint_grid[0][1])
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)

    def test_all_criminals_to_left_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all criminals in a horizontal unit (row, reversed) are connected"""
        # Pair(14, 12) is row 3, cols 2-0 (indices 14, 13, 12) - reversed
        unit = Unit(UnitType.BETWEEN, Pair(14, 12))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        expected_constraint = And(
            Implies(
                And(
                    simple_4x5_constraint_grid[3][0],
                    simple_4x5_constraint_grid[3][2]
                ),
                simple_4x5_constraint_grid[3][1]
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)

    def test_all_innocents_to_left_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all innocents in a horizontal unit (row, reversed) are connected"""
        unit = Unit(UnitType.BETWEEN, Pair(10, 8))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.INNOCENT)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # For innocents, we check NOT criminal
        expected_constraint = And(
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[2][0]),
                    Not(simple_4x5_constraint_grid[2][2])
                ),
                Not(simple_4x5_constraint_grid[2][1])
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)

    def test_all_criminals_below_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all criminals in a vertical unit (column, downward) are connected"""
        # Pair(0, 16) is col 0, rows 0-4 (indices 0, 4, 8, 12, 16)
        unit = Unit(UnitType.BETWEEN, Pair(0, 16))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        expected_constraint = And(
            Implies(
                And(
                    simple_4x5_constraint_grid[0][0],
                    simple_4x5_constraint_grid[2][0]
                ),
                simple_4x5_constraint_grid[1][0]
            ),
            Implies(
                And(
                    simple_4x5_constraint_grid[1][0],
                    simple_4x5_constraint_grid[3][0]
                ),
                simple_4x5_constraint_grid[2][0]
            ),
            Implies(
                And(
                    simple_4x5_constraint_grid[2][0],
                    simple_4x5_constraint_grid[4][0]
                ),
                simple_4x5_constraint_grid[3][0]
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)

    def test_all_innocents_below_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all innocents in a vertical unit (column, downward) are connected"""
        # Pair(1, 17) is col 1, rows 0-4 (indices 1, 5, 9, 13, 17)
        unit = Unit(UnitType.BETWEEN, Pair(1, 17))
        const = AllTraitsAreNeighborsInUnit(unit, Trait.INNOCENT)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # For innocents, we check NOT criminal
        expected_constraint = And(
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[0][1]),
                    Not(simple_4x5_constraint_grid[2][1])
                ),
                Not(simple_4x5_constraint_grid[1][1])
            ),
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[1][1]),
                    Not(simple_4x5_constraint_grid[3][1])
                ),
                Not(simple_4x5_constraint_grid[2][1])
            ),
            Implies(
                And(
                    Not(simple_4x5_constraint_grid[2][1]),
                    Not(simple_4x5_constraint_grid[4][1])
                ),
                Not(simple_4x5_constraint_grid[3][1])
            )
        )
        actual_constraint = result.constraint

        assert actual_constraint.eq(expected_constraint)




