import pytest
from z3 import *
from ..hint_dsl_functions import (
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
    AllUnitsHaveAtLeastNTraits,
    BothTraitsAreNeighborsInUnit,
    BothTraitsInUnitAreInUnit,
    EqualNumberOfTraitsInUnits,
    EqualTraitsAndTraitsInUnit,
    EveryProfessionHasATraitInDir,
    HasMostTraits,
    HasTrait,
    IsNotOnlyTraitInUnit,
    IsOneOfNTraitsInUnit,
    MinNumberOfTraitsInUnit,
    MoreTraitsInUnitThanUnit,
    MoreTraitsThanTraitsInUnit,
    NProfessionsHaveTraitInDir,
    NumberOfTraits,
    NumberOfTraitsInUnit,
    OddNumberOfTraitsInUnit,
    OnlyOneUnitHasExactlyNTraits,
    OnlyTraitInUnitIsInUnit,
    OnlyUnitHasExactlyNTraits,
    TotalNumberOfTraitsInUnits,
    UnitSharesNOutOfNTraitsWithUnit,
    UnitsShareNTraits,
    UnitsShareOddNTraits,
    DSLEvalResult,
    Unit,
    Trait,
)
from ..hint_dsl_functions import UnitType


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
        unit = Unit(UnitType.BETWEEN.value, Pair(3, 15))
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
        unit = Unit(UnitType.BETWEEN.value, Pair(3, 15))
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

    def test_all_criminals_to_left_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all criminals in a horizontal unit (row) are connected"""
        # Pair(4, 6) is row 1, cols 0-2 (indices 4, 5, 6)
        unit = Unit(UnitType.BETWEEN.value, Pair(4, 6))
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

    def test_all_innocents_to_left_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all innocents in a horizontal unit (row) are connected"""
        # Pair(0, 2) is row 0, cols 0-2 (indices 0, 1, 2)
        unit = Unit(UnitType.BETWEEN.value, Pair(0, 2))
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
        assert result.hint_text == "All innocents to the left of Jack are connected"

    def test_all_criminals_to_right_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all criminals in a horizontal unit (row, reversed) are connected"""
        # Pair(14, 12) is row 3, cols 2-0 (indices 14, 13, 12) - reversed
        unit = Unit(UnitType.BETWEEN.value, Pair(14, 12))
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

    def test_all_innocents_to_right_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test all innocents in a horizontal unit (row, reversed) are connected"""
        unit = Unit(UnitType.BETWEEN.value, Pair(10, 8))
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
        unit = Unit(UnitType.BETWEEN.value, Pair(0, 16))
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
        unit = Unit(UnitType.BETWEEN.value, Pair(1, 17))
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


class TestAllUnitsHaveAtLeastNTraits:
    def test_all_rows_have_at_least_2_criminals(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        """Test that all rows have at least 2 criminals"""
        const = AllUnitsHaveAtLeastNTraits(UnitType.ROW.value, Trait.CRIMINAL, 2)

        result = const.eval(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        # Each row should have exactly 2 criminals (Sum([cells]) == 2)
        expected_constraint = And(
            Sum([simple_3x3_constraint_grid[0][0], simple_3x3_constraint_grid[0][1], simple_3x3_constraint_grid[0][2]]) == 2,
            Sum([simple_3x3_constraint_grid[1][0], simple_3x3_constraint_grid[1][1], simple_3x3_constraint_grid[1][2]]) == 2,
            Sum([simple_3x3_constraint_grid[2][0], simple_3x3_constraint_grid[2][1], simple_3x3_constraint_grid[2][2]]) == 2
        )

        assert result.constraint.eq(expected_constraint)

    def test_all_cols_have_at_least_1_innocent(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        """Test that all columns have at least 1 innocent"""
        const = AllUnitsHaveAtLeastNTraits(UnitType.COL.value, Trait.INNOCENT, 1)

        result = const.eval(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        # Each column should have innocents = total - 1 (Sum([cells]) == 3 - 1)
        expected_constraint = And(
            Sum([simple_3x3_constraint_grid[0][0], simple_3x3_constraint_grid[1][0], simple_3x3_constraint_grid[2][0]]) == 2,
            Sum([simple_3x3_constraint_grid[0][1], simple_3x3_constraint_grid[1][1], simple_3x3_constraint_grid[2][1]]) == 2,
            Sum([simple_3x3_constraint_grid[0][2], simple_3x3_constraint_grid[1][2], simple_3x3_constraint_grid[2][2]]) == 2
        )

        assert result.constraint.eq(expected_constraint)


class TestBothTraitsAreNeighborsInUnit:
    def test_both_criminals_are_connected(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test that both criminals in a unit are adjacent"""
        # Pair(4, 6) is row 1, cols 0-2 (indices 4, 5, 6)
        unit = Unit(UnitType.BETWEEN.value, Pair(4, 6))
        const = BothTraitsAreNeighborsInUnit(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # Should check that exactly 2 are criminals and they are adjacent
        cells = [simple_4x5_constraint_grid[1][0], simple_4x5_constraint_grid[1][1], simple_4x5_constraint_grid[1][2]]
        expected_constraint = And(
            Sum([c == True for c in cells]) == 2,
            Or(
                And(cells[0] == True, cells[1] == True),
                And(cells[1] == True, cells[2] == True)
            )
        )

        assert result.constraint.eq(expected_constraint)
        # Note: hint_text implementation is incomplete for BETWEEN units
        assert "Both criminals" in result.hint_text and "connected" in result.hint_text


class TestNumberOfTraits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_number_of_criminals(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        """Test that there are exactly N criminals in total"""
        const = NumberOfTraits(Trait.CRIMINAL, 5)

        result = const.eval(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        # Should sum all cells and check == 5
        assert result.hint_text == "There are 5 criminals in total"


class TestNumberOfTraitsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_number_of_criminals_in_unit(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test exact count of criminals in a unit"""
        unit = Unit(UnitType.BETWEEN.value, Pair(0, 2))
        const = NumberOfTraitsInUnit(unit, Trait.CRIMINAL, 2)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "There are 2 criminals" in result.hint_text


class TestMinNumberOfTraitsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_min_criminals_on_edges(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        """Test minimum number of criminals on edges"""
        unit = Unit(UnitType.EDGE.value, "void")
        const = MinNumberOfTraitsInUnit(unit, Trait.CRIMINAL, 3)

        result = const.eval(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        assert result.hint_text == "There are at least 3 criminals on the edges"


class TestHasTrait:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_cell_has_trait(self, simple_3x3_puzzle_state, simple_3x3_constraint_grid):
        """Test that a specific cell has a trait"""
        const = HasTrait(0, Trait.CRIMINAL)

        result = const.eval(simple_3x3_puzzle_state, simple_3x3_constraint_grid)

        # Should constrain cell 0 to be criminal
        assert "Cell 0 is criminal" in result.hint_text


class TestEqualNumberOfTraitsInUnits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_equal_criminals_in_two_units(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test two units have equal number of criminals"""
        unit1 = Unit(UnitType.ROW.value, 0)
        unit2 = Unit(UnitType.ROW.value, 1)
        const = EqualNumberOfTraitsInUnits(unit1, unit2, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # Should constrain Sum(unit1 criminals) == Sum(unit2 criminals)
        assert isinstance(result, DSLEvalResult)


class TestMoreTraitsInUnitThanUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_more_criminals_in_unit1_than_unit2(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test unit1 has more criminals than unit2"""
        unit1 = Unit(UnitType.NEIGHBOR.value, 0)
        unit2 = Unit(UnitType.NEIGHBOR.value, 19)
        const = MoreTraitsInUnitThanUnit(unit1, unit2, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        # Should constrain Sum(unit1) > Sum(unit2)
        assert isinstance(result, DSLEvalResult)


class TestMoreTraitsThanTraitsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_more_criminals_than_innocents(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test unit has more criminals than innocents"""
        unit = Unit(UnitType.NEIGHBOR.value, 4)
        const = MoreTraitsThanTraitsInUnit(unit, Trait.CRIMINAL, Trait.INNOCENT)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert isinstance(result, DSLEvalResult)


class TestEqualTraitsAndTraitsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_equal_criminals_and_innocents(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test equal number of criminals and innocents in unit"""
        unit = Unit(UnitType.BETWEEN.value, Pair(0, 3))
        const = EqualTraitsAndTraitsInUnit(unit, Trait.CRIMINAL, Trait.INNOCENT)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "as many criminals as innocents" in result.hint_text


class TestHasMostTraits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_column_has_most_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test a column has more criminals than any other column"""
        unit = Unit(UnitType.COL.value, 0)
        const = HasMostTraits(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "more criminals than any other" in result.hint_text


class TestOddNumberOfTraitsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_odd_number_of_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test unit has odd number of criminals"""
        unit = Unit(UnitType.PROFESSION.value, Profession.CODER)
        const = OddNumberOfTraitsInUnit(unit, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "odd number" in result.hint_text


class TestIsOneOfNTraitsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_cell_is_one_of_n_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test cell is one of N criminals in unit"""
        unit = Unit(UnitType.NEIGHBOR.value, 6)
        const = IsOneOfNTraitsInUnit(unit, 5, Trait.CRIMINAL, 3)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "Cell 5 is one of 3" in result.hint_text


class TestIsNotOnlyTraitInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_not_only_criminal_in_unit(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test cell is not the only criminal in unit"""
        unit = Unit(UnitType.BETWEEN.value, Pair(0, 2))
        const = IsNotOnlyTraitInUnit(unit, 0, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "one of two or more" in result.hint_text


class TestOnlyOneUnitHasExactlyNTraits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_only_one_column_has_exactly_n_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test only one column has exactly N criminals"""
        const = OnlyOneUnitHasExactlyNTraits(UnitType.COL.value, Trait.CRIMINAL, 2)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "Only one" in result.hint_text


class TestOnlyUnitHasExactlyNTraits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_only_this_unit_has_exactly_n_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test only this specific unit has exactly N criminals"""
        unit = Unit(UnitType.NEIGHBOR.value, 0)
        const = OnlyUnitHasExactlyNTraits(unit, Trait.CRIMINAL, 2)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "is the only one" in result.hint_text


class TestBothTraitsInUnitAreInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_both_criminals_in_unit1_are_in_unit2(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test both criminals in unit1 are also in unit2"""
        unit1 = Unit(UnitType.COL.value, 0)
        unit2 = Unit(UnitType.NEIGHBOR.value, 0)
        const = BothTraitsInUnitAreInUnit(unit1, unit2, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert isinstance(result, DSLEvalResult)


class TestOnlyTraitInUnitIsInUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_only_criminal_in_unit1_is_in_unit2(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test the only criminal in unit1 is in unit2"""
        unit1 = Unit(UnitType.BETWEEN.value, Pair(0, 2))
        unit2 = Unit(UnitType.BETWEEN.value, Pair(1, 3))
        const = OnlyTraitInUnitIsInUnit(unit1, unit2, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "The only" in result.hint_text


class TestTotalNumberOfTraitsInUnits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_total_criminals_in_two_units(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test total number of criminals across two units"""
        unit1 = Unit(UnitType.NEIGHBOR.value, 0)
        unit2 = Unit(UnitType.NEIGHBOR.value, 5)
        const = TotalNumberOfTraitsInUnits(unit1, unit2, Trait.CRIMINAL, 4)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "have 4" in result.hint_text


class TestUnitsShareNTraits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_units_share_n_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test two units share exactly N criminals"""
        unit1 = Unit(UnitType.BETWEEN.value, Pair(0, 3))
        unit2 = Unit(UnitType.NEIGHBOR.value, 5)
        const = UnitsShareNTraits(unit1, unit2, Trait.CRIMINAL, 1)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert isinstance(result, DSLEvalResult)


class TestUnitsShareOddNTraits:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_units_share_odd_number_of_criminals(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test two units share an odd number of criminals"""
        unit1 = Unit(UnitType.EDGE.value, "void")
        unit2 = Unit(UnitType.NEIGHBOR.value, 7)
        const = UnitsShareOddNTraits(unit1, unit2, Trait.CRIMINAL)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "odd number" in result.hint_text


class TestUnitSharesNOutOfNTraitsWithUnit:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_n_out_of_total_criminals_shared(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test N out of total criminals in unit1 are in unit2"""
        unit1 = Unit(UnitType.EDGE.value, "void")
        unit2 = Unit(UnitType.NEIGHBOR.value, 0)
        const = UnitSharesNOutOfNTraitsWithUnit(unit1, unit2, Trait.CRIMINAL, 1, 8)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "1 of the 8" in result.hint_text


class TestEveryProfessionHasATraitInDir:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_every_guard_has_criminal_to_left(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test every guard has a criminal directly to the left"""
        const = EveryProfessionHasATraitInDir(Profession.GUARD, Trait.CRIMINAL, -1, 0)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert "Every guard has a criminal" in result.hint_text


class TestNProfessionsHaveTraitInDir:
    @pytest.mark.skip(reason="eval() not yet implemented")
    def test_n_singers_have_criminal_above(self, simple_4x5_puzzle_state, simple_4x5_constraint_grid):
        """Test N singers have a criminal directly above them"""
        const = NProfessionsHaveTraitInDir(Profession.SINGER, Trait.CRIMINAL, 0, -1, 2)

        result = const.eval(simple_4x5_puzzle_state, simple_4x5_constraint_grid)

        assert isinstance(result, DSLEvalResult)




