import pytest
from z3 import *

from cbs_bench.generate.validate import PuzzleValidator, PuzzleValidationResult
from cbs_bench.models import PuzzleCell, Status
from cbs_bench.generate.hint_dsl_functions import (
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


class TestPuzzleValidator:

    def test_validate_correct(self):
        puzzle_state = [
            [
                PuzzleCell(
                    name='Aaron',
                    profession='guard',
                    gender='male',
                    orig_hint="all_units_have_at_least_n_traits(row,innocent,2)",
                    clue="",
                    is_criminal=True,
                    paths=[],
                    status=Status.CRIMINAL,
                )
                ,

                PuzzleCell(
                    name='Beth',
                    profession='clerk',
                    gender='female',
                    orig_hint=None,
                    clue="",
                    is_criminal=False,
                    paths=[[0]]
                )
                ,

                PuzzleCell(
                    name='Carlos',
                    profession='coder',
                    gender='male',
                    orig_hint=None,
                    clue="",
                    is_criminal=False,
                    paths=[[0]]
                )
            ],
        ]
        validator = PuzzleValidator(puzzle_state)
        actual = validator.validate()
        expected = PuzzleValidationResult(
            valid=True,
            invalid_message=None,
        )

        assert actual == expected

    def test_validate_incorrect_logic_error(self):
        puzzle_state = [
            [
                PuzzleCell(
                    name='Aaron',
                    profession='guard',
                    gender='male',
                    orig_hint="all_units_have_at_least_n_traits(row,innocent,3)",
                    clue="",
                    is_criminal=True,
                    paths=[],
                    status=Status.CRIMINAL,
                )
                ,

                PuzzleCell(
                    name='Beth',
                    profession='clerk',
                    gender='female',
                    orig_hint=None,
                    clue="",
                    is_criminal=False,
                    paths=[[0]],
                )
                ,

                PuzzleCell(
                    name='Carlos',
                    profession='coder',
                    gender='male',
                    orig_hint=None,
                    clue="",
                    is_criminal=False,
                    paths=[[0]],
                )
            ],
        ]
        validator = PuzzleValidator(puzzle_state)
        actual = validator.validate()
        expected = PuzzleValidationResult(
            valid=False,
            invalid_message='Conflicting constraints: Aaron is criminal AND All rows have at least three innocents',
        )

        assert actual == expected

    def test_validate_incorrect_unreachable_cells(self):
        puzzle_state = [
            [
                PuzzleCell(
                    name='Aaron',
                    profession='guard',
                    gender='male',
                    orig_hint="all_units_have_at_least_n_traits(row,innocent,1)",
                    clue="",
                    is_criminal=True,
                    paths=[],
                    status=Status.CRIMINAL,
                )
                ,

                PuzzleCell(
                    name='Beth',
                    profession='clerk',
                    gender='female',
                    orig_hint=None,
                    clue="",
                    is_criminal=False,
                    paths=[[0]],
                )
                ,

                PuzzleCell(
                    name='Carlos',
                    profession='coder',
                    gender='male',
                    orig_hint=None,
                    clue="",
                    is_criminal=False,
                    paths=[[0]],
                )
            ],
        ]
        validator = PuzzleValidator(puzzle_state)
        actual = validator.validate()
        expected = PuzzleValidationResult(
            valid=False,
            invalid_message=PuzzleValidator.VALIDATION_ERROR_UNREACHABLE_CELLS,
        )

        assert actual == expected
