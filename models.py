"""Data models, enums, and exceptions for the logic solver."""

from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import threading


class CellData(BaseModel):
    """Raw puzzle cell data from cluesbysam."""
    name: str
    profession: str
    hint: str
    criminal: bool
    gender: str
    orig_hint: str = ""
    paths: List[List[int]] = []


class Status(Enum):
    """Status of a puzzle cell during gameplay."""
    UNKNOWN = 0
    CRIMINAL = 1
    INNOCENT = 2


class PuzzleCell(BaseModel):
    """A cell in the puzzle grid during gameplay."""
    name: str
    profession: str
    gender: str
    orig_hint: str = ""
    clue: str
    status: Status = Status.UNKNOWN
    had_mistake: bool = False
    paths: List[List[int]] = []
    is_criminal: bool = False

type PuzzleState = List[List[PuzzleCell]]


class SerializationMethod(Enum):
    """Method for serializing puzzle state."""
    DEFAULT = "default"


@dataclass
class TestResult:
    """Complete test results for a model run."""
    puzzle_data: List[dict]
    model_name: str
    puzzle_identifier: str
    test_date: str
    duration_seconds: float
    conversation: List[dict]
    moves: List[dict]
    completed: bool
    total_moves: int
    max_moves_reached: bool
    tokens_used: int
    cost_usd: float


class ModelCommunicationError(Exception):
    """Raised when model API communication fails."""
    pass


class GameStateError(Exception):
    """Raised for invalid game state transitions."""
    pass


class ModelProgress:
    """Track progress of a single model test (thread-safe)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.current_move = 0
        self.correct_moves = 0
        self.incorrect_moves = 0
        self.completed = False
        self.error = None
        self.lock = threading.Lock()

    def update(self, move_number: int, is_correct: bool = None):
        """Update progress with move information."""
        with self.lock:
            self.current_move = move_number
            if is_correct is not None:
                if is_correct:
                    self.correct_moves += 1
                else:
                    self.incorrect_moves += 1

    def mark_completed(self, completed: bool = True):
        """Mark the test as completed."""
        with self.lock:
            self.completed = completed

    def mark_error(self, error: str):
        """Mark the test as errored."""
        with self.lock:
            self.error = error
            self.completed = True

    def get_status(self) -> Dict:
        """Get current status snapshot."""
        with self.lock:
            return {
                'model': self.model_name,
                'move': self.current_move,
                'correct': self.correct_moves,
                'incorrect': self.incorrect_moves,
                'completed': self.completed,
                'error': self.error
            }

import enum
class PuzzleDifficulty(enum.Enum):
    EASY = 0
    MEDIUM = 1
    TRICKY = 2
    HARD = 3
    BRUTAL = 4
    EVIL = 5


MODEL_COST_USD_PER_1M_TOKENS = {  # non-cached tokens
    "gpt-5-pro": {
        "input": 125,
        "output": 1000,
    },
    "claude-sonnet-4-5-20250929": {
        "input": 300,
        "output": 1500,
    },
    "deepseek-chat": {
        "input": 28,
        "output": 42,
    },
    "gemini-2.5-pro": {
        "input": 125,
        "output": 1000,
    },
}


PUZZLE_PACK_1_DIFFICULTIES = {
    **{x: PuzzleDifficulty.EASY for x in range(1,6)},
    **{x: PuzzleDifficulty.MEDIUM for x in range(6,21)},
    **{x: PuzzleDifficulty.TRICKY for x in range(21,36)},
    **{x: PuzzleDifficulty.HARD for x in range(36,46)},
    **{x: PuzzleDifficulty.BRUTAL for x in range(46,49)},
    **{x: PuzzleDifficulty.EVIL for x in range(49,51)},
}