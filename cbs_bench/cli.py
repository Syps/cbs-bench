"""
CLI module for logic-solver.

This module contains all Click command-line interface commands and helpers,
including test execution, statistics generation, and game replay functionality.
It also contains Z3 constraint solving utilities for logic puzzles.
"""

import click
import os
import json
from typing import List, Tuple, Dict
from collections import defaultdict
from datetime import datetime

from z3 import Bool, BoolRef, Solver, sat, Sum, If, ExprRef

# Import from other modules
from .fetch import (
    fetch_clues_from_website,
    load_puzzle_from_cache,
    fetch_and_cache_puzzle
)
from .models import CellData, PuzzleCell, TestResult, PuzzleDifficulty, PUZZLE_PACK_1_DIFFICULTIES
from .solve import (
    AVAILABLE_MODELS,
    ModelFactory,
    ModelTester,
    initialize_puzzle_state,
    serialize_puzzle_state,
    SerializationMethod,
    run_model_test,
    run_all_models_parallel,
    save_test_results,
    print_test_evaluation,
    replace_name_references
)

# Constants
ROWS = 5
COLS = 4


# =============================================================================
# CLI Helper Functions (for stats command)
# =============================================================================

def validate_puzzle(value):
    """Validate puzzle argument - must be date (YYYYMMDD) or URL."""
    if value is None:
        return datetime.now().strftime("%Y%m%d")

    # Check if it's a URL
    if value.startswith(('http://', 'https://')):
        return value

    # Check if it's a valid date in YYYYMMDD format
    try:
        datetime.strptime(value, "%Y%m%d")
        return value
    except ValueError:
        raise click.BadParameter(f"Puzzle must be a date in YYYYMMDD format or a URL, got: {value}")

def load_test_data(test_dir: str) -> Tuple[dict, list]:
    """Load metadata and moves from a test directory.

    Returns:
        Tuple of (metadata_dict, moves_list) or (None, None) if files missing/invalid
    """
    try:
        metadata_path = os.path.join('.test_results', test_dir, 'metadata.json')
        moves_path = os.path.join('.test_results', test_dir, 'moves.json')

        if not os.path.exists(metadata_path) or not os.path.exists(moves_path):
            return None, None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        with open(moves_path, 'r') as f:
            moves = json.load(f)

        return metadata, moves
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load data from {test_dir}: {e}")
        return None, None

def is_legitimate_finish(metadata: dict, moves: list) -> bool:
    """Check if game ended legitimately (not an early stop).

    A game is legitimate if ANY of these are true:
    1. Puzzle was completed successfully
    2. Max moves limit was reached
    3. Last 5 moves are not all incorrect (not stuck)
    """
    # Criterion 1: Completed successfully
    if metadata['completed']:
        return True

    # Criterion 2: Hit max moves
    if metadata['max_moves_reached']:
        return True

    # Criterion 3: Last 5 moves not all incorrect
    last_5_moves = moves[-5:]

    return len(last_5_moves) >=5 and all(c['correct'] == False for c in last_5_moves)

def calculate_model_stats(tests: List[Tuple[dict, list]]) -> dict:
    """Calculate all statistics for a list of tests.

    Args:
        tests: List of (metadata, moves) tuples

    Returns:
        Dictionary with statistics: puzzles_completed, pct_moves_correct, etc.
    """
    if not tests:
        return {
            'puzzles_completed': 0,
            'pct_moves_correct': 0.0,
            'avg_cells_correct': 0.0,
            'avg_time_per_move': 0.0,
            'puzzles_completed_no_early_stop': 0
        }

    # Puzzles completed
    puzzles_completed = sum(1 for metadata, _ in tests if metadata.get('completed', False))

    # Percentage of moves correct
    total_moves = 0
    correct_moves = 0
    for metadata, moves in tests:
        total_moves += len(moves)
        correct_moves += sum(1 for move in moves if move.get('correct', False))

    pct_moves_correct = (correct_moves / total_moves * 100) if total_moves > 0 else 0.0

    # Average cells correct (% moves correct as decimal * 20 cells per game)
    avg_cells_correct = (pct_moves_correct / 100) * 20

    # Average time per move
    total_duration = sum(metadata.get('duration_seconds', 0.0) for metadata, _ in tests)
    avg_time_per_move = total_duration / total_moves if total_moves > 0 else 0.0

    # Puzzles completed (excluding early stops)
    legitimate_tests = [(m, mv) for m, mv in tests if is_legitimate_finish(m, mv)]
    puzzles_completed_no_early_stop = sum(1 for m, _ in legitimate_tests if m.get('completed', False))

    return {
        'puzzles_completed': puzzles_completed,
        'pct_moves_correct': pct_moves_correct,
        'avg_cells_correct': avg_cells_correct,
        'avg_time_per_move': avg_time_per_move,
        'puzzles_completed_no_early_stop': puzzles_completed_no_early_stop
    }

def extract_puzzle_pack_number(puzzle_identifier: str) -> int | None:
    """Extract puzzle number from puzzle pack URL.

    Args:
        puzzle_identifier: Either a date string or URL

    Returns:
        Puzzle number if URL matches puzzle pack pattern, None otherwise
    """
    import re

    # Pattern for puzzle pack URL: .../pack-1/12
    pattern = r'/pack-1/(\d+)/?$'
    match = re.search(pattern, puzzle_identifier)

    if match:
        return int(match.group(1))

    return None

def get_puzzle_difficulty(puzzle_identifier: str) -> PuzzleDifficulty | None:
    """Get difficulty level for a puzzle identifier.

    Args:
        puzzle_identifier: Either a date string or URL

    Returns:
        PuzzleDifficulty enum value, or None if not a puzzle pack puzzle
    """
    puzzle_num = extract_puzzle_pack_number(puzzle_identifier)

    if puzzle_num is None:
        return None

    return PUZZLE_PACK_1_DIFFICULTIES.get(puzzle_num)

def format_stats_table(model_stats: Dict[str, dict]) -> str:
    """Format statistics as a printable table.

    Args:
        model_stats: Dictionary mapping model_name to stats dict

    Returns:
        Formatted table string
    """
    if not model_stats:
        return "No test results found."

    lines = []
    lines.append("="*145)
    lines.append("MODEL PERFORMANCE STATISTICS")
    lines.append("="*145)
    lines.append(f"{'Model':<35} {'# Puzzles Completed':>19}  {'% Moves Correct':>16}  {'Avg Cells Correct':>18}  {'Avg Time/Move':>14}  {'# Completed (No Early Stop)':>28}")
    lines.append("="*145)

    # Sort by model name
    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]

        puzzles_completed = stats['puzzles_completed']
        pct_correct = f"{stats['pct_moves_correct']:.1f}%"
        avg_cells = f"{stats['avg_cells_correct']:.1f}"
        time_per_move = f"{stats['avg_time_per_move']:.1f}s"
        completed_no_early = stats['puzzles_completed_no_early_stop']

        lines.append(f"{model_name:<35} {puzzles_completed:>19}  {pct_correct:>16}  {avg_cells:>18}  {time_per_move:>14}  {completed_no_early:>28}")

    lines.append("="*145)

    return "\n".join(lines)

def format_difficulty_stats_table(difficulty_stats: Dict[PuzzleDifficulty, Dict[str, dict]],
                                  difficulty_puzzles: Dict[PuzzleDifficulty, set[int]]) -> str:
    """Format difficulty-based statistics as a printable table.

    Args:
        difficulty_stats: Dictionary mapping difficulty to {model_name: stats_dict}
        difficulty_puzzles: Dictionary mapping difficulty to set of puzzle numbers

    Returns:
        Formatted table string
    """
    if not difficulty_stats:
        return "No test results found."

    lines = []

    # Sort difficulties by their enum value (EASY=0, MEDIUM=1, etc.)
    for difficulty in sorted(difficulty_stats.keys(), key=lambda d: d.value):
        model_stats = difficulty_stats[difficulty]
        puzzle_numbers = sorted(difficulty_puzzles.get(difficulty, set()))

        lines.append("\n" + "="*145)
        lines.append(f"DIFFICULTY: {difficulty.name}")
        lines.append(f"Puzzles: {', '.join(map(str, puzzle_numbers))}")
        lines.append("="*145)
        lines.append(f"{'Model':<35} {'# Puzzles Completed':>19}  {'% Moves Correct':>16}  {'Avg Cells Correct':>18}  {'Avg Time/Move':>14}  {'# Completed (No Early Stop)':>28}")
        lines.append("-"*145)

        # Sort by model name
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name]

            puzzles_completed = stats['puzzles_completed']
            pct_correct = f"{stats['pct_moves_correct']:.1f}%"
            avg_cells = f"{stats['avg_cells_correct']:.1f}"
            time_per_move = f"{stats['avg_time_per_move']:.1f}s"
            completed_no_early = stats['puzzles_completed_no_early_stop']

            lines.append(f"{model_name:<35} {puzzles_completed:>19}  {pct_correct:>16}  {avg_cells:>18}  {time_per_move:>14}  {completed_no_early:>28}")

        lines.append("-"*145)

    return "\n".join(lines)


# =============================================================================
# Click Commands
# =============================================================================

@click.group()
def cli():
    pass

@click.command()
def main():
    print("Hello from logic-solver!")

@click.command()
@click.argument('json_path', type=click.Path(exists=True))
def ingest(json_path):
    """Ingest JSON puzzle file and replace hint references."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    cell_data_list = [CellData(**item) for item in data]

    for cell_data in cell_data_list:
        cell_data.hint = replace_name_references(cell_data.hint, cell_data_list)

    print(f"Successfully ingested {len(cell_data_list)} cells from {json_path}")
    print([c.model_dump() for c in cell_data_list])
    return cell_data_list

@click.command()
def fetch():
    """Fetch the latest clues from cluesbysam.com and save to a dated JSON file."""
    try:
        clues_data, filename = fetch_clues_from_website()
        click.echo(f"Successfully fetched and saved clues to {filename}")
    except Exception as e:
        click.echo(f"Error fetching clues: {e}", err=True)
        raise

@click.command()
@click.argument("model", type=click.Choice(AVAILABLE_MODELS), help="The model to generate the puzzle with")
@click.argument("json_path", type=click.Path(exists=True))
@click.option("--rows", "-r", type=int, default=3, help="Number of rows in puzzle grid")
@click.option("--cols", "-c", type=int, default=3, help="Number of columns in puzzle grid")
def generate(rows, cols):
    """Generate a solvable puzzle and save to a dated JSON file."""
    # 1. Generate
    # 2. Validate
    # 3. Cache
    pass

@click.command()
@click.option('--model', type=click.Choice(AVAILABLE_MODELS), help='Name of the model to test')
@click.option('--puzzle', type=str, help='Puzzle to test on (YYYYMMDD date or URL, defaults to today)')
@click.option('--serialization',
              type=click.Choice([e.value for e in SerializationMethod]),
              default=SerializationMethod.DEFAULT.value,
              help='Puzzle serialization method')
@click.option('--preview', is_flag=True, help='Show only the initial puzzle state without running model test')
@click.option('--all-models', is_flag=True, help='Test all available models in parallel')
@click.option('--ppp', type=str, help='Puzzle pack puzzles: comma-separated list of puzzle numbers (e.g., "1,2,3")')
def test(model, puzzle, serialization, preview, all_models, ppp):
    """Test a model on a specific puzzle."""
    # Validate required parameters
    if not preview and not model and not all_models:
        click.echo("Error: --model is required unless using --preview or --all-models", err=True)
        return

    if model and all_models:
        click.echo("Error: Cannot specify both --model and --all-models", err=True)
        return

    if puzzle and ppp:
        click.echo("Error: Cannot specify both --puzzle and --ppp", err=True)
        return

    # Handle --ppp (puzzle pack puzzles)
    if ppp:
        # Parse comma-separated puzzle numbers
        try:
            puzzle_numbers = [int(n.strip()) for n in ppp.split(',')]
        except ValueError:
            click.echo("Error: --ppp must be a comma-separated list of numbers (e.g., '1,2,3')", err=True)
            return

        # Construct puzzle pack URLs
        puzzle_urls = [f"https://cluesbysam.com/s/user/671c185070c51ea6/pack-1/{num}/" for num in puzzle_numbers]

        print(f"Testing {len(puzzle_urls)} puzzle pack puzzle(s): {', '.join(map(str, puzzle_numbers))}")

        # Test each puzzle
        for puzzle_url in puzzle_urls:
            print(f"\n{'='*80}")
            print(f"Testing puzzle: {puzzle_url}")
            print(f"{'='*80}\n")

            # Run the test for this puzzle
            _run_single_test(model, puzzle_url, serialization, preview, all_models)

        print(f"\n{'='*80}")
        print(f"Completed testing {len(puzzle_urls)} puzzle(s)")
        print(f"{'='*80}")
        return

    validated_puzzle = validate_puzzle(puzzle)
    _run_single_test(model, validated_puzzle, serialization, preview, all_models)


def _run_single_test(model, validated_puzzle, serialization, preview, all_models):
    """Run a single test on a puzzle (helper function for test command)."""

    # Step 1: Fetch and/or load puzzle
    print(f"Loading puzzle: {validated_puzzle}")

    # Try to load from cache first
    clues_data, cache_filename = load_puzzle_from_cache(validated_puzzle)

    if clues_data:
        print(f"Loaded puzzle from cache: {cache_filename}")
    else:
        print(f"Puzzle not cached, fetching...")
        clues_data, cache_filename = fetch_and_cache_puzzle(validated_puzzle)
        print(f"Fetched and cached puzzle: {cache_filename}")

    print(f"Puzzle contains {len(clues_data)} clues")

    # Preview mode: just show initial state and exit
    if preview:
        print(f"\n=== PUZZLE PREVIEW ===")
        serialization_method = SerializationMethod(serialization)
        current_state = initialize_puzzle_state(clues_data)
        initial_state = serialize_puzzle_state(clues_data, current_state, serialization_method)
        print(initial_state)
        return

    serialization_method = SerializationMethod(serialization)

    # Handle --all-models flag
    if all_models:
        print(f"\n🚀 Starting parallel tests with all models for puzzle: {validated_puzzle}")
        print(f"Testing {len([m for m in AVAILABLE_MODELS if m != 'human'])} models in parallel...\n")

        try:
            results = run_all_models_parallel(clues_data, validated_puzzle, serialization_method)

            # Save all results
            print("\nSaving results...")
            for result in results:
                results_path = save_test_results(result)
                print(f"  {result.model_name}: {results_path}")

        except Exception as e:
            click.echo(f"Parallel test failed: {e}", err=True)
            raise

        return

    # Validate model for testing
    try:
        ModelFactory.create_model(model)
        if model == 'human':
            print(f"\n🎮 Starting human gameplay mode for puzzle: {validated_puzzle}")
            print("You can type 'quit' at any time to exit the game.")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Step 2: Run the test
    try:
        print(f"\nStarting test with model: {model}")
        test_result = run_model_test(model, clues_data, validated_puzzle, serialization_method)

        # Step 3: Save and print results
        results_path = save_test_results(test_result)
        print_test_evaluation(test_result, results_path)

    except Exception as e:
        click.echo(f"Test failed: {e}", err=True)
        raise

@click.command()
@click.option('--model', type=str, help='Filter stats for specific model(s) - comma-separated list (optional)')
@click.option('--only-same-puzzles', is_flag=True, help='Only analyze puzzles that all models have attempted')
@click.option('--puzzle', type=str, help='Filter stats for a specific puzzle identifier (optional)')
@click.option('--by-difficulty', is_flag=True, help='Group stats by puzzle difficulty (puzzle pack 1 only)')
def stats(model, only_same_puzzles, puzzle, by_difficulty):
    """Display performance statistics from test results."""
    # Check if .test_results exists
    if not os.path.exists('.test_results'):
        click.echo("No test results found. The .test_results/ directory does not exist.", err=True)
        return

    # Scan directories and load test data
    try:
        test_dirs = [d for d in os.listdir('.test_results') if os.path.isdir(os.path.join('.test_results', d))]
    except OSError as e:
        click.echo(f"Error reading .test_results/ directory: {e}", err=True)
        return

    if not test_dirs:
        click.echo("No test results found in .test_results/")
        return

    # Load and group test data by model
    model_tests = defaultdict(list)

    for test_dir in test_dirs:
        metadata, moves = load_test_data(test_dir)

        if metadata is None or moves is None:
            continue

        model_name = metadata.get('model_name', 'unknown')
        model_tests[model_name].append((metadata, moves))

    # Filter by specific model(s) if requested
    if model:
        # Parse comma-separated list of models
        requested_models = [m.strip() for m in model.split(',')]

        # Filter to only requested models
        filtered_tests = {}
        missing_models = []

        for requested_model in requested_models:
            if requested_model in model_tests:
                filtered_tests[requested_model] = model_tests[requested_model]
            else:
                missing_models.append(requested_model)

        # Report missing models
        if missing_models:
            click.echo(f"Warning: No test results found for model(s): {', '.join(missing_models)}", err=True)

        # Check if we have any valid models
        if not filtered_tests:
            click.echo(f"No test results found for any of the requested models")
            return

        model_tests = filtered_tests

    # Filter by specific puzzle if requested
    if puzzle:
        puzzle_found = False
        for model_name in list(model_tests.keys()):
            filtered_tests = [
                (metadata, moves) for metadata, moves in model_tests[model_name]
                if metadata.get('puzzle_identifier') == puzzle
            ]
            if filtered_tests:
                puzzle_found = True
                model_tests[model_name] = filtered_tests
            else:
                # Remove models that don't have this puzzle
                del model_tests[model_name]

        if not puzzle_found:
            click.echo(f"No test results found for puzzle: {puzzle}")
            return

        print(f"Analyzing results for puzzle: {puzzle}\n")

    # Check if any valid tests remain
    if not model_tests:
        click.echo("No valid test results found (all tests have tokens_used=0 or were filtered out)")
        return

    # Filter to only same puzzles if requested
    if only_same_puzzles:
        if len(model_tests) < 2:
            click.echo("Warning: --only-same-puzzles requires at least 2 models, ignoring flag", err=True)
        else:
            # Find puzzles that all models have attempted
            # Create sets of puzzle identifiers for each model
            model_puzzle_sets = {}
            for model_name, tests in model_tests.items():
                puzzle_ids = {metadata.get('puzzle_identifier') for metadata, _ in tests}
                model_puzzle_sets[model_name] = puzzle_ids

            # Find intersection of all puzzle sets (puzzles attempted by ALL models)
            common_puzzles = set.intersection(*model_puzzle_sets.values())

            if not common_puzzles:
                click.echo("No common puzzles found across all models")
                return

            # Filter each model's tests to only include common puzzles
            for model_name in model_tests.keys():
                filtered_tests = [
                    (metadata, moves) for metadata, moves in model_tests[model_name]
                    if metadata.get('puzzle_identifier') in common_puzzles
                ]
                model_tests[model_name] = filtered_tests

            # Print info about filtering
            print(f"Analyzing {len(common_puzzles)} puzzle(s) common to all {len(model_tests)} model(s)")
            print(f"Common puzzles: {', '.join(sorted(common_puzzles))}\n")

    # Handle --by-difficulty flag
    if by_difficulty:
        # Group tests by difficulty
        difficulty_model_tests = defaultdict(lambda: defaultdict(list))
        difficulty_puzzles = defaultdict(set)

        for model_name, tests in model_tests.items():
            for metadata, moves in tests:
                puzzle_id = metadata.get('puzzle_identifier')
                puzzle_num = extract_puzzle_pack_number(puzzle_id)
                difficulty = get_puzzle_difficulty(puzzle_id)

                # Only include puzzle pack puzzles
                if difficulty is not None and puzzle_num is not None:
                    difficulty_model_tests[difficulty][model_name].append((metadata, moves))
                    difficulty_puzzles[difficulty].add(puzzle_num)

        # Check if we have any puzzle pack puzzles
        if not difficulty_model_tests:
            click.echo("No puzzle pack puzzles found in test results. --by-difficulty only works with puzzle pack 1 puzzles.")
            return

        # Calculate stats per difficulty and model
        difficulty_stats = {}
        for difficulty, model_tests_dict in difficulty_model_tests.items():
            difficulty_stats[difficulty] = {}
            for model_name, tests in model_tests_dict.items():
                difficulty_stats[difficulty][model_name] = calculate_model_stats(tests)

        # Print formatted table
        table = format_difficulty_stats_table(difficulty_stats, difficulty_puzzles)
        print(table)

    else:
        # Calculate stats per model (original behavior)
        model_stats = {}
        for model_name, tests in model_tests.items():
            model_stats[model_name] = calculate_model_stats(tests)

        # Print formatted table
        table = format_stats_table(model_stats)
        print(table)

@click.command()
@click.argument('test_id', type=str)
def replay(test_id):
    """Replay a recorded game session from .test_results/{test_id}."""
    import time

    test_dir = os.path.join('.test_results', test_id)

    if not os.path.exists(test_dir):
        click.echo(f"Error: Test result directory '{test_dir}' not found", err=True)
        return

    # Load the recorded data
    try:
        with open(os.path.join(test_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        with open(os.path.join(test_dir, 'moves.json'), 'r') as f:
            moves = json.load(f)

        with open(os.path.join(test_dir, 'conversation.json'), 'r') as f:
            conversation = json.load(f)

        with open(os.path.join(test_dir, 'puzzle.json'), 'r') as f:
            puzzle_data = json.load(f)

    except FileNotFoundError as e:
        click.echo(f"Error: Missing file in test directory: {e}", err=True)
        return
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in test files: {e}", err=True)
        return

    # Print header
    print("=" * 80)
    print(f"🔄 REPLAYING GAME SESSION: {test_id}")
    print("=" * 80)
    print(f"Model: {metadata['model_name']}")
    print(f"Puzzle: {metadata['puzzle_identifier']}")
    print(f"Date: {metadata['test_date']}")
    print(f"Duration: {metadata['duration_seconds']:.1f} seconds")
    print(f"Completed: {'✅' if metadata['completed'] else '❌'}")
    print(f"Total Moves: {metadata['total_moves']}")
    if 'tokens_used' in metadata:
        print(f"Tokens Used: {metadata['tokens_used']:,}")
    if 'cost_usd' in metadata:
        print(f"Cost: ${metadata['cost_usd']:.4f}")
    print("=" * 80)

    # Group conversation by move cycles
    move_index = 0
    conversation_index = 0

    while conversation_index < len(conversation):
        msg = conversation[conversation_index]

        if msg['role'] == 'system':
            print(f"\n🤖 SYSTEM MESSAGE:")
            print("-" * 40)
            print(msg['content'])
            print("-" * 40)
            conversation_index += 1

        elif msg['role'] == 'user':
            # This should be a puzzle state
            print(f"\n📋 PUZZLE STATE (Move {move_index + 1}):")
            print("-" * 40)
            print(msg['content'])
            print("-" * 40)
            conversation_index += 1

            # Look for the corresponding assistant response
            if conversation_index < len(conversation) and conversation[conversation_index]['role'] == 'assistant':
                assistant_msg = conversation[conversation_index]
                print(f"\n🤔 MODEL RESPONSE:")
                print("-" * 40)
                print(assistant_msg['content'])
                print("-" * 40)
                conversation_index += 1

                # Show the corresponding move data if available
                if move_index < len(moves):
                    move_data = moves[move_index]
                    status_emoji = "✅" if move_data['correct'] else "❌"
                    print(f"\n{status_emoji} MOVE RESULT:")
                    print(f"Move: {move_data['move']}")
                    print(f"Correct: {move_data['correct']}")
                    print(f"Feedback: {move_data['feedback']}")
                    print(f"Timestamp: {move_data['timestamp']}")
                    move_index += 1

            # Add a pause for readability
            print("\n" + "=" * 80)
            time.sleep(0.5)  # Brief pause between moves
        else:
            conversation_index += 1

    # Show final summary
    print(f"\n🏁 GAME SUMMARY:")
    print("-" * 40)
    correct_moves = sum(1 for move in moves if move['correct'])
    print(f"Total Moves: {len(moves)}")
    print(f"Correct Moves: {correct_moves}")
    print(f"Accuracy: {correct_moves/len(moves)*100:.1f}%" if moves else "N/A")
    print(f"Game Completed: {'Yes ✅' if metadata['completed'] else 'No ❌'}")

    if not metadata['completed']:
        if metadata.get('max_moves_reached', False):
            print("⚠️  Game ended due to max moves limit")
        else:
            print("⚠️  Game ended early (likely due to error or interruption)")


# =============================================================================
# Z3 Code (for constraint solving)
# =============================================================================

class Cell:

    def __init__(
            self,
            row: int,
            col: int,
            name: str,
            profession: str,
            raw_clue: str | None = None,
            clue_constraints = None
    ):
        self.row = row
        self.col = col
        self.name = name
        self.profession = profession
        self.raw_clue = raw_clue
        self.clue_constraints = clue_constraints
        self.is_criminal: BoolRef = Bool(f"{row}_{col}_{profession}")

    def apply_constraints(self, grid: "Grid"):
        if self.clue_constraints is None:
            return

        print(f"Adding constraints {self.clue_constraints}")
        grid.solver.add(self.clue_constraints(grid))

    def display(self, grid: "Grid"):
        print(self.name)
        print(self.profession)
        print('-')
        print(self.raw_clue)



class Grid:
    def __init__(self, rows: list[list[Cell]]):
        self.rows = rows
        self.solver = Solver()
        self._name_to_index = {}
        self._profession_to_indexes = defaultdict(list)

        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                self._name_to_index[cell.name] = (i, j)
                self._profession_to_indexes[cell.profession].append((i, j))
                cell.apply_constraints(self)

    @property
    def variables(self) -> list[BoolRef]:
        return [c.is_criminal for row in self.rows for c in row]

    @property
    def cells(self) -> list[Cell]:
        return [c for row in self.rows for c in row]

    def name_to_index(self, name: str) -> tuple[int, int]:
        return self._name_to_index[name]

    def check(self):
        return self.solver.check()

    def add_clue(self, clue: ExprRef):
        self.solver.add(clue)

    def set_is_criminal(self, name, is_criminal: bool):
        row, col = self.name_to_index(name)
        print(f"Setting {name} ({row},{col}) as {'criminal' if is_criminal else 'innocent'}")
        self.solver.add(self.rows[row][col].is_criminal == is_criminal)

    def display(self):
        if self.solver.check() != sat:
            print("Unsatisfiable grid")
            return
        model = self.solver.model()
        for row in self.rows:
            print("\n " + "- - - -")
            print("|", sep="", end="")
            for cell in row:
                is_criminal = model.eval(cell.is_criminal).py_value()
                if is_criminal is None:
                    label = "?"
                elif is_criminal:
                    label = "c"
                else:
                    label = "i"
                print(f"{label}|", sep="", end="")
            print("\n " + "- - - -")




def odd_number_left_true(rows: list[list[Cell]], index: tuple[int, int]) -> ExprRef:
    cells = [rows[index[0]][i] for i in range(index[1])]

    true_count = Sum([If(b.is_criminal, 1, 0) for b in cells])

    # Check if count is odd
    return true_count % 2 == 1


def create_grid():
    return [Bool(f"cell_{i}") for i in range(4)]


def example():
    adam = Cell(0,0, "Adam", "singer",
             raw_clue='There\'s an odd number of criminals to the left of Cheryl',
        )
    rows = [[
        adam,
        Cell(0, 1, "Betty", "teacher"),
        Cell(0, 2, "Cheryl", "farmer"),
        Cell(0, 3, "Evie", "doctor"),
    ]]
    grid = Grid(rows)
    grid.add_clue(adam.is_criminal == True)
    grid.solver.add(odd_number_left_true(rows, (0, 2)))

    determined = find_determined_variables(grid)

    if len(determined) == 0:
        print("Unable to determine any further cells given constraints")
        return

    for name, is_criminal in determined.items():
        print(f"{name} is {'a criminal' if is_criminal else 'innocent'}")
        grid.set_is_criminal(name, is_criminal)


    grid.display()

    return grid

def clue_grid():
    return {
        "Adam": "",
        "Cheryl": "",
    }

def find_determined_variables(grid: Grid):
    solver = grid.solver
    if solver.check() != sat:
        return {}  # No solution exists

    determined = {}

    for cell in grid.cells:
        possible_values = [True, False]
        can_be_values = []

        for value in possible_values:
            solver.push()  # Save state
            solver.add(cell.is_criminal == value)

            if solver.check() == sat:
                can_be_values.append(value)

            solver.pop()  # Restore state

        # If only one value is possible, the variable is determined
        if len(can_be_values) == 1:
            determined[cell.name] = can_be_values[0]
        elif len(can_be_values) == 0:
            print(f"Warning: Variable {cell.name} has no valid values (contradictory constraints)")

    return determined


# =============================================================================
# Command Registration
# =============================================================================

cli.add_command(main)
cli.add_command(ingest)
cli.add_command(fetch)
cli.add_command(test)
cli.add_command(stats)
cli.add_command(replay)
cli.add_command(generate)

if __name__ == "__main__":
    cli()
