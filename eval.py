"""
eval.py - Model evaluation framework for logic puzzle solving

This module provides the core evaluation infrastructure for testing AI models
on logic puzzle solving tasks. It includes:
- Model factory for creating different LLM instances
- Game chat memory for conversation tracking
- Model testing framework with progress tracking
- Puzzle state management and serialization
- Move validation and game state updates
- Parallel model testing capabilities
- Results management and analysis
"""

import json
import os
import re
import time
import hashlib
import threading
from typing import List, Tuple, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_xai import ChatXAI
from langchain.memory import ChatMessageHistory

from models import (
    CellData,
    PuzzleCell,
    PuzzleState,
    Status,
    SerializationMethod,
    TestResult,
    ModelProgress,
    ModelCommunicationError,
    GameStateError
)


# Grid dimensions
ROWS = 5
COLS = 4

# Available models for testing
AVAILABLE_MODELS = [
    "gpt-5",
    # "gpt-5-mini",
    # "gpt-5-nano",
    # "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    # "gemini-2.5-flash",
    # "gemini-2.5-flash-lite",
    "deepseek-chat",
    # "deepseek-reasoner",
    # "grok-4"
]

# Header for parallel model testing
PARALLEL_TEST_HEADER = "\n\nReady. Set. Deduce! 🏎️💨"


class ModelFactory:
    @staticmethod
    def create_model(model_name: str):
        """Create appropriate model client based on model name."""
        if model_name == 'human':
            return None  # Human mode doesn't need a model
        elif model_name.startswith('gpt'):
            return ChatOpenAI(model_name=model_name, temperature=0)
        elif model_name.startswith('claude'):
            return ChatAnthropic(model_name=model_name, temperature=0, max_tokens=4096)
        elif model_name.startswith("gemini"):
            return ChatGoogleGenerativeAI(model=model_name, temperature=0)
        elif model_name.startswith("deepseek"):
            return ChatDeepSeek(model=model_name, temperature=0)
        elif model_name.startswith("grok"):
            return ChatXAI(model=model_name, temperature=0)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def get_model_config(model_name: str) -> dict:
        """Get model-specific configuration."""
        return {
            "temperature": 0,
            "max_tokens": 8192,
            "timeout": 30
        }


class GameChatMemory:
    def __init__(self):
        self.message_history = ChatMessageHistory()
        self.conversation_log = []

    def add_system_message(self, content: str):
        msg = SystemMessage(
            content=content,
            additional_kwargs={
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        )
        self.message_history.add_message(msg)
        self._log_message("system", content)

    def add_user_message(self, content: str):
        msg = HumanMessage(content=content)
        self.message_history.add_message(msg)
        self._log_message("user", content)

    def add_ai_message(self, content: str):
        msg = AIMessage(content=content)
        self.message_history.add_message(msg)
        self._log_message("assistant", content)

    def _log_message(self, role: str, content: str):
        self.conversation_log.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_messages(self):
        return self.message_history.messages

    def get_conversation_log(self):
        return self.conversation_log


class ModelTester:
    def __init__(self, model_name: str, quiet_mode: bool = False):
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name)
        self.memory = GameChatMemory()
        self.config = ModelFactory.get_model_config(model_name)
        self.total_tokens = 0
        self.total_cost = 0.0
        self.is_human = (model_name == 'human')
        self.quiet_mode = quiet_mode

    def log(self, *args, **kwargs):
        """Log to stdout only if not in quiet mode."""
        if not self.quiet_mode:
            print(*args, **kwargs)

    def send_system_message(self, content: str):
        """Send initial system message with game rules."""
        if self.is_human:
            print("\n" + "="*50)
            print("GAME RULES:")
            print("="*50)
            print(content)
            print("="*50)
        else:
            # Still log it for conversation history
            self.memory.add_system_message(content)

    def send_message_and_get_response(self, content: str, max_retries: int = 3) -> str:
        """Send message with full context and get model response."""
        if self.is_human:
            return self._get_human_response(content)
        else:
            return self._get_ai_response(content, max_retries)

    def _get_human_response(self, content: str) -> str:
        """Get response from human player."""
        print("\n" + "-"*50)
        print("CURRENT GAME STATE:")
        print("-"*50)
        print(content)
        print("-"*50)

        while True:
            try:
                response = input("\nEnter your move (format: 'MOVE: [name] is [innocent|criminal]') or 'quit' to exit: ").strip()

                if response.lower() == 'quit':
                    raise KeyboardInterrupt("Game quit by user")

                # Validate basic format
                if response.upper().startswith('MOVE:'):
                    # Log the human interaction
                    self.memory.add_user_message(content)
                    self.memory.add_ai_message(response)
                    return response
                else:
                    print("Invalid format. Please use: MOVE: [name] is [innocent|criminal]")

            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise
            except EOFError:
                print("\nInput stream ended.")
                raise KeyboardInterrupt("Input stream ended")

    def _get_ai_response(self, content: str, max_retries: int = 3) -> str:
        """Get response from AI model."""
        # Log user message to stdout
        self.log("\n" + "="*80)
        self.log("📤 USER MESSAGE:")
        self.log("="*80)
        self.log(content)
        self.log("="*80)

        self.memory.add_user_message(content)

        for attempt in range(max_retries):
            try:
                response = self.model.invoke(self.memory.get_messages())
                response_content = response.content

                # Log AI response to stdout
                self.log(f"\n🤖 {self.model_name.upper()} RESPONSE:")
                self.log("-"*80)
                self.log(response_content)
                self.log("-"*80)

                # Track token usage and cost
                self._update_usage_metrics(response)

                # Log to conversation history (for replay)
                self.memory.add_ai_message(response_content)
                return response_content

            except Exception as e:
                if attempt == max_retries - 1:
                    error_msg = f"Model communication failed after {max_retries} attempts: {e}"
                    print(error_msg)
                    self.memory.add_ai_message(error_msg)
                    raise ModelCommunicationError(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff

    def _update_usage_metrics(self, response):
        """Update token usage and cost metrics from response."""
        # Try to extract token usage from response metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            metadata = response.response_metadata

            # For OpenAI models
            if 'token_usage' in metadata:
                usage = metadata['token_usage']
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', input_tokens + output_tokens)

                self.total_tokens += total_tokens
                self.total_cost += self._calculate_cost_openai(input_tokens, output_tokens)

            # For Anthropic models
            elif 'usage' in metadata:
                input_tokens = metadata['usage'].get('input_tokens', 0)
                output_tokens = metadata['usage'].get('output_tokens', 0)
                total_tokens = input_tokens + output_tokens

                self.total_tokens += total_tokens
                self.total_cost += self._calculate_cost_anthropic(input_tokens, output_tokens)


    def _calculate_cost_openai(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI models based on input/output tokens."""
        # Pricing per 1M tokens (as of latest update)
        model_pricing = {
            'gpt-5': {
                'input': 1.250,  # $1.250 / 1M tokens
                'output': 10.000  # $10.000 / 1M tokens
            },
            'gpt-5-mini': {
                'input': 0.250,  # $0.250 / 1M tokens
                'output': 2.000   # $2.000 / 1M tokens
            },
            'gpt-5-nano': {
                'input': 0.050,  # $0.050 / 1M tokens
                'output': 0.400   # $0.400 / 1M tokens
            },
            'deepseek-chat': {
                'input': 0.28,
                'output': 0.42
            }
        }

        # Find matching model
        for model_key, pricing in model_pricing.items():
            if model_key in self.model_name.lower():
                input_cost = (input_tokens / 1000000) * pricing['input']
                output_cost = (output_tokens / 1000000) * pricing['output']
                return input_cost + output_cost

        # Default cost if model not found (using gpt-5-mini rates as fallback)
        input_cost = (input_tokens / 1000000) * 0.250
        output_cost = (output_tokens / 1000000) * 2.000
        return input_cost + output_cost

    def _calculate_cost_anthropic(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic models."""
        # Approximate costs (per 1k tokens) - update as needed
        input_cost_per_1k = 0.003  # Claude-3 input cost
        output_cost_per_1k = 0.015  # Claude-3 output cost

        return ((input_tokens / 1000) * input_cost_per_1k +
                (output_tokens / 1000) * output_cost_per_1k)


    def get_conversation_history(self):
        return self.memory.get_conversation_log()

    def get_usage_metrics(self) -> Tuple[int, float]:
        """Get total tokens used and cost."""
        return self.total_tokens, self.total_cost


def translate_hint_keywords(name: str, hint: str, cells: List[PuzzleCell]) -> str:
    """Translate hint keywords to plain English following JS parsing rules."""
    if not hint or hint == "NO HINT":
        return hint

    s = hint
    flat_cells = [cell for row in cells for cell in row]

    # Find current cell index (equivalent to 'a' in JS)
    current_cell_index = None
    for i, cell in enumerate(flat_cells):
        if cell.name == name:
            current_cell_index = i
            break

    # Column names (Kd array equivalent)
    column_names = ["A", "B", "C", "D"]

    # 1. Replace #C:([0-9]) with column names
    def replace_column(match):
        col_num = int(match.group(1))
        return column_names[col_num - 1] if 1 <= col_num <= len(column_names) else match.group(0)
    s = re.sub(r'#C:([0-9])', replace_column, s)

    # 2. Replace #NAMES:([0-9]+) with possessive forms
    def replace_names_possessive(match):
        index = int(match.group(1))
        if index >= len(flat_cells):
            return match.group(0)
        cell_name = flat_cells[index].name
        if current_cell_index == index:
            return "my"
        return f"#NAME:{index}'s" if not cell_name.endswith('s') else f"#NAME:{index}'"
    s = re.sub(r'#NAMES:([0-9]+)', replace_names_possessive, s)

    # 3. Replace "#NAME:([0-9]+) and #NAME:([0-9]+)" patterns
    def replace_name_and_name(match):
        index1, index2 = int(match.group(1)), int(match.group(2))
        if current_cell_index == index1:
            return f"#NAME:{index2} and I"
        elif current_cell_index == index2:
            return f"#NAME:{index1} and I"
        return match.group(0)
    s = re.sub(r'#NAME:([0-9]+) and #NAME:([0-9]+)', replace_name_and_name, s)

    # 4. Replace "^#NAME:([0-9]+) (is|has)" at start of string
    def replace_name_is_has(match):
        index = int(match.group(1))
        verb = match.group(2)
        if current_cell_index == index:
            if verb == "is":
                return "I am"
            elif verb == "has":
                return "I have"
            else:
                return f"I {verb}"
        return f"#NAME:{index} {verb}"
    s = re.sub(r'^#NAME:([0-9]+) (is|has)', replace_name_is_has, s)

    # 5. Replace #BETWEEN:pair\(([0-9]+),([0-9]+)\) with complex logic
    def replace_between(match):
        is_prefix = match.group(1) or ""  # Captures "is " prefix if present
        index1, index2 = int(match.group(2)), int(match.group(3))

        # Ensure B <= Z (smaller index first)
        B = min(index1, index2)
        Z = max(index1, index2)

        # Calculate positions
        K = B % COLS  # B column
        F = B // COLS  # B row
        ae = Z % COLS  # Z column
        P = Z // COLS  # Z row

        de = (F == P)  # same row
        ve = B - (1 if de else COLS)  # previous cell
        k = Z + (1 if de else COLS)   # next cell

        if de:  # same row
            if K == 0 and ae == COLS - 1:
                return f"{is_prefix}in row {F + 1}"
            if K == 0:
                ref_name = "me" if current_cell_index == k else f"#NAME:{k}"
                return f"{'is ' if is_prefix else ''}to the left of {ref_name}"
            if ae == COLS - 1:
                ref_name = "me" if current_cell_index == ve else f"#NAME:{ve}"
                return f"{'is ' if is_prefix else ''}to the right of {ref_name}"
        else:  # same column
            if F == 0 and P == ROWS - 1:
                return f"{is_prefix}in column {column_names[K]}"
            if F == 0:
                ref_name = "me" if current_cell_index == k else f"#NAME:{k}"
                return f"{'is ' if is_prefix else ''}above {ref_name}"
            if P == ROWS - 1:
                ref_name = "me" if current_cell_index == ve else f"#NAME:{ve}"
                return f"{'is ' if is_prefix else ''}below {ref_name}"

        # General case
        if current_cell_index == ve:
            return f"{'is ' if is_prefix else ''}in between #NAME:{k} and me"
        elif current_cell_index == k:
            return f"{'is ' if is_prefix else ''}in between #NAME:{ve} and me"
        else:
            return f"{'is ' if is_prefix else ''}in between #NAME:{ve} and #NAME:{k}"

    s = re.sub(r'(is )?#BETWEEN:pair\(([0-9]+),([0-9]+)\)', replace_between, s)

    # 6. Replace #PROF(S?):([a-z]+)
    def replace_profession(match):
        plural = match.group(1) or ""  # 'S' if plural
        profession = match.group(2)
        return f"{profession}{'s' if plural else ''}"
    s = re.sub(r'#PROF(S?):([a-z]+)', replace_profession, s)

    # 7. Replace various #NAME:([0-9]+) patterns
    # First handle "neighboring #NAME:([0-9]+)"
    def replace_neighboring_name(match):
        index = int(match.group(1))
        return "neighboring me" if current_cell_index == index else match.group(0)
    s = re.sub(r'neighboring #NAME:([0-9]+)', replace_neighboring_name, s)

    # Then handle "^#NAME:([0-9]+)" at start of string
    def replace_name_start(match):
        index = int(match.group(1))
        return "I" if current_cell_index == index else match.group(0)
    s = re.sub(r'^#NAME:([0-9]+)', replace_name_start, s)

    # Then handle all other "#NAME:([0-9]+)" -> "me" if current cell
    def replace_name_me(match):
        index = int(match.group(1))
        return "me" if current_cell_index == index else match.group(0)
    s = re.sub(r'#NAME:([0-9]+)', replace_name_me, s)

    # Finally replace remaining "#NAME:([0-9]+)" with actual names
    def replace_name_final(match):
        index = int(match.group(1))
        if index >= len(flat_cells):
            return match.group(0)
        cell_name = flat_cells[index].name
        return cell_name.capitalize()
    s = re.sub(r'#NAME:([0-9]+)', replace_name_final, s)

    # 8. Final cleanup
    s = s.replace(" exactly 0 ", " no ")
    s = s[0].upper() + s[1:] if len(s) > 0 else s

    return s


def validate_hint_keywords(hints: List[str]) -> None:
    """Validate that all hint keywords are recognized. Exit if unknown keywords found."""
    # Known keyword patterns
    known_patterns = [
        r'#C:\d+',           # Column references
        r'#R:\d+',           # Row references
        r'#NAMES?:\d+',      # Name references (singular and plural)
        r'#PROFS?:\w+',      # Profession references (singular and plural)
        r'#BETWEEN:pair\(\d+,\d+\)'  # Between references
    ]

    unknown_keywords = set()

    for hint in hints:
        if not hint or hint == "NO HINT":
            continue

        # Find all potential keywords (#WORD patterns)
        keywords = re.findall(r'#[A-Z][A-Z0-9_]*(?::[^\s\]]+)?', hint)

        for keyword in keywords:
            # Check if this keyword matches any known pattern
            is_known = False
            for pattern in known_patterns:
                if re.match(pattern, keyword):
                    is_known = True
                    break

            if not is_known:
                unknown_keywords.add(keyword)

    if unknown_keywords:
        print("ERROR: Unknown hint keywords found:")
        for keyword in sorted(unknown_keywords):
            print(f"  - {keyword}")
        print("\nPlease add translations for these keywords before proceeding.")
        exit(1)


def replace_name_references(hint: str, cells: List[CellData]) -> str:
    def replace_match(match):
        index = int(match.group(1))
        return cells[index].name.capitalize()

    return re.sub(r'#NAME:(\d+)', replace_match, hint)


def serialize_puzzle_state(clues_data: List[dict], current_state: PuzzleState, method: SerializationMethod) -> str:
    """Serialize puzzle state for model consumption."""
    if method == SerializationMethod.DEFAULT:
        # Count unsolved cells
        unsolved_count = 0
        solved_count = 0

        grid_str = "PUZZLE GRID (5 rows x 4 columns):\n"
        solved_cells = []

        for row_idx, row in enumerate(current_state):
            row_str = f"Row {row_idx + 1}: "
            for col_idx, cell in enumerate(row):
                if cell.status == Status.UNKNOWN:
                    status_str = "?"
                    unsolved_count += 1
                elif cell.status == Status.CRIMINAL:
                    status_str = "C"
                    solved_count += 1
                    solved_cells.append(cell)
                else:  # INNOCENT
                    status_str = "I"
                    solved_count += 1
                    solved_cells.append(cell)

                row_str += f"[{cell.name.capitalize()}({cell.profession}):{status_str}] "
            grid_str += row_str + "\n"

        grid_str += f"\nSolved: {solved_count}/20, Remaining: {unsolved_count}"

        # Add visible hints for solved cells
        if solved_cells:
            grid_str += "\n\nVISIBLE HINTS:"
            for cell in solved_cells:
                status_name = "CRIMINAL" if cell.status == Status.CRIMINAL else "INNOCENT"
                translated_hint = translate_hint_keywords(cell.name, cell.clue, current_state)
                grid_str += f"\n- {cell.name} ({status_name}): {translated_hint}"
        return grid_str


def initialize_puzzle_state(clues_data: List[dict]) -> List[List[PuzzleCell]]:
    """Initialize puzzle state as 2D list of PuzzleCell objects."""
    # Validate hint keywords before creating puzzle state
    hints = [person.get('hint', '') for person in clues_data]
    validate_hint_keywords(hints)

    # Create 5x4 grid
    grid = []

    for row in range(ROWS):
        grid_row = []
        for col in range(COLS):
            # Calculate index in flattened list (row-major order)
            idx = row * COLS + col

            if idx < len(clues_data):

                person_data = clues_data[idx]
                is_criminal = person_data['criminal']
                if len(person_data['paths']) == 0:
                    status = Status.CRIMINAL if is_criminal else Status.INNOCENT
                else:
                    status = Status.UNKNOWN
                cell = PuzzleCell(
                    name=person_data['name'],
                    profession=person_data['profession'],
                    gender=person_data['gender'],
                    orig_hint=person_data.get('orig_hint', "NO ORIG_HINT"),
                    clue=person_data.get("hint", "NO HINT"),
                    status=status,
                    had_mistake=False,
                    paths=person_data.get('paths', []),
                    is_criminal=person_data.get('criminal', False)
                )
            else:
                raise ValueError(f"Should not happen: index={idx}")

            grid_row.append(cell)
        grid.append(grid_row)

    return grid


def extract_move_from_response(response: str) -> str:
    """Extract move from model response."""
    pattern = r"MOVE:\s*(\w+)\s+is\s+(innocent|criminal)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return f"{match.group(1)} is {match.group(2)}"

    raise ValueError(f"Could not extract move from response: {response}")


def validate_move(move: str, current_state: List[List[PuzzleCell]], clues_data: List[dict]) -> Tuple[bool, str]:
    """Validate if a move is logically deducible and correct."""
    # Parse the move
    parts = move.split(' is ')
    if len(parts) != 2:
        return False, "Invalid move format"

    name, classification = parts
    classification = classification.lower()

    if classification not in ['innocent', 'criminal']:
        return False, "Classification must be 'innocent' or 'criminal'"

    # Find the target cell in the grid
    target_cell = None
    for row in current_state:
        for cell in row:
            if cell.name.lower() == name.lower():
                target_cell = cell
                break
        if target_cell:
            break

    if not target_cell:
        return False, f"Person '{name}' not found in puzzle"

    if target_cell.status != Status.UNKNOWN:
        return False, f"{name} is already classified"

    # Check if the move is logically deducible based on paths
    flat_cells = [cell for row in current_state for cell in row]

    # Check if any path makes this move deducible
    is_deducible = False
    for path in target_cell.paths:
        # A path makes the move deducible if all cells in the path are already identified
        path_complete = True
        for index in path:
            if 0 <= index < len(flat_cells):
                if flat_cells[index].status == Status.UNKNOWN:
                    path_complete = False
                    break
            else:
                path_complete = False
                break

        if path_complete:
            is_deducible = True
            break

    # If not logically deducible, mark as mistake and reject
    if not is_deducible:
        target_cell.had_mistake = True
        return False, f"Move cannot be logically deduced from the currently available information. You need to solve other people first to unlock the logical path for {name}."

    # Move is deducible, now check if it's correct
    expected_criminal = target_cell.is_criminal
    move_says_criminal = (classification == 'criminal')

    if move_says_criminal == expected_criminal:
        return True, "Correct move!"
    else:
        target_cell.had_mistake = True
        expected_status = "criminal" if expected_criminal else "innocent"
        return False, f"Incorrect! {name} is actually {expected_status}, not {classification}."


def update_puzzle_state(current_state: List[List[PuzzleCell]], move: str) -> List[List[PuzzleCell]]:
    """Update puzzle state after a correct move."""
    parts = move.split(' is ')
    if len(parts) == 2:
        name, classification = parts
        classification = classification.lower()

        # Find and update the cell
        for row in current_state:
            for cell in row:
                if cell.name.lower() == name.lower():
                    if classification == 'criminal':
                        cell.status = Status.CRIMINAL
                    else:  # innocent
                        cell.status = Status.INNOCENT
                    break

    return current_state


def is_puzzle_complete(current_state: List[List[PuzzleCell]]) -> bool:
    """Check if puzzle is completely solved."""
    for row in current_state:
        for cell in row:
            if cell.status == Status.UNKNOWN:
                return False
    return True


def get_system_prompt() -> str:
    """Get initial system prompt with game rules."""
    return """
GAME RULES:
- You have a grid of 20 people (5 rows of 4)
- Each person is either innocent or criminal
- Each person has a name, profession, and maybe a clue (not all clues are helpful. Some are just banter).
- The clue is only visible once you have correctly marked the person as criminal or innocent.
- Your goal is to determine who is innocent and who is criminal using only logic.
- There is always a logical choice, even if you think there isn't.
- You can only make one move at a time

TERMINOLOGY:
-  "Neighbors" always include diagonal neighbors. One person can have up to 8 neighbors.
- In between (or sometimes just between) means the persons between the two, not including the two.
- Connected means a chain of orthogonal adjacency. For example "all criminals in row 1 are connected" means there are no innocents between any two criminals in that row.
- To the left/right always means somewhere in the same row. Above/below always means somewhere in the same column.
- Directly to the left/right/above/below always means the neighbor to the left/right/above/below.
- Common neighbors means those who are neighbors of both persons. It does not include the persons themselves.
- Share means "have in common". For example, "Share an odd number of innocent neighbors" means "have an odd number of common innocent neighbors".
- Edge means the 14 persons "surrounding" the board, including corners.

RESPONSE FORMAT:
Your response must contain exactly one move in this format:
MOVE: [person_name] is [innocent|criminal]

Example: "MOVE: Alice is innocent"

IMPORTANT:
- Only make one move per response
- Base your reasoning on the clues provided
- Explain your reasoning before stating your move
- If you're unsure, make your best guess

STRATEGY:
- For each hint given, identify the information that you can definitively verify.
Then, try to identify if any of the pieces of definitive information overlap in such a way
that transitively prove the state of one of the unknown cells.

Ready to start? I'll give you the current puzzle state.
"""


def get_test_results_dir(puzzle_identifier: str, model_name: str) -> str:
    """Get test results directory path."""
    if puzzle_identifier.startswith(('http://', 'https://')):
        identifier = hashlib.md5(puzzle_identifier.encode()).hexdigest()
    else:
        identifier = puzzle_identifier

    return os.path.join(".test_results", f"{identifier}-{model_name}")


def save_test_results(test_result: TestResult) -> str:
    """Save test results to directory structure."""
    results_dir = get_test_results_dir(test_result.puzzle_identifier, test_result.model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save conversation.json
    conversation_path = os.path.join(results_dir, "conversation.json")
    with open(conversation_path, 'w') as f:
        json.dump(test_result.conversation, f, indent=2)

    # Save moves.json
    moves_path = os.path.join(results_dir, "moves.json")
    with open(moves_path, 'w') as f:
        json.dump(test_result.moves, f, indent=2)

    # Save metadata.json
    metadata = {
        "model_name": test_result.model_name,
        "puzzle_identifier": test_result.puzzle_identifier,
        "test_date": test_result.test_date,
        "duration_seconds": test_result.duration_seconds,
        "completed": test_result.completed,
        "total_moves": test_result.total_moves,
        "max_moves_reached": test_result.max_moves_reached,
        "tokens_used": test_result.tokens_used,
        "cost_usd": test_result.cost_usd
    }
    metadata_path = os.path.join(results_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save puzzle.json
    puzzle_path = os.path.join(results_dir, "puzzle.json")
    with open(puzzle_path, 'w') as f:
        json.dump(test_result.puzzle_data, f, indent=2)

    return results_dir


def print_test_evaluation(test_result: TestResult, results_path: str):
    """Print test evaluation summary."""
    print(f"\n=== TEST RESULTS ===")
    print(f"Model: {test_result.model_name}")
    print(f"Puzzle: {test_result.puzzle_identifier}")
    print(f"Completed: {'✓' if test_result.completed else '✗'}")
    print(f"Total Moves: {test_result.total_moves}")
    print(f"Duration: {test_result.duration_seconds:.2f} seconds")
    print(f"Tokens Used: {test_result.tokens_used:,}")
    print(f"Cost: ${test_result.cost_usd:.4f}")
    print(f"Results saved to: {results_path}")

    if test_result.max_moves_reached:
        print("⚠ Warning: Maximum moves (100) reached without completion")


def run_model_test(
    model_name: str,
    clues_data: List[dict],
    puzzle_identifier: str,
    serialization_method: SerializationMethod
) -> TestResult:
    """Run the complete model test with LangChain."""

    start_time = time.time()
    moves = []
    current_state: List[List[PuzzleCell]] = initialize_puzzle_state(clues_data)
    max_moves = 40

    # Initialize model tester
    model_tester = ModelTester(model_name)

    # Send system message with game rules
    system_prompt = get_system_prompt()
    model_tester.send_system_message(system_prompt)

    # Main game loop
    move_count = 1
    consecutive_mistakes = 0
    while move_count < max_moves + 1:
        try:
            print(f"\n{'='*80}")
            print(f"🎯 MOVE {move_count} (Model: {model_name})")
            print(f"{'='*80}")

            # Always send current puzzle state (feedback is handled internally)
            state_message = serialize_puzzle_state(clues_data, current_state, serialization_method)

            # Send state to model and get response
            model_response = model_tester.send_message_and_get_response(state_message)

            # Extract and validate move
            move = extract_move_from_response(model_response)
            is_correct, feedback = validate_move(move, current_state, clues_data)

            # Record move
            moves.append({
                "move_number": move_count,
                "move": move,
                "correct": is_correct,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })

            if is_correct:
                print("✅ Correct")
                # Add success feedback to conversation
                model_tester.memory.add_user_message(f"✅ CORRECT! {feedback}")
                current_state = update_puzzle_state(current_state, move)
                consecutive_mistakes = 0  # Reset consecutive mistakes counter

                if is_puzzle_complete(current_state):
                    print(f"\n✅ PUZZLE COMPLETED!")
                    print(f"All moves were correct. Congratulations!")
                    break
            else:
                print("❌ Incorrect")
                # Add error feedback to conversation
                model_tester.memory.add_user_message(f"❌ INCORRECT! {feedback}")
                # Track consecutive mistakes
                consecutive_mistakes += 1

                # Check for early stop condition
                if consecutive_mistakes >= 5:
                    print(f"\n❌ EARLY STOP: 5 consecutive incorrect moves")
                    print(f"Game ended early due to repeated mistakes")
                    break

            move_count += 1

        except KeyboardInterrupt:
            print(f"\nGame interrupted by user after {move_count} moves.")
            break
        except (ModelCommunicationError, ValueError, GameStateError) as e:
            moves.append({
                "move_number": move_count,
                "move": "ERROR",
                "correct": False,
                "feedback": str(e),
                "timestamp": datetime.now().isoformat()
            })
            break

    # Calculate results
    duration = time.time() - start_time
    completed = is_puzzle_complete(current_state)
    max_moves_reached = move_count >= max_moves + 1
    tokens_used, cost_usd = model_tester.get_usage_metrics()

    return TestResult(
        puzzle_data=clues_data,
        model_name=model_name,
        puzzle_identifier=puzzle_identifier,
        test_date=datetime.now().isoformat(),
        duration_seconds=duration,
        conversation=model_tester.get_conversation_history(),
        moves=moves,
        completed=completed,
        total_moves=move_count,
        max_moves_reached=max_moves_reached,
        tokens_used=tokens_used,
        cost_usd=cost_usd
    )


def run_model_test_with_progress(
    model_name: str,
    clues_data: List[dict],
    puzzle_identifier: str,
    serialization_method: SerializationMethod,
    progress: ModelProgress
) -> TestResult:
    """Run model test and update progress tracker."""
    try:
        start_time = time.time()
        moves = []
        current_state: List[List[PuzzleCell]] = initialize_puzzle_state(clues_data)
        max_moves = 40

        # Initialize model tester in quiet mode for parallel execution
        model_tester = ModelTester(model_name, quiet_mode=True)

        # Send system message with game rules
        system_prompt = get_system_prompt()
        model_tester.send_system_message(system_prompt)

        # Main game loop
        move_count = 1
        consecutive_mistakes = 0
        while move_count < max_moves + 1:
            try:
                # Update progress
                progress.update(move_count)

                # Always send current puzzle state
                state_message = serialize_puzzle_state(clues_data, current_state, serialization_method)

                # Send state to model and get response
                model_response = model_tester.send_message_and_get_response(state_message)

                # Extract and validate move
                move = extract_move_from_response(model_response)
                is_correct, feedback = validate_move(move, current_state, clues_data)

                # Update progress with result
                progress.update(move_count, is_correct)

                # Record move
                moves.append({
                    "move_number": move_count,
                    "move": move,
                    "correct": is_correct,
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat()
                })

                if is_correct:
                    model_tester.memory.add_user_message(f"✅ CORRECT! {feedback}")
                    current_state = update_puzzle_state(current_state, move)
                    consecutive_mistakes = 0

                    if is_puzzle_complete(current_state):
                        progress.mark_completed(True)
                        break
                else:
                    model_tester.memory.add_user_message(f"❌ INCORRECT! {feedback}")
                    consecutive_mistakes += 1

                    if consecutive_mistakes >= 5:
                        progress.mark_completed(False)
                        break

                move_count += 1

            except KeyboardInterrupt:
                progress.mark_error("Interrupted by user")
                break
            except (ModelCommunicationError, ValueError, GameStateError) as e:
                moves.append({
                    "move_number": move_count,
                    "move": "ERROR",
                    "correct": False,
                    "feedback": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                progress.mark_error(str(e))
                break

        # Calculate results
        duration = time.time() - start_time
        completed = is_puzzle_complete(current_state)
        max_moves_reached = move_count >= max_moves + 1
        tokens_used, cost_usd = model_tester.get_usage_metrics()

        if not progress.get_status()['completed']:
            progress.mark_completed(completed)

        return TestResult(
            puzzle_data=clues_data,
            model_name=model_name,
            puzzle_identifier=puzzle_identifier,
            test_date=datetime.now().isoformat(),
            duration_seconds=duration,
            conversation=model_tester.get_conversation_history(),
            moves=moves,
            completed=completed,
            total_moves=move_count,
            max_moves_reached=max_moves_reached,
            tokens_used=tokens_used,
            cost_usd=cost_usd
        )
    except Exception as e:
        progress.mark_error(str(e))
        raise


def display_parallel_progress(progress_trackers: Dict[str, ModelProgress], stop_event: threading.Event):
    """Display real-time progress of all models."""
    while not stop_event.is_set():
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")

        # Display ASCII art header with soft red background
        print(PARALLEL_TEST_HEADER)
        print()  # Blank line for spacing

        print("="*100)
        print(f"{'Model':<30} {'Move':<10} {'Correct':<10} {'Incorrect':<10} {'Status':<20}")
        print("="*100)

        for model_name in sorted(progress_trackers.keys()):
            status = progress_trackers[model_name].get_status()

            # Format status text
            if status['error']:
                status_text = f"ERROR: {status['error'][:15]}"
            elif status['completed']:
                if status['correct'] == 19:
                    status_text = "✅ COMPLETED"
                else:
                    status_text = "❌ FAILED"
            else:
                status_text = "⏳ Running..."

            print(f"{model_name:<30} {status['move']:<10} {status['correct']:<10} {status['incorrect']:<10} {status_text:<20}")

        print("="*100)
        time.sleep(0.5)


def run_all_models_parallel(
    clues_data: List[dict],
    puzzle_identifier: str,
    serialization_method: SerializationMethod
) -> List[TestResult]:
    """Run all models in parallel and display progress."""
    # Filter out 'human' model
    models_to_test = [m for m in AVAILABLE_MODELS if m != 'human']

    # Create progress trackers
    progress_trackers = {model: ModelProgress(model) for model in models_to_test}

    # Start progress display thread
    stop_event = threading.Event()
    display_thread = threading.Thread(
        target=display_parallel_progress,
        args=(progress_trackers, stop_event)
    )
    display_thread.daemon = True
    display_thread.start()

    # Run tests in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(models_to_test)) as executor:
        future_to_model = {
            executor.submit(
                run_model_test_with_progress,
                model,
                clues_data,
                puzzle_identifier,
                serialization_method,
                progress_trackers[model]
            ): model
            for model in models_to_test
        }

        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error testing {model}: {e}")

    # Stop progress display
    stop_event.set()
    display_thread.join(timeout=1)

    # Clear screen and show final results
    print("\033[2J\033[H", end="")
    print("\n" + "="*100)
    print("FINAL RESULTS")
    print("="*100)
    print(f"{'Model':<30} {'Completed':<15} {'Moves':<10} {'Correct':<10} {'Incorrect':<10} {'Duration':<12} {'Cost':<10}")
    print("="*100)

    for result in sorted(results, key=lambda r: r.model_name):
        completed_icon = "✅" if result.completed else "❌"
        correct = sum(1 for m in result.moves if m.get('correct', False))
        incorrect = sum(1 for m in result.moves if not m.get('correct', False))

        print(f"{result.model_name:<30} {completed_icon:<15} {result.total_moves:<10} {correct:<10} {incorrect:<10} {result.duration_seconds:>10.1f}s ${result.cost_usd:>8.4f}")

    print("="*100)

    return results
