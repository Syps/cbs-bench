import pdb
import json5
import click
import json
import re
import requests
import hashlib
import sys
import time
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain.memory import ChatMessageHistory

from z3 import *
from collections import defaultdict
from pydantic import BaseModel

ROWS = 5
COLS = 4


AVAILABLE_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "deepseek-chat",
    "deepseek-reasoner",
]

class CellData(BaseModel):
    name: str
    profession: str
    hint: str
    criminal: bool
    gender: str
    orig_hint: str = ""
    paths: List[List[int]] = []

class Status(Enum):
    UNKNOWN = 0
    CRIMINAL = 1
    INNOCENT = 2


class PuzzleCell(BaseModel):
    name: str
    profession: str
    gender: str
    clue: str
    status: Status = Status.UNKNOWN
    had_mistake: bool = False
    paths: List[List[int]] = []
    is_criminal: bool = False


class SerializationMethod(Enum):
    DEFAULT = "default"

@dataclass
class TestResult:
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
    pass

class GameStateError(Exception):
    pass

class ModelFactory:
    @staticmethod
    def create_model(model_name: str):
        """Create appropriate model client based on model name."""
        if model_name == 'human':
            return None  # Human mode doesn't need a model
        elif model_name.startswith('gpt'):
            return ChatOpenAI(model_name=model_name, temperature=0)
        elif model_name.startswith('claude'):
            return ChatAnthropic(model_name=model_name, temperature=0)
        elif model_name.startswith("gemini"):
            return ChatGoogleGenerativeAI(model=model_name, temperature=0)
        elif model_name.startswith("deepseek"):
            return ChatDeepSeek(model=model_name, temperature=0)
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
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name)
        self.memory = GameChatMemory()
        self.config = ModelFactory.get_model_config(model_name)
        self.total_tokens = 0
        self.total_cost = 0.0
        self.is_human = (model_name == 'human')

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
        # Print user message to stdout
        print("\n" + "="*80)
        print("📤 USER MESSAGE:")
        print("="*80)
        print(content)
        print("="*80)

        self.memory.add_user_message(content)

        for attempt in range(max_retries):
            try:
                response = self.model.invoke(self.memory.get_messages())
                response_content = response.content

                # Print AI response to stdout
                print(f"\n🤖 {self.model_name.upper()} RESPONSE:")
                print("-"*80)
                print(response_content)
                print("-"*80)

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

def fetch_clues_from_website(url=None):
    """Fetch clues data from cluesbysam.com (or custom URL) and save to dated JSON file."""
    base_url = url or "https://cluesbysam.com"
    
    # Fetch the main page
    response = requests.get(base_url)
    response.raise_for_status()
    
    # Parse HTML to find the script tag
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find script tag in head that references index JS file
    script_tags = soup.find('head').find_all('script', src=True)
    index_script_src = None
    
    for script in script_tags:
        src = script.get('src')
        if src and 'index-' in src and src.endswith('.js'):
            index_script_src = src
            break
    
    if not index_script_src:
        raise ValueError("Could not find index JS file in HTML")
    
    # Construct full URL for the JS file
    js_url = urljoin(base_url, index_script_src)
    
    # Fetch the JS file
    js_response = requests.get(js_url)
    js_response.raise_for_status()
    
    # Search for the clues data pattern in the JS content
    js_content = js_response.text
    
    # Look for array of objects with criminal, profession, name, gender fields
    # Pattern matches: =[{criminal:!0,profession:"...",name:"...",gender:"...",...},...],nextVar=
    # Stop at ] followed by comma and next variable assignment
    pattern = r'=(\[\{.*?criminal:.*?profession:.*?name:.*?gender:.*?\}]),'
    
    match = re.search(pattern, js_content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find clues data in JS file")
    
    # Extract the array part
    clues_text = match.group(1)
    
    # Try to convert JS object to valid JSON
    # Replace single quotes with double quotes, handle JS boolean values
    # json_text = clues_text.replace("'", '"').replace('True', 'true').replace('False', 'false')
    json_text = clues_text.replace('!0', 'true').replace('!1', 'false')
    with open('json_text.txt', 'w') as f:
        f.write(json_text)
    # import sys;sys.exit(0)
    
    clues_data = json5.loads(json_text)
    
    # Create filename in .cache folder
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    if url:
        # Use MD5 hash of URL for filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        filename = os.path.join(cache_dir, f"{url_hash}.json")
    else:
        # Use today's date for default URL
        today = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(cache_dir, f"{today}.json")
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(clues_data, f, indent=2)
    
    print(f"Fetched {len(clues_data)} clues and saved to {filename}")
    print("Preview of data:")
    print(json.dumps(clues_data[:2], indent=2))  # Show first 2 items
    
    return clues_data, filename

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

@click.command()
def main():
    print("Hello from logic-solver!")

@click.command()
@click.argument('json_path', type=click.Path(exists=True))
def ingest(json_path):
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

def get_cache_filename(puzzle_value):
    """Get the cache filename for a puzzle value (date or URL)."""
    cache_dir = ".cache"
    
    if puzzle_value.startswith(('http://', 'https://')):
        # URL: use MD5 hash
        url_hash = hashlib.md5(puzzle_value.encode()).hexdigest()
        return os.path.join(cache_dir, f"{url_hash}.json")
    else:
        # Date string: use date directly
        return os.path.join(cache_dir, f"{puzzle_value}.json")

def load_puzzle_from_cache(puzzle_value):
    """Load puzzle data from cache if it exists."""
    cache_filename = get_cache_filename(puzzle_value)
    
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            return json.load(f), cache_filename
    
    return None, cache_filename

def fetch_and_cache_puzzle(puzzle_value):
    """Fetch puzzle data and cache it."""
    if puzzle_value.startswith(('http://', 'https://')):
        # Custom URL
        clues_data, filename = fetch_clues_from_website(puzzle_value)
    else:
        # Date string - use default cluesbysam.com
        clues_data, filename = fetch_clues_from_website()
        
        # For date strings, we need to rename the file to match the date
        cache_dir = ".cache"
        expected_filename = os.path.join(cache_dir, f"{puzzle_value}.json")
        if filename != expected_filename:
            os.rename(filename, expected_filename)
            filename = expected_filename
    
    return clues_data, filename

def serialize_puzzle_state(clues_data: List[dict], current_state: List[List[PuzzleCell]], method: SerializationMethod) -> str:
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
                    clue=person_data.get('hint', "NO HINT"),
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
            # Print move result to stdout
            status_emoji = "✅" if is_correct else "❌"
            # print(f"\n{status_emoji} MOVE RESULT:")
            # print(f"Move: {move}")
            # print(f"Feedback: {feedback}")
            # print("="*80)
            
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

@click.command()
@click.option('--model', type=click.Choice(AVAILABLE_MODELS), help='Name of the model to test')
@click.option('--puzzle', type=str, help='Puzzle to test on (YYYYMMDD date or URL, defaults to today)')
@click.option('--serialization',
              type=click.Choice([e.value for e in SerializationMethod]),
              default=SerializationMethod.DEFAULT.value,
              help='Puzzle serialization method')
@click.option('--preview', is_flag=True, help='Show only the initial puzzle state without running model test')
def test(model, puzzle, serialization, preview):
    """Test a model on a specific puzzle."""
    # Validate required parameters
    if not preview and not model:
        click.echo("Error: --model is required unless using --preview", err=True)
        return

    validated_puzzle = validate_puzzle(puzzle)

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
        serialization_method = SerializationMethod(serialization)
        test_result = run_model_test(model, clues_data, validated_puzzle, serialization_method)

        # Step 3: Save and print results
        results_path = save_test_results(test_result)
        print_test_evaluation(test_result, results_path)

    except Exception as e:
        click.echo(f"Test failed: {e}", err=True)
        raise

"""
1. Initialize all cells in grid
2. Set clue constraints for starting clue
3. 
"""


"""
Example clues -> constraints translations
-----------------------------------------

Clue: 'There's an odd number of criminals to the left of Cheryl'
Constraint Logic:
Assume Cheryl pos = (cR, cC)
The grid is a list of rows
Total cells left of Cheryl =  cC
'Sum True's mod 2 == 1'
bools = [grid[cR][i] for i in range(cC)]
total_true = Sum([If(b, 1, 0) for b in bools])]
constraint = total_true % 2 == 1


Official constraints
--------------------
- all_traits_are_neighbors_in_unit(unit(between,pair(12,15)),innocent)
    
- number_of_traits_in_unit(unit(neighbor,14),innocent,6)
    - #NAME:14 has exactly 6 innocent neighbors
    
- unit_shares_n_out_of_n_traits_with_unit(unit(neighbor,1),unit(neighbor,2),innocent,1,2)
    - Only 1 of the 2 innocents neighboring #NAME:1 is #NAMES:2 neighbor

- units_share_odd_n_traits(unit(neighbor,14),unit(neighbor,15),innocent)
    - #NAME:14 and #NAME:15 share an odd number of innocent neighbours

- is_one_of_n_traits_in_unit(unit(neighbor,2),3,criminal,4)
    - #NAME:3 is one of #NAMES:2 4 criminal neighbors
    
- is_not_only_trait_in_unit(unit(row,2),4,innocent)
    - #NAME:4 is one of two or more innocents in row2

- all_traits_are_neighbors_in_unit(unit(between,pair(3,19)),criminal)
    - All criminals #BETWEEN:pair(3,19) are connected

- odd_number_of_traits_in_unit(unit(col,3),criminal)
    - There's an odd number of criminals in column#C:3

- more_traits_in_unit_than_unit(unit(neighbor,16),unit(neighbor,3),innocent)
    - #NAME:16 has more innocent neighbors than #NAME:3

- has_most_traits(unit(row,4),innocent)
    - Row 4 has more innocents than any other row

- equal_traits_and_traits_in_unit(unit(profession,cop),innocent,criminal)
    - There's an equal number of innocent and criminal #PROFS:cop


"""




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


@click.group()
def cli():
    pass

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

cli.add_command(main)
cli.add_command(ingest)
cli.add_command(fetch)
cli.add_command(test)
cli.add_command(replay)

if __name__ == "__main__":
    cli()
    # def green_background_text_only(text):
    #     GREEN_BG = '\033[102m'
    #     RESET = '\033[0m'
    #
    #     lines = text.split('\n')
    #     colored_lines = [f"{GREEN_BG}{line}{RESET}" for line in lines]
    #     return '\n'.join(colored_lines)
    #
    #
    # # Usage
    # multiline_text = """This is line 1
    # This is line 2
    # This is line 3"""
    #
    # print(green_background_text_only(multiline_text))