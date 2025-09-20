import pdb
import json5
import click
import json
import re
import requests
import hashlib
import os
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
from langchain.memory import ChatMessageHistory

from z3 import *
from collections import defaultdict
from pydantic import BaseModel

class CellData(BaseModel):
    name: str
    profession: str
    hint: str
    criminal: bool
    gender: str
    orig_hint: str = ""
    paths: List[List[int]] = []

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

class ModelCommunicationError(Exception):
    pass

class GameStateError(Exception):
    pass

class ModelFactory:
    @staticmethod
    def create_model(model_name: str):
        """Create appropriate LangChain model based on model name."""
        if model_name.startswith('gpt'):
            return ChatOpenAI(model_name=model_name, temperature=0)
        elif model_name.startswith('claude'):
            return ChatAnthropic(model_name=model_name, temperature=0)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def get_model_config(model_name: str) -> dict:
        """Get model-specific configuration."""
        return {
            "temperature": 0,
            "max_tokens": 1000,
            "timeout": 30
        }

class GameChatMemory:
    def __init__(self):
        self.message_history = ChatMessageHistory()
        self.conversation_log = []
    
    def add_system_message(self, content: str):
        msg = SystemMessage(content=content)
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
    
    def send_system_message(self, content: str):
        """Send initial system message with game rules."""
        self.memory.add_system_message(content)
    
    def send_message_and_get_response(self, content: str, max_retries: int = 3) -> str:
        """Send message with full context and get model response."""
        self.memory.add_user_message(content)
        
        for attempt in range(max_retries):
            try:
                messages = self.memory.get_messages()
                response = self.model.invoke(messages)
                response_content = response.content
                
                self.memory.add_ai_message(response_content)
                return response_content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    error_msg = f"Model communication failed after {max_retries} attempts: {e}"
                    self.memory.add_ai_message(error_msg)
                    raise ModelCommunicationError(error_msg)
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_conversation_history(self):
        return self.memory.get_conversation_log()

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
    pattern = r'=(\[\{.*?criminal:.*?profession:.*?name:.*?gender:.*?\}.*?]),\w+='
    
    match = re.search(pattern, js_content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find clues data in JS file")
    
    # Extract the array part
    clues_text = match.group(1)
    
    # Try to convert JS object to valid JSON
    # Replace single quotes with double quotes, handle JS boolean values
    # json_text = clues_text.replace("'", '"').replace('True', 'true').replace('False', 'false')
    json_text = clues_text.replace('!0', 'true').replace('!1', 'false')
    
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

def serialize_puzzle_state(clues_data: List[dict], current_state: dict, method: SerializationMethod) -> str:
    """Serialize puzzle state for model consumption."""
    if method == SerializationMethod.DEFAULT:
        unsolved_count = sum(1 for person in clues_data if person['name'] not in current_state.get('solved', {}))
        return f"PUZZLE STATE (STUB): {len(clues_data)} people, {unsolved_count} unsolved"
    
    raise ValueError(f"Unsupported serialization method: {method}")

def initialize_puzzle_state(clues_data: List[dict]) -> dict:
    """Initialize empty puzzle state."""
    return {
        "solved": {},  # name -> innocent/criminal
        "grid": [[None for _ in range(4)] for _ in range(5)],  # 5x4 grid
        "people": {person['name']: person for person in clues_data}
    }

def extract_move_from_response(response: str) -> str:
    """Extract move from model response."""
    pattern = r"MOVE:\s*(\w+)\s+is\s+(innocent|criminal)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return f"{match.group(1)} is {match.group(2)}"
    
    raise ValueError(f"Could not extract move from response: {response}")

def validate_move(move: str, current_state: dict, clues_data: List[dict]) -> Tuple[bool, str]:
    """Validate if a move is correct."""
    # TODO: Implement actual validation logic
    return True, "Move validation not implemented"

def update_puzzle_state(current_state: dict, move: str) -> dict:
    """Update puzzle state after a correct move."""
    # TODO: Implement state update logic
    # Parse move and update current_state['solved']
    parts = move.split(' is ')
    if len(parts) == 2:
        name, classification = parts
        current_state['solved'][name] = classification
    return current_state

def is_puzzle_complete(current_state: dict) -> bool:
    """Check if puzzle is completely solved."""
    return len(current_state.get('solved', {})) == 20

def get_system_prompt() -> str:
    """Get initial system prompt with game rules."""
    return """
You are playing a logic puzzle game called "Clues by Sam". Here are the rules:

GAME RULES:
- You have a 5x4 grid of people (20 total)
- Each person is either innocent or criminal
- Each person has a name, profession, and may have a clue
- Your goal is to determine who is innocent and who is criminal
- You can only make one move at a time

RESPONSE FORMAT:
Your response must contain exactly one move in this format:
MOVE: [person_name] is [innocent|criminal]

Example: "MOVE: Alice is innocent"

IMPORTANT:
- Only make one move per response
- Base your reasoning on the clues provided
- Explain your reasoning before stating your move
- If you're unsure, make your best guess

Ready to start? I'll give you the current puzzle state.
"""

def get_state_prompt_template() -> str:
    """Template for puzzle state messages."""
    return """
CURRENT PUZZLE STATE:
{serialized_state}

REMAINING UNSOLVED: {unsolved_count} people

What is your next move?
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
        "max_moves_reached": test_result.max_moves_reached
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
    current_state = initialize_puzzle_state(clues_data)
    max_moves = 100
    
    # Initialize model tester
    model_tester = ModelTester(model_name)
    
    # Send system message with game rules
    system_prompt = get_system_prompt()
    model_tester.send_system_message(system_prompt)
    
    # Main game loop
    move_count = 0
    
    while move_count < max_moves:
        try:
            # Serialize current puzzle state
            state_message = serialize_puzzle_state(clues_data, current_state, serialization_method)
            
            # Send state to model and get response
            model_response = model_tester.send_message_and_get_response(state_message)
            
            # Extract and validate move
            move = extract_move_from_response(model_response)
            is_correct, feedback = validate_move(move, current_state, clues_data)
            
            # Record move
            moves.append({
                "move_number": move_count + 1,
                "move": move,
                "correct": is_correct,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            
            if is_correct:
                current_state = update_puzzle_state(current_state, move)
                
                if is_puzzle_complete(current_state):
                    model_tester.send_message_and_get_response("Congratulations! You've solved the puzzle!")
                    break
            else:
                feedback_message = f"Incorrect move. {feedback} Please try again."
                model_tester.send_message_and_get_response(feedback_message)
            
            move_count += 1
            
        except (ModelCommunicationError, ValueError, GameStateError) as e:
            moves.append({
                "move_number": move_count + 1,
                "move": "ERROR",
                "correct": False,
                "feedback": str(e),
                "timestamp": datetime.now().isoformat()
            })
            break
    
    # Calculate results
    duration = time.time() - start_time
    completed = is_puzzle_complete(current_state)
    max_moves_reached = move_count >= max_moves
    
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
        max_moves_reached=max_moves_reached
    )

@click.command()
@click.option('--model', required=True, type=str, help='Name of the model to test')
@click.option('--puzzle', type=str, help='Puzzle to test on (YYYYMMDD date or URL, defaults to today)')
@click.option('--serialization', 
              type=click.Choice([e.value for e in SerializationMethod]), 
              default=SerializationMethod.DEFAULT.value,
              help='Puzzle serialization method')
def test(model, puzzle, serialization):
    """Test a model on a specific puzzle."""
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
    
    # Validate model
    try:
        ModelFactory.create_model(model)
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

cli.add_command(main)
cli.add_command(ingest)
cli.add_command(fetch)
cli.add_command(test)

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