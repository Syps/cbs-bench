# CBS Benchmark

A benchmarking tool for evaluating large language models on logic puzzles using the game ["Clues by Sam"](https://cluesbysam.com).

## Overview

Test frontier LLMs on their ability to solve deductive reasoning puzzles. The tool fetches daily logic puzzles from cluesbysam.com and evaluates how well different models can work through the logical constraints to identify criminals and innocents in a 5x4 grid of characters.

## TODO
- [ ] Single command to test multiple/all models
- [ ] Command to generate performance comparison for a given puzzle (using cache)
  -  [ ] Show emoji version of performance
- [ ] Scoring algo

## How It Works

Each puzzle presents a 5x4 grid (20 people) where each person is either innocent or criminal. Players receive:
- Names and professions for all people
- Logic clues that become visible only after correctly identifying someone as innocent or criminal
- A logical dependency system where solving certain people unlocks clues for others

The goal is to use pure deductive reasoning to identify everyone correctly.

## Key Features

- **Multi-Model Support**: Test OpenAI GPT, Anthropic Claude, Google Gemini, DeepSeek, and xAI Grok models
- **Automatic Puzzle Fetching**: Downloads and caches daily puzzles from cluesbysam.com
- **Human Mode**: Play puzzles yourself to understand the challenge
- **Comprehensive Logging**: Records conversations, moves, timing, and token usage
- **Replay System**: Review recorded game sessions step-by-step
- **Smart Validation**: Ensures moves are logically deducible before checking correctness

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd logic-solver

# Install dependencies (requires Python 3.12+)
uv sync

# Set up API keys for model testing
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Usage

### Fetch Latest Puzzle
```bash
logic-solver fetch
```

### Test a Model
```bash
# Test Claude Sonnet 4 on today's puzzle
logic-solver test --model claude-sonnet-4-20250514

# Test on a specific date's puzzle
logic-solver test --model gpt-5 --puzzle 20241015

# Test on a custom URL
logic-solver test --model gemini-2.5-pro --puzzle https://example.com/custom-puzzle
```

### Preview a Puzzle
```bash
# See the puzzle without running a model
logic-solver test --preview --puzzle 20241015
```

### Play as Human
```bash
# Play the puzzle yourself
logic-solver test --model human
```

### Replay a Session
```bash
# Replay a recorded game (see .test_results/ for available sessions)
logic-solver replay 20241015-claude-sonnet-4-20250514
```

## Supported Models

- **OpenAI**: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- **Anthropic**: `claude-sonnet-4-20250514`, `claude-sonnet-4-5-20250929`
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- **DeepSeek**: `deepseek-chat`, `deepseek-reasoner`
- **xAI**: `grok-4`
- **Human**: `human` (for manual gameplay)

## Game Rules

### Grid Layout
- 5 rows � 4 columns = 20 people
- Each person has a name, profession, and gender
- Each person is either innocent or criminal

### Terminology
- **Neighbors**: Include diagonal neighbors (up to 8 per person)
- **Between**: People between two others, not including the endpoints
- **Connected**: Chain of orthogonal adjacency with no gaps
- **Left/Right**: Same row; **Above/Below**: Same column
- **Edge**: The 14 people around the perimeter

### Constraints
- Only one move per turn: "MOVE: [name] is [innocent|criminal]"
- Moves must be logically deducible from available information
- Incorrect moves are rejected with feedback
- Game ends when all 20 people are correctly identified

## Output Files

Results are saved to `.test_results/{puzzle-id}-{model-name}/`:
- `metadata.json`: Test summary (completion, timing, cost)
- `conversation.json`: Full conversation history
- `moves.json`: All moves attempted with results
- `puzzle.json`: Original puzzle data

## Example Session

```bash
$ logic-solver test --model claude-sonnet-4-20250514 --puzzle 20241015

Loading puzzle: 20241015
Loaded puzzle from cache: .cache/20241015.json
Puzzle contains 20 clues

Starting test with model: claude-sonnet-4-20250514

===============================================================================
<� MOVE 1 (Model: claude-sonnet-4-20250514)
===============================================================================

=� USER MESSAGE:
===============================================================================
PUZZLE GRID (5 rows x 4 columns):
Row 1: [Amy(sleuth):?] [Bobby(builder):?] [Daniel(singer):?] [Evie(sleuth):?]
Row 2: [Gary(guard):?] [Hilda(builder):?] [Isaac(guard):?] [Julie(guard):?]
Row 3: [Karen(cop):?] [Logan(sleuth):?] [Mary(judge):?] [Nancy(farmer):?]
Row 4: [Pam(builder):?] [Ruby(singer):?] [Steve(judge):?] [Tyler(farmer):?]
Row 5: [Vicky(mech):?] [Wally(cop):?] [Xavi(mech):?] [Zed(mech):?]

Solved: 0/20, Remaining: 20
===============================================================================

> CLAUDE-SONNET-4-20250514 RESPONSE:
-------------------------------------------------------------------------------
Looking at this logic puzzle, I need to find someone I can definitively identify as innocent or criminal based on the available information...

MOVE: Karen is innocent
-------------------------------------------------------------------------------

 Correct

=== TEST RESULTS ===
Model: claude-sonnet-4-20250514
Puzzle: 20241015
Completed: 
Total Moves: 23
Duration: 45.2 seconds
Tokens Used: 25,847
Cost: $0.1234
Results saved to: .test_results/20241015-claude-sonnet-4-20250514
```

## Development

### Project Structure
- `hello.py`: Main application logic and CLI
- `pyproject.toml`: Dependencies and project configuration
- `.cache/`: Cached puzzle data
- `.test_results/`: Recorded game sessions

### Key Components
- **Puzzle Fetching**: Scrapes cluesbysam.com and parses JavaScript puzzle data
- **Hint Translation**: Converts coded hints to natural language
- **Model Integration**: LangChain-based interfaces for different LLM providers
- **Validation Engine**: Z3-powered logical constraint checking
- **Game State Management**: Tracks puzzle progress and move history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.