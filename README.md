# CBS Benchmark

> **Benchmarking frontier AI model reasoning capability with [Clues By Sam](https://cluesbysam.com)**

Can large language models truly reason, or are they just sophisticated pattern matchers? This benchmark explores that question through [Clues by Sam](https://cluesbysam.com), a daily logic puzzle that demands pure deduction without shortcuts.

## What This Is

Clues by Sam presents players with a 5x4 grid of 20 people, each either innocent or criminal. Your only tools are logic clues that unlock progressively as you correctly identify people. No guessing allowed—each move must be logically deducible or it's rejected.

This benchmark converts those web-based puzzles into ASCII grids and runs frontier AI models through them in an agentic loop, measuring how well they handle constraint-based reasoning under pressure. The results are fascinating: GPT-5 Pro achieved near-perfect accuracy, outperforming human solvers, while other models struggled. Does this prove reasoning ability, or just really good pattern recognition? Judge for yourself.

**Read the full analysis:** [Clues By Sam LLM Benchmark](https://www.nicksypteras.com/blog/cbs-benchmark.html)

---

## How It Works

Each puzzle presents a 5x4 grid (20 people) where each person is either innocent or criminal. Players receive:
- Names and professions for all people
- Logic clues that become visible only after correctly identifying someone as innocent or criminal
- A logical dependency system where solving certain people unlocks clues for others

The goal is to use pure deductive reasoning to identify everyone correctly.

## Key Features

- **Multi-Model Support** — Test OpenAI GPT, Anthropic Claude, Google Gemini, DeepSeek, and xAI Grok models
- **Automatic Puzzle Fetching** — Downloads and caches daily puzzles from cluesbysam.com
- **Human Mode** — Play puzzles yourself to experience the challenge firsthand
- **Comprehensive Logging** — Tracks conversations, moves, timing, token usage, and costs
- **Replay System** — Review any recorded game session step-by-step
- **Performance Analytics** — Generate statistics grouped by model, puzzle, or difficulty
- **Smart Validation** — Rejects moves that aren't logically deducible from available information

---

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
export GOOGLE_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export XAI_API_KEY="your-xai-key"
```

---

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

# Test all models in parallel
logic-solver test --all-models --puzzle 20241015
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

### View Statistics
```bash
# Show overall stats for all models
logic-solver stats

# Show stats for specific model(s)
logic-solver stats --model claude-sonnet-4-5-20250929
logic-solver stats --model "gpt-5,deepseek-chat,claude-sonnet-4-5-20250929"

# Show stats grouped by difficulty (puzzle pack 1 only)
logic-solver stats --by-difficulty

# Show stats for a specific puzzle
logic-solver stats --puzzle 20241015

# Compare models on same puzzles only
logic-solver stats --only-same-puzzles
```

### Replay a Session
```bash
# Replay a recorded game (see .test_results/ for available sessions)
logic-solver replay 20241015-claude-sonnet-4-20250514
```

### Makefile Commands
```bash
# Generate stats for selected models grouped by difficulty
make stats
```

---

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | `gpt-5`, `gpt-5-mini`, `gpt-5-nano` |
| **Anthropic** | `claude-sonnet-4-20250514`, `claude-sonnet-4-5-20250929` |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner` |
| **xAI** | `grok-4` |
| **Human** | `human` *(play the puzzle yourself)* |

---

## Game Rules

### Grid Layout
- 5 rows × 4 columns = 20 people total
- Each person has a name, profession, and gender
- Each person is either innocent or criminal (binary classification)

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

---

## Output Files

Results are saved to `.test_results/{puzzle-id}-{model-name}/`:
- `metadata.json`: Test summary (completion, timing, cost)
- `conversation.json`: Full conversation history
- `moves.json`: All moves attempted with results
- `puzzle.json`: Original puzzle data

---

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

---

## Development

### Project Structure
```
├── cli.py              # Click-based CLI commands and Z3 constraint solver
├── eval.py             # Model testing, evaluation, and agentic loop
├── fetch.py            # Puzzle fetching and caching from cluesbysam.com
├── models.py           # Pydantic models and data structures
├── pyproject.toml      # Dependencies and project configuration
├── .cache/             # Cached puzzle data
└── .test_results/      # Recorded game sessions with full logs
```

### Key Components
- **Puzzle Fetching** (`fetch.py`) — Scrapes cluesbysam.com and parses JavaScript puzzle data
- **Hint Translation** — Converts coded hint references to natural language
- **Model Integration** (`eval.py`) — LangChain-based interfaces for different LLM providers
- **Validation Engine** (`cli.py`) — Z3-powered logical constraint checking
- **Game State Management** — Tracks puzzle progress and validates move legality

---

## Contributing

Contributions are welcome! Whether you want to add support for new models, improve the evaluation logic, or enhance the statistics system:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

---

## Citation

If you use this benchmark in your research or analysis, please link back to the [original blog post](https://www.nicksypteras.com/blog/cbs-benchmark.html).

---

## License

MIT License - see LICENSE file for details.

---

Built with curiosity about AI reasoning by [Nick Sypteras](https://www.nicksypteras.com)