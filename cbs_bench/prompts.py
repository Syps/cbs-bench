SOLVER_SYSTEM_PROMPT = """
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

GENERATOR_PROMPT_TEMPLATE = """
You are a master puzzle creator. You need to create a puzzle for the following logic game:

# GAME RULES:
- You have a grid of {total_cells} people ({rows} rows of {cols} columns)
- Each person is either innocent or criminal
- Each person has a name, profession, and hint defined in a DSL string (syntax rules described in `# DSL Syntax`
- The hint describes 
- The clue is only visible once you have correctly marked the person as criminal or innocent.
- Your goal is to determine who is innocent and who is criminal using only logic.
- There is always a logical choice, even if you think there isn't.
- You can only make one move at a time
"""