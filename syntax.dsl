# Logic Puzzle DSL Syntax Reference

## Overview

This DSL (Domain Specific Language) is used to express constraints for logic puzzles where cells in a grid are either "criminal" or "innocent" and have associated professions. The puzzles can be any dimensions (rows and columns) but are usually 4x5 or smaller.

## Grid Layout

```
Grid indices (5 rows × 4 columns):
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    16 17 18 19
```

Cell indexing: `index = row * 4 + col`
- Cell 0: row 0, col 0 (top-left)
- Cell 3: row 0, col 3 (top-right)
- Cell 19: row 4, col 3 (bottom-right)

---

## Type System

### Enumerations

#### trait
Represents whether a cell is criminal or innocent.
- Values: `criminal`, `innocent`
- Example: `criminal`

#### unit_type
Represents the type of unit/group being referenced.
- Values: `row`, `col`, `corner`, `edge`, `between`, `neighbor`, `profession`
- Example: `row`

#### profession
Represents the profession of a cell.
- Values: `builder`, `clerk`, `coder`, `cook`, `cop`, `doctor`, `farmer`, `guard`, `judge`, `mech`, `painter`, `pilot`, `singer`, `sleuth`, `teacher`
- Example: `guard`

### Special Constants

#### void
A special constant used with edge/corner units that don't require a selector.
- Usage: `unit(edge,void)`, `unit(corner,void)`

---

## Constructor Functions

### pair(int, int)
Creates a pair of cell indices for defining a range.

**Syntax:** `pair(a_index, b_index)`

**Parameters:**
- `a_index`: First cell index (0-19)
- `b_index`: Second cell index (0-19)

**Constraints:**
- Both indices must be in the same row OR the same column
- Used exclusively with `unit(between, ...)`

**Examples:**
```
pair(0,3)      # Top row, left to right (cells 0,1,2,3)
pair(3,15)     # Right column, top to bottom (cells 3,7,11,15)
pair(5,7)      # Row 1, partial (cells 5,6,7)
```

---

### unit(unit_type, selector)
Creates a unit (group of cells) for constraint evaluation.

**Syntax:** `unit(type, selector)`

**Parameters:**
- `type`: One of the unit_type values
- `selector`: Depends on unit_type (see table below)

**Unit Type Reference:**

| unit_type   | selector type | selector example | meaning                          |
|-------------|---------------|------------------|----------------------------------|
| row         | int           | 0                | All cells in row 0               |
| col         | int           | 2                | All cells in column 2            |
| corner      | void          | void             | The 4 corner cells               |
| edge        | void          | void             | All cells on the grid edges      |
| between     | pair          | pair(0,3)        | Cells between two indices        |
| neighbor    | int           | 5                | Neighbors of cell 5 (8 max)      |
| profession  | profession    | guard            | All cells with profession=guard  |

**Examples:**
```
unit(row,0)                  # All cells in row 0: [0,1,2,3]
unit(col,2)                  # All cells in column 2: [2,6,10,14,18]
unit(corner,void)            # Four corners: [0,3,16,19]
unit(edge,void)              # All edge cells
unit(between,pair(0,3))      # Horizontal range: [0,1,2,3]
unit(between,pair(0,16))     # Vertical range: [0,4,8,12,16]
unit(neighbor,5)             # Neighbors of cell 5: [0,1,2,4,6,8,9,10]
unit(profession,guard)       # All guards in the grid
```

**Important Notes:**
- `between` returns cells in sorted (ascending) order
- `neighbor` includes all 8 adjacent cells (or fewer at edges)
- Edge cells include entire top row, bottom row, and left/right columns (with overlap at corners)

---

## Constraint Functions

### Counting and Cardinality

#### number_of_traits(trait, count)
Specifies the exact total number of a trait across the entire grid.

**Syntax:** `number_of_traits(trait, count)`

**Example:**
```
number_of_traits(criminal,16)
# Meaning: There are exactly 16 criminals in total
```

---

#### number_of_traits_in_unit(unit, trait, count)
Specifies the exact number of a trait within a specific unit.

**Syntax:** `number_of_traits_in_unit(unit, trait, count)`

**Examples:**
```
number_of_traits_in_unit(unit(between,pair(1,5)),innocent,1)
# Meaning: There is exactly 1 innocent between cells 1 and 5

number_of_traits_in_unit(unit(row,0),criminal,2)
# Meaning: Row 0 has exactly 2 criminals
```

---

#### min_number_of_traits_in_unit(unit, trait, min_count)
Specifies a minimum number of a trait within a unit.

**Syntax:** `min_number_of_traits_in_unit(unit, trait, min_count)`

**Example:**
```
min_number_of_traits_in_unit(unit(edge,void),criminal,7)
# Meaning: There are at least 7 criminals on the edges
```

---

#### all_units_have_at_least_n_traits(unit_type, trait, n)
Every unit of a given type has at least N instances of the trait.

**Syntax:** `all_units_have_at_least_n_traits(unit_type, trait, n)`

**Examples:**
```
all_units_have_at_least_n_traits(row,innocent,2)
# Meaning: Each row has at least 2 innocents

all_units_have_at_least_n_traits(col,criminal,1)
# Meaning: Each column has at least 1 criminal
```

---

#### odd_number_of_traits_in_unit(unit, trait)
The unit contains an odd number of the specified trait.

**Syntax:** `odd_number_of_traits_in_unit(unit, trait)`

**Example:**
```
odd_number_of_traits_in_unit(unit(profession,guard),innocent)
# Meaning: There's an odd number of innocent guards
```

---

### Comparison and Equality

#### equal_number_of_traits_in_units(unit_1, unit_2, trait)
Two units have the same number of a trait.

**Syntax:** `equal_number_of_traits_in_units(unit_1, unit_2, trait)`

**Example:**
```
equal_number_of_traits_in_units(unit(neighbor,0),unit(neighbor,7),innocent)
# Meaning: Cell 0 and cell 7 have an equal number of innocent neighbors
```

---

#### equal_traits_and_traits_in_unit(unit, trait_1, trait_2)
Within a unit, two different traits have equal counts.

**Syntax:** `equal_traits_and_traits_in_unit(unit, trait_1, trait_2)`

**Example:**
```
equal_traits_and_traits_in_unit(unit(between,pair(15,19)),criminal,innocent)
# Meaning: There are as many criminals as innocents between cells 15 and 19
```

---

#### more_traits_in_unit_than_unit(unit_1, unit_2, trait)
Unit 1 has more of the trait than unit 2.

**Syntax:** `more_traits_in_unit_than_unit(unit_1, unit_2, trait)`

**Example:**
```
more_traits_in_unit_than_unit(unit(neighbor,2),unit(neighbor,19),innocent)
# Meaning: Cell 2 has more innocent neighbors than cell 19
```

---

#### more_traits_than_traits_in_unit(unit, trait_1, trait_2)
Within a unit, trait_1 count exceeds trait_2 count.

**Syntax:** `more_traits_than_traits_in_unit(unit, trait_1, trait_2)`

**Example:**
```
more_traits_than_traits_in_unit(unit(neighbor,17),criminal,innocent)
# Meaning: Cell 17 has more criminal than innocent neighbors
```

---

#### has_most_traits(unit, trait)
The specified unit has more of the trait than any other unit of the same type.

**Syntax:** `has_most_traits(unit, trait)`

**Example:**
```
has_most_traits(unit(col,4),innocent)
# Meaning: Column 4 has more innocents than any other column
```

---

### Uniqueness and Exclusivity

#### only_one_unit_has_exactly_n_traits(unit_type, trait, n)
Exactly one unit of the given type has exactly N of the trait.

**Syntax:** `only_one_unit_has_exactly_n_traits(unit_type, trait, n)`

**Example:**
```
only_one_unit_has_exactly_n_traits(col,innocent,1)
# Meaning: Only one column has exactly 1 innocent
```

---

#### only_unit_has_exactly_n_traits(unit, trait, n)
This specific unit is the only one (among units of its type) with exactly N of the trait.

**Syntax:** `only_unit_has_exactly_n_traits(unit, trait, n)`

**Example:**
```
only_unit_has_exactly_n_traits(unit(neighbor,19),innocent,1)
# Meaning: Cell 19 is the only cell with exactly 1 innocent neighbor
```

---

### Connectivity and Adjacency

#### all_traits_are_neighbors_in_unit(unit, trait)
All instances of the trait in the unit form a connected path (no gaps).

**Syntax:** `all_traits_are_neighbors_in_unit(unit, trait)`

**Semantics:**
- For a linear sequence (row/column/between), all traits must be adjacent
- If cells at positions i and i+2 are the trait, then i+1 must also be the trait

**Examples:**
```
all_traits_are_neighbors_in_unit(unit(between,pair(0,16)),criminal)
# Meaning: All criminals in the left column form a connected vertical line

all_traits_are_neighbors_in_unit(unit(row,2),innocent)
# Meaning: All innocents in row 2 are adjacent to each other
```

---

#### both_traits_are_neighbors_in_unit(unit, trait)
Exactly 2 of the trait exist in the unit and they are adjacent.

**Syntax:** `both_traits_are_neighbors_in_unit(unit, trait)`

**Example:**
```
both_traits_are_neighbors_in_unit(unit(between,pair(3,15)),innocent)
# Meaning: There are exactly 2 innocents between cells 3 and 15, and they are next to each other
```

---

### Cell-Specific Constraints

#### has_trait(cell_id, trait)
A specific cell has the given trait.

**Syntax:** `has_trait(cell_id, trait)`

**Example:**
```
has_trait(19,innocent)
# Meaning: Cell 19 is innocent
```

---

#### is_one_of_n_traits_in_unit(unit, cell_id, trait, n)
The specified cell is one of exactly N instances of the trait in the unit.

**Syntax:** `is_one_of_n_traits_in_unit(unit, cell_id, trait, n)`

**Example:**
```
is_one_of_n_traits_in_unit(unit(neighbor,6),5,criminal,5)
# Meaning: Cell 5 is one of 5 criminals that are neighbors of cell 6
```

---

#### is_not_only_trait_in_unit(unit, cell_id, trait)
The specified cell has the trait, but is not the only one in the unit with that trait.

**Syntax:** `is_not_only_trait_in_unit(unit, cell_id, trait)`

**Example:**
```
is_not_only_trait_in_unit(unit(between,pair(5,7)),5,criminal)
# Meaning: Cell 5 is a criminal, but there are other criminals between cells 5 and 7
```

---

### Inter-Unit Relationships

#### both_traits_in_unit_are_in_unit(unit_1, unit_2, trait)
Both instances of the trait in unit_1 are also in unit_2.

**Syntax:** `both_traits_in_unit_are_in_unit(unit_1, unit_2, trait)`

**Example:**
```
both_traits_in_unit_are_in_unit(unit(col,3),unit(neighbor,11),innocent)
# Meaning: Both innocents in column 3 are neighbors of cell 11
```

---

#### only_trait_in_unit_is_in_unit(unit_1, unit_2, trait)
The single instance of the trait in unit_1 is also in unit_2.

**Syntax:** `only_trait_in_unit_is_in_unit(unit_1, unit_2, trait)`

**Example:**
```
only_trait_in_unit_is_in_unit(unit(between,pair(0,2)),unit(between,pair(1,3)),innocent)
# Meaning: There's only one innocent between cells 0-2, and it's also between cells 1-3
```

---

#### total_number_of_traits_in_units(unit_1, unit_2, trait, total)
The combined count of the trait across two units equals the specified total.

**Syntax:** `total_number_of_traits_in_units(unit_1, unit_2, trait, total)`

**Example:**
```
total_number_of_traits_in_units(unit(neighbor,11),unit(neighbor,16),innocent,4)
# Meaning: Cells 11 and 16 have 4 innocent neighbors combined
```

---

#### units_share_n_traits(unit_1, unit_2, trait, n)
Exactly N cells with the trait appear in both units (intersection count).

**Syntax:** `units_share_n_traits(unit_1, unit_2, trait, n)`

**Examples:**
```
units_share_n_traits(unit(between,pair(2,14)),unit(neighbor,13),innocent,0)
# Meaning: No innocents between cells 2-14 are neighbors of cell 13

units_share_n_traits(unit(row,0),unit(col,2),criminal,1)
# Meaning: Exactly 1 criminal appears in both row 0 and column 2 (their intersection)
```

---

#### units_share_odd_n_traits(unit_1, unit_2, trait)
An odd number of cells with the trait appear in both units.

**Syntax:** `units_share_odd_n_traits(unit_1, unit_2, trait)`

**Example:**
```
units_share_odd_n_traits(unit(edge,void),unit(neighbor,7),innocent)
# Meaning: An odd number of innocents on the edges are neighbors of cell 7
```

---

#### unit_shares_n_out_of_n_traits_with_unit(unit_1, unit_2, trait, n, total)
Out of 'total' instances of the trait in unit_1, exactly 'n' are also in unit_2.

**Syntax:** `unit_shares_n_out_of_n_traits_with_unit(unit_1, unit_2, trait, n, total)`

**Example:**
```
unit_shares_n_out_of_n_traits_with_unit(unit(edge,void),unit(neighbor,2),criminal,1,10)
# Meaning: There are 10 criminals on the edges, and exactly 1 of them is a neighbor of cell 2
```

---

### Profession-Direction Constraints

#### every_profession_has_a_trait_in_dir(profession, trait, dx, dy)
Every cell with the given profession has a cell with the trait in the specified direction.

**Syntax:** `every_profession_has_a_trait_in_dir(profession, trait, dx, dy)`

**Parameters:**
- `dx`: Column offset (-1 = left, 0 = same, 1 = right)
- `dy`: Row offset (-1 = up, 0 = same, 1 = down)

**Direction Reference:**
- `(-1, 0)`: directly to the left
- `(1, 0)`: directly to the right
- `(0, -1)`: directly above
- `(0, 1)`: directly below

**Example:**
```
every_profession_has_a_trait_in_dir(guard,criminal,-1,0)
# Meaning: Every guard has a criminal directly to their left
```

---

#### n_professions_have_trait_in_dir(profession, trait, dx, dy, n)
Exactly N cells with the profession have a cell with the trait in the specified direction.

**Syntax:** `n_professions_have_trait_in_dir(profession, trait, dx, dy, n)`

**Examples:**
```
n_professions_have_trait_in_dir(singer,criminal,0,-1,0)
# Meaning: No singer has a criminal directly above them

n_professions_have_trait_in_dir(teacher,innocent,1,0,2)
# Meaning: Exactly 2 teachers have an innocent directly to their right
```

---

## Usage Guidelines

### Writing Valid Expressions

1. **All functions return constraints** - Each function call represents a logical constraint
2. **Nested constructors** - Use `unit()` and `pair()` to build arguments for constraint functions
3. **No variables** - The DSL is purely functional, no variable assignment
4. **Type matching** - Ensure parameter types match function signatures

### Common Patterns

**Counting in a region:**
```
number_of_traits_in_unit(unit(row,0),criminal,2)
```

**Comparing two regions:**
```
more_traits_in_unit_than_unit(unit(col,0),unit(col,3),innocent)
```

**Adjacency in a line:**
```
all_traits_are_neighbors_in_unit(unit(between,pair(0,3)),criminal)
```

**Profession-based:**
```
every_profession_has_a_trait_in_dir(guard,criminal,0,1)
```

### Invalid Expressions

❌ `unit(row,pair(0,1))` - row takes int, not pair
❌ `pair(0,5)` - cells 0 and 5 are not in same row/column
❌ `number_of_traits(unit(row,0),2)` - wrong signature, needs trait
❌ `has_trait(criminal,5)` - parameters reversed

---

## Complete Example

For a 5×4 grid, here are some valid constraints:

```
# Global counts
number_of_traits(criminal,10)

# Row constraints
all_units_have_at_least_n_traits(row,innocent,1)
number_of_traits_in_unit(unit(row,0),criminal,3)

# Column constraints
has_most_traits(unit(col,2),innocent)

# Neighbor relationships
equal_number_of_traits_in_units(unit(neighbor,0),unit(neighbor,19),criminal)
more_traits_in_unit_than_unit(unit(neighbor,5),unit(neighbor,10),innocent)

# Connectivity
all_traits_are_neighbors_in_unit(unit(between,pair(0,16)),criminal)

# Profession-based
every_profession_has_a_trait_in_dir(guard,criminal,-1,0)
n_professions_have_trait_in_dir(cook,innocent,0,1,3)

# Inter-unit
units_share_n_traits(unit(row,0),unit(col,0),criminal,1)
```

---

## Grid Coordinate Reference

Quick reference for common cell positions and their indices:

```
Top-left corner: 0
Top-right corner: 3
Bottom-left corner: 16
Bottom-right corner: 19

First row: 0,1,2,3
Last row: 16,17,18,19
First column: 0,4,8,12,16
Last column: 3,7,11,15,19

Center cells (for 5×4): 5,6,9,10
```

---

## Notes

- The DSL is case-sensitive
- All professions and traits are lowercase
- Cell indices are 0-based
- The `void` keyword is a special constant, not a string
- Direction offsets (dx, dy) use standard coordinate system where positive y is down
