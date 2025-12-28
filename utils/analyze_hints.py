#!/usr/bin/env python3
"""
Maintenance script to extract hint data from cached JSON files.
Iterates over all JSON files in .cache folder and extracts orig_hint and hint pairs.
Also extracts unique function signatures and enums from the DSL.
"""

import json
import csv
import os
import re
from pathlib import Path
from typing import List, Tuple, Set


class DSLParser:
    """Parser for extracting function signatures and enums from DSL expressions."""

    def __init__(self):
        self.signatures = {}  # signature -> (example orig_hint, hint)
        self.enums = {}  # enum -> example orig_hint
        self.current_orig_hint = None
        self.current_hint = None

    def parse_orig_hint(self, orig_hint: str, hint: str):
        """Parse an orig_hint expression and extract signatures and enums."""
        self.current_orig_hint = orig_hint
        self.current_hint = hint
        self.parse_expression(orig_hint)

    def parse_expression(self, expr: str) -> Tuple[str, str]:
        """
        Parse a DSL expression and return (value, type).
        Type can be 'int', 'enum', or a function name.
        """
        expr = expr.strip()

        # Check if it's a number
        if re.match(r'^-?\d+$', expr):
            return expr, 'int'

        # Check if it's a function call
        if '(' in expr:
            # Extract function name
            func_name = expr[:expr.index('(')]
            # Extract arguments
            args_str = expr[expr.index('(') + 1:expr.rindex(')')]
            args = self.split_args(args_str)

            # Parse each argument and build signature
            arg_types = []
            for arg in args:
                _, arg_type = self.parse_expression(arg)
                arg_types.append(arg_type)

            # Store the signature with example if not already present
            signature = f"{func_name}({','.join(arg_types)})"
            if signature not in self.signatures:
                self.signatures[signature] = (self.current_orig_hint, self.current_hint)

            return expr, func_name

        # Otherwise it's an enum
        if expr not in self.enums:
            self.enums[expr] = self.current_orig_hint
        return expr, 'enum'

    def split_args(self, args_str: str) -> List[str]:
        """Split arguments by comma, respecting nested parentheses."""
        args = []
        current_arg = []
        depth = 0

        for char in args_str:
            if char == ',' and depth == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current_arg.append(char)

        if current_arg:
            args.append(''.join(current_arg).strip())

        return args


def main():
    cache_dir = Path(".cache")
    output_dir = Path("out")
    output_file = output_dir / "all_hints.csv"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Collect all hint pairs
    hint_pairs = []

    # Iterate over all JSON files in .cache
    for json_file in cache_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both array and single object structures
            if isinstance(data, list):
                items = data
            else:
                items = [data]

            # Extract orig_hint and hint pairs
            for item in items:
                if isinstance(item, dict) and "orig_hint" in item:
                    orig_hint = item.get("orig_hint", "")
                    hint = item.get("hint", "")
                    hint_pairs.append((orig_hint, hint))

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process {json_file}: {e}")
            continue

    # Write hints to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["orig_hint", "hint"])
        writer.writerows(hint_pairs)

    print(f"Extracted {len(hint_pairs)} hint pairs")
    print(f"Output written to: {output_file}")

    # Parse DSL expressions to extract signatures and enums
    parser = DSLParser()
    for orig_hint, hint in hint_pairs:
        if orig_hint:
            parser.parse_orig_hint(orig_hint, hint)

    print(f"\nExtracted {len(parser.signatures)} unique function signatures")
    print(f"Extracted {len(parser.enums)} unique enums")

    # Write signatures to CSV
    signatures_file = output_dir / "dsl_signatures.csv"
    with open(signatures_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["signature", "example", "hint"])
        for sig in sorted(parser.signatures.keys()):
            orig_hint, hint = parser.signatures[sig]
            writer.writerow([sig, orig_hint, hint])
    print(f"Signatures written to: {signatures_file}")

    # Write enums to CSV
    enums_file = output_dir / "dsl_enums.csv"
    with open(enums_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["enum", "example"])
        for enum in sorted(parser.enums.keys()):
            writer.writerow([enum, parser.enums[enum]])
    print(f"Enums written to: {enums_file}")


if __name__ == "__main__":
    main()
