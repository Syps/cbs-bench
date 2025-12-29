"""Mainly used for testing"""

from cbs_bench.generate.gen import create_puzzle


create_puzzle(
    rows=3,
    cols=3,
    model_name="gpt-5"
)