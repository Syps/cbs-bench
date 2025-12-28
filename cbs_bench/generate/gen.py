from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from cbs_bench.models import PuzzleState, PuzzleCell, GeneratedPuzzle
from cbs_bench.prompts import GENERATOR_PROMPT_TEMPLATE
from cbs_bench.solve import ModelFactory

def create_puzzle(rows: int, cols: int, model_name: str) -> PuzzleState:
    prompt = GENERATOR_PROMPT_TEMPLATE.format(rows=rows, cols=cols)
    model = ModelFactory.create_model(model_name)
    memory = ChatMessageHistory()
    memory.add_message(SystemMessage(
            content=prompt,
            additional_kwargs={
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ))

    model = model.with_structured_output(GeneratedPuzzle, method="json_schema")
    result = model.invoke(memory.messages)



