from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from click import echo
import time
import pdb

from pygments.lexers import q

from cbs_bench.models import PuzzleState, PuzzleCell, GeneratedPuzzle, GeneratedPuzzleCell, Status
from cbs_bench.prompts import GENERATE_PUZZLE_SYSTEM_PROMPT
from cbs_bench.generate.validate import PuzzleValidator
from cbs_bench.solve import ModelFactory


def generated_to_puzzle_state(generated: GeneratedPuzzle, rows: int, cols: int) -> PuzzleState:
    puzzle_state = []

    if rows * cols != len(generated.cells):
        raise ValueError(
            f"With rows={rows} and cols={cols}, generated array length should be {rows * cols}, "
            f"but it was {len(generated.cells)} instead."
        )

    for index, generated_cell in enumerate(generated.cells):
        row = index // cols

        if len(puzzle_state) - 1 != row:
            puzzle_state.append([])

        status = Status.UNKNOWN

        if len(generated_cell.paths) == 0:
            status = Status.CRIMINAL if generated_cell.is_criminal else Status.INNOCENT

        puzzle_state[row].append(
            PuzzleCell(
                name=generated_cell.name,
                profession=generated_cell.profession,
                gender=generated_cell.gender,
                orig_hint=generated_cell.hint_dsl,
                clue="",
                status=status,
                is_criminal=generated_cell.is_criminal,
                paths=generated_cell.paths,
            )
        )


    return puzzle_state


def create_puzzle(*, rows: int, cols: int, model_name: str) -> PuzzleState:
    prompt = GENERATE_PUZZLE_SYSTEM_PROMPT.format(
        rows=rows,
        columns=cols,
        expected_arr_length=rows * cols,
        last_index=rows * cols - 1,
    )
    model = ModelFactory.create_model(model_name)
    memory = ChatMessageHistory()
    memory.add_message(SystemMessage(
        content=prompt,
        additional_kwargs={
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ))

    model = model.with_structured_output(GeneratedPuzzle, method="json_schema")

    print(f"Invoking {model_name} with message beginning '{memory.messages[0].content[-120:]}...'")
    start = time.time()
    result = model.invoke(memory.messages)
    print(f"Done. Took {time.time() - start} seconds.")
    # pdb.set_trace()
    puzzle_state = generated_to_puzzle_state(result, rows, cols)

    validator = PuzzleValidator(puzzle_state)
    validation_result = validator.validate()

    if not validation_result.valid:
        echo(
            message=validation_result.invalid_message,
            err=True
        )

    return puzzle_state


# test_puzzle = GeneratedPuzzle(
#     cells=[
#         GeneratedPuzzleCell(name='Aaron', profession='guard', gender='male',
#                             hint_dsl='number_of_traits_in_unit(unit(row,0),criminal,2)',
#                             is_criminal=True, paths=[[4]]),
#         GeneratedPuzzleCell(name='Beth', profession='clerk', gender='female', hint_dsl='has_trait(5,innocent)',
#                             is_criminal=False, paths=[[]]),
#         GeneratedPuzzleCell(name='Carlos', profession='coder', gender='male', hint_dsl='has_trait(8,criminal)',
#                             is_criminal=True, paths=[[0, 1]]),
#         GeneratedPuzzleCell(name='Dana', profession='cop', gender='female',
#                             hint_dsl='number_of_traits_in_unit(unit(col,0),criminal,2)', is_criminal=True, paths=[[5]]),
#         GeneratedPuzzleCell(name='Ethan', profession='judge', gender='male', hint_dsl='has_trait(0,criminal)',
#                             is_criminal=True, paths=[[5]]),
#         GeneratedPuzzleCell(name='Fiona', profession='teacher', gender='female',
#                             hint_dsl='number_of_traits_in_unit(unit(row,1),criminal,2)', is_criminal=False,
#                             paths=[[1]]),
#         GeneratedPuzzleCell(name='Greg', profession='doctor', gender='male', hint_dsl='has_trait(7,innocent)',
#                             is_criminal=False, paths=[[3, 4]]),
#         GeneratedPuzzleCell(name='Hana', profession='builder', gender='female',
#                             hint_dsl='more_traits_in_unit_than_unit(unit(row,0),unit(row,2),criminal)',
#                             is_criminal=False, paths=[[6]]),
#         GeneratedPuzzleCell(name='Ivan', profession='pilot', gender='male',
#                             hint_dsl='both_traits_are_neighbors_in_unit(unit(row,2),innocent)', is_criminal=True,
#                             paths=[[2]])])
#
# pdb.set_trace()
# puzzle_state = generated_to_puzzle_state(test_puzzle, 3, 3)
# validator = PuzzleValidator(puzzle_state)
# validation_result = validator.validate()
