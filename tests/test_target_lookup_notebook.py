import ast
import json
from pathlib import Path

import pytest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "qal/examples/jupyter-notebook/target_preset_lookup_example.ipynb"
)


@pytest.mark.parametrize("cell_id", ["7b6bb3f4", "c3f0e030"])
@pytest.mark.parametrize("query", ["IG", "", "RCS-ICG", "RCS-ICG-ST01-QUEL03"])
def test_lookup_notebook_handles_queries_on_rerun(cell_id, query, capsys):
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    namespace = {}
    exec(cells["c279ee15"], namespace)

    def run_query(value):
        tree = ast.parse(cells[cell_id])
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TARGET_QUERY"
                for target in node.targets
            ):
                node.value = ast.Constant(value=value)
        exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)

    # Populate the result first so failed reruns cannot reuse a previous preset.
    run_query("RCS-ICG-ST01-QUEL03")
    capsys.readouterr()
    run_query(query)
    output = capsys.readouterr()

    assert output.err == ""
    if query == "IG":
        assert output.out == "No product preset matches 'IG'\n"
    elif query == "":
        assert output.out == "query must be a non-empty serial number or SKU fragment\n"
    elif query == "RCS-ICG" and cell_id == "7b6bb3f4":
        assert output.out.startswith("Ambiguous target query 'RCS-ICG'")
    else:
        assert "RCS-ICG-ST01-QUEL03" in output.out
        if cell_id == "7b6bb3f4":
            assert namespace["preset"]["tag"] == "icg"
        else:
            assert namespace["matches"]
        return

    if cell_id == "7b6bb3f4":
        assert namespace["preset"] is None
    else:
        assert namespace["matches"] == []
