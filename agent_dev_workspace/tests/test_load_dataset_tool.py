import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools


def test_load_dataset_tool():
    session = SessionState()
    tools = make_tools(session)
    
    load_dataset_tool = next(t for t in tools if t.name == "load_dataset_tool")
    
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Assert return values based on Known Answer 1 in TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
    
    # Assert session state updates
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
