import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_reduction import make_tools as make_reduction_tools


def test_binarize_labels_tool():
    # Test error handling when no dataframe is loaded
    fresh_session = SessionState()
    reduction_tools_no_data = make_reduction_tools(fresh_session)
    binarize_tool_no_data = next(t for t in reduction_tools_no_data if t.name == "binarize_labels_tool")
    err_result = binarize_tool_no_data.invoke({})
    assert "error" in err_result

    # Test standard usage with loaded dataset
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({})

    assert result["classes"] == ["catA", "catB"]
    assert result["n_documents"] == 8
    assert result["class_counts"] == {"catA": 4, "catB": 4}
    assert "binarized_labels" in session.artifacts
