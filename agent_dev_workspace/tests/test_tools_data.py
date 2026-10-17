from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_data import make_tools as make_data_tools


def test_check_missing_tool():
    session = SessionState()

    # Load dataset
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load check_missing_tool
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({})

    # Check known answer #2 from TEST_FIXTURE.md
    # Row 7 is empty text -> exactly 1 missing row
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]
    assert result["has_missing"] is True


def test_check_duplicates_tool():
    # Case 1: drop=False
    session1 = SessionState()
    premade_tools1 = make_premade_tools(session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools1 = make_data_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    result1 = check_duplicates_tool1.invoke({"drop": False})

    # Assert Case 1 values against Known Answer #3
    assert result1["duplicate_count"] == 1
    assert result1["duplicate_indices"] == [5, 6]
    assert result1["dropped"] is False
    assert len(session1.dataframe) == 8

    # Case 2: drop=True
    session2 = SessionState()
    premade_tools2 = make_premade_tools(session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools2 = make_data_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    result2 = check_duplicates_tool2.invoke({"drop": True})

    # Assert Case 2 values against Known Answer #3
    assert result2["remaining_rows"] == 6
    assert len(session2.dataframe) == 6
