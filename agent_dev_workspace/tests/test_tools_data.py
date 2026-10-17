import pytest
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


def test_describe_data_tool():
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})

    # Assert Known Answer #18 from TEST_FIXTURE.md
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)

    cat_stats = result["category_stats"]
    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == pytest.approx(0.0, abs=1e-3)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == pytest.approx(29.0, abs=1e-3)

    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0, abs=1e-3)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == pytest.approx(30.0, abs=1e-3)

    assert session.pending_figure is not None


def test_tokenize_tool():
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    # Assert Known Answer #10 from TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert session.artifacts["tokens"] == expected_tokens
    assert list(session.dataframe["unigrams"]) == expected_tokens
    assert result["total_tokens"] == 27
    assert result["n_documents"] == 8


def test_list_files_tool():
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Assert Known Answer #11 from TEST_FIXTURE.md
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

    # Test error handling for non-existent subdirectory
    error_result = list_files_tool.invoke({"subdirectory": "no_such_dir"})
    assert "error" in error_result
