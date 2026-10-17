import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_dtm import make_tools as make_dtm_tools


def test_build_dtm_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load build_dtm_tool from student implementation
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    result = build_dtm_tool.invoke({})

    # Check known answer #4 from TEST_FIXTURE.md
    # Vocabulary: alpha, always, beta, delta, gamma (5 terms)
    # Sparsity: 21 non-zero out of 40 total elements -> 47.5% sparsity
    assert result["n_documents"] == 8
    assert result["vocabulary_size"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check session attributes set by build_dtm_tool
    assert session.feature_matrix is not None
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    session = SessionState()

    # Load dataset
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load dtm tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Run build_dtm_tool first to populate session.feature_matrix and session.feature_names
    build_dtm_tool.invoke({})

    result = term_frequency_tool.invoke({})

    # Check known answer #14 from TEST_FIXTURE.md
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_frequencies = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }

    assert result["total_terms"] == 5
    assert result["total_word_count"] == 27
    assert result["top_terms"] == expected_frequencies

    # Check session artifact
    assert session.artifacts["term_frequencies"] == expected_frequencies


def test_dtm_heatmap_tool():
    session = SessionState()

    # Load dataset
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Load dtm tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Run build_dtm_tool first
    build_dtm_tool.invoke({})

    # 1. Default call (returns full 8x5 matrix slice)
    result_default = dtm_heatmap_tool.invoke({})
    assert result_default["n_documents_sliced"] == 8
    assert result_default["n_terms_sliced"] == 5
    assert result_default["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert result_default["sliced_matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # 2. Slice call with n_terms=2, n_documents=3
    result_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert result_sliced["n_documents_sliced"] == 3
    assert result_sliced["n_terms_sliced"] == 2
    assert result_sliced["term_labels"] == ["alpha", "always"]
    assert result_sliced["sliced_matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
