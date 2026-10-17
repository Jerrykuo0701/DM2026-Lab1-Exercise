import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
    session = SessionState()

    # Load dataset
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Assert Known Answer #16 from TEST_FIXTURE.md
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_3_4 = cosine_similarity_tool.invoke({"doc_index_a": 3, "doc_index_b": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_7 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 7})
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)

    # Test error on out-of-range index
    error_res = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 99})
    assert "error" in error_res


def test_feature_correlation_matrix_tool():
    session = SessionState()

    # Load dataset
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load feature_correlation_matrix_tool
    exploration_tools = make_exploration_tools(session)
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    result = feature_correlation_matrix_tool.invoke({})

    # Assert Known Answer #19 from TEST_FIXTURE.md
    terms = result["terms"]
    assert terms == ["alpha", "gamma", "delta", "beta", "always"]

    corr_matrix = result["correlation_matrix"]

    # Diagonal elements must be 1.0
    for i in range(len(terms)):
        assert corr_matrix[i][i] == pytest.approx(1.0, abs=1e-3)

    # Helper mapping term pair to correlation value
    def get_corr(t1: str, t2: str) -> float:
        i1 = terms.index(t1)
        i2 = terms.index(t2)
        return corr_matrix[i1][i2]

    assert get_corr("alpha", "beta") == pytest.approx(0.8885, abs=1e-3)
    assert get_corr("gamma", "delta") == pytest.approx(0.8307, abs=1e-3)
    assert get_corr("alpha", "gamma") == pytest.approx(-0.5718, abs=1e-3)
    assert get_corr("alpha", "delta") == pytest.approx(-0.6882, abs=1e-3)
    assert get_corr("always", "delta") == pytest.approx(0.3780, abs=1e-3)

    assert session.pending_figure is not None
