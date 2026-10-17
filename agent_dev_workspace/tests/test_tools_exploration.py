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
