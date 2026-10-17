import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
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

    # Filter by variance
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert Known Answer #5 from TEST_FIXTURE.md
    variances = result["variances"]
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)

    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms"] == ["always"]
