import math
import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_dtm import make_tools as make_dtm_tools
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


def test_reduce_dimensions_pca():
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    coords = result["coordinates"]
    evr = result["explained_variance_ratio"]

    # Known Answer #7 from TEST_FIXTURE.md
    assert coords[0] == pytest.approx([2.3669, 0.8871], abs=1e-3)
    assert coords[7] == pytest.approx([0.0894, -1.4822], abs=1e-3)
    assert evr == pytest.approx([0.7532, 0.1965], abs=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    # Note: Exact coordinate values for Known Answer #8 are not asserted here because
    # t-SNE coordinate output depends on library/BLAS/platform versions.
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    args = {"method": "tsne", "perplexity": 2, "random_state": 42}
    result1 = reduce_dimensions_tool.invoke(args)

    coords1 = result1["coordinates"]
    assert len(coords1) == 8
    for row in coords1:
        assert len(row) == 2
        assert all(isinstance(x, float) and math.isfinite(x) for x in row)

    assert session.pending_figure is not None

    # Test determinism given identical random_state
    result2 = reduce_dimensions_tool.invoke(args)
    coords2 = result2["coordinates"]
    assert coords1 == coords2


def test_reduce_dimensions_umap():
    # Note: Exact coordinate values for Known Answer #9 are not asserted here because
    # UMAP coordinate output depends on library/BLAS/platform versions.
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    args = {"method": "umap", "n_neighbors": 3, "random_state": 42}
    result1 = reduce_dimensions_tool.invoke(args)

    coords1 = result1["coordinates"]
    assert len(coords1) == 8
    for row in coords1:
        assert len(row) == 2
        assert all(isinstance(x, float) and math.isfinite(x) for x in row)

    assert session.pending_figure is not None

    # Test determinism given identical random_state
    result2 = reduce_dimensions_tool.invoke(args)
    coords2 = result2["coordinates"]
    assert coords1 == coords2
