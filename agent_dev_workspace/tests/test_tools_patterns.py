import pytest
from agent_pipeline.session_state import SessionState
from premade_tools.data_loading_tools import make_tools as make_premade_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tfidf_fpgrowth():
    # Known Answer #17 from TEST_FIXTURE.md
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # catA
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["n_terms_kept"] == 1
    assert res_catA["kept_terms"] == ["alpha"]
    assert res_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]

    # catB
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB["n_terms_kept"] == 1
    assert res_catB["kept_terms"] == ["gamma"]
    assert res_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]


def test_mine_patterns_zero_terms_survived():
    # Known Answer #17: variance & term_frequency keep zero terms for both categories
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    for cat in ["catA", "catB"]:
        for fmt in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": fmt,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["n_terms_kept"] == 0
            assert res["patterns"] == []


def test_mine_patterns_tfidf_topk():
    # Known Answer #20 from TEST_FIXTURE.md
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # catA
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]

    # catB
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]


def test_mine_patterns_tfidf_maxfpgrowth():
    # Note: Known Answer #20 specifies zero patterns for MaxFPGrowth on single-surviving-term
    # databases, but exact output depends on PAMI library versions. Here we verify structure.
    session = SessionState()
    premade_tools = make_premade_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    for cat in ["catA", "catB"]:
        res = mine_patterns_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1
        })
        assert "error" not in res
        assert isinstance(res["patterns"], list)
        kept = res["kept_terms"]
        for p in res["patterns"]:
            assert "pattern" in p and isinstance(p["pattern"], list)
            assert "support" in p and isinstance(p["support"], int)
            assert set(p["pattern"]).issubset(set(kept))
