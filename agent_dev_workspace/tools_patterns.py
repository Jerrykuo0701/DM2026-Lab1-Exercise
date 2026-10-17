"""Tools for frequent pattern mining."""

import os
import tempfile
from typing import List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    def apply_vocabulary_filtering(counts: np.ndarray, filtering_method: str) -> np.ndarray:
        """Filters columns of a document-term matrix using percentile bands."""
        if counts.shape[1] == 0:
            return np.array([], dtype=bool)

        if filtering_method == "variance":
            variances = np.var(counts, axis=0, ddof=1) if counts.shape[0] > 1 else np.var(counts, axis=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            return (variances >= p5) & (variances <= p95)
        elif filtering_method == "term_frequency":
            freqs = np.sum(counts, axis=0)
            p5 = np.percentile(freqs, 5)
            p95 = np.percentile(freqs, 95)
            return (freqs >= p5) & (freqs <= p95)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(counts).toarray()
            mean_tfidf = np.mean(tfidf, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            return mean_tfidf >= p20
        else:
            return np.array([False] * counts.shape[1], dtype=bool)

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 1,
    ) -> dict:
        """Mines frequent patterns for a specific document category using PAMI.

        Args:
            category_name: Name of the category to filter documents by.
            filtering_method: Method to filter vocabulary ("variance", "tfidf", or "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", or "maxfpgrowth"). Defaults to "fpgrowth".
            min_sup: Minimum support count for fpgrowth/maxfpgrowth. Defaults to 1.
            k: Number of top patterns for topk algorithm. Defaults to 1.

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            n_terms_kept, kept_terms, and patterns list.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' not found in session dataframe."}

        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"error": f"Category '{category_name}' not found in session dataframe."}

        texts = df_cat["text"].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(stop_words="english")
        try:
            counts = vectorizer.fit_transform(texts).toarray()
            terms = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            counts = np.empty((len(texts), 0))
            terms = np.array([])

        if counts.shape[1] == 0:
            kept_mask = np.array([], dtype=bool)
        else:
            if filtering_method not in ("variance", "tfidf", "term_frequency"):
                return {"error": f"Unknown filtering_method '{filtering_method}'."}
            kept_mask = apply_vocabulary_filtering(counts, filtering_method)

        kept_terms = terms[kept_mask].tolist() if len(terms) > 0 else []
        n_terms_kept = len(kept_terms)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        if n_terms_kept == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_terms_kept": 0,
                "kept_terms": [],
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        kept_counts = counts[:, kept_mask]
        binary_df = pd.DataFrame(kept_counts >= 1, columns=kept_terms)

        non_empty_mask = binary_df.sum(axis=1) > 0
        binary_df = binary_df[non_empty_mask]

        if binary_df.empty:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_terms_kept": n_terms_kept,
                "kept_terms": kept_terms,
                "patterns": [],
                "note": "no non-empty transactions for kept terms",
            }
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        patterns = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "transactions.txt")

            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(binary_df.astype(int))
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                obj = FPGrowth.FPGrowth(db_file, minSup=min_sup)
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                obj = FAE.FAE(db_file, k=k)
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            obj.mine()
            patterns_df = obj.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    p_str = str(row["Patterns"]).strip()
                    s_val = row["Support"]
                    items = [str(x) for x in p_str.split() if str(x)]
                    try:
                        support = int(s_val)
                    except (ValueError, TypeError):
                        support = 0
                    patterns.append({"pattern": items, "support": support})

        session.artifacts[artifact_key] = patterns
        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_terms_kept": n_terms_kept,
            "kept_terms": kept_terms,
            "patterns": patterns,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
