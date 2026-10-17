"""Tools for feature selection and filtering."""

import numpy as np
import pandas as pd
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.01,
    ) -> dict:
        """Flags near-constant terms by variance for removal from the global document-term matrix.

        Args:
            threshold: Minimum population variance threshold. Terms below this threshold are flagged for removal. Defaults to 0.01.

        Returns:
            Dict containing result_id, threshold, n_terms, n_kept, n_removed, kept_terms, removed_terms, and variances.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built in session. Run build_dtm_tool first."}

        # Dense array for calculating column variances across documents
        dense_matrix = session.feature_matrix.toarray()
        
        # Calculate population variance (ddof=0) across rows for each term column
        raw_variances = np.var(dense_matrix, axis=0, ddof=0)

        kept_terms = []
        removed_terms = []
        variances_dict = {}

        for term, var in zip(session.feature_names, raw_variances):
            var_float = float(var)
            variances_dict[term] = var_float
            if var_float >= threshold:
                kept_terms.append(term)
            else:
                removed_terms.append(term)

        # Build full report DataFrame for visualize_result_tool
        # First column: term, Second column: variance, sorted descending
        report_data = [
            {"term": term, "variance": float(var)}
            for term, var in zip(session.feature_names, raw_variances)
        ]
        full_report_df = pd.DataFrame(report_data).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": float(threshold),
            "n_terms": int(len(session.feature_names)),
            "n_kept": int(len(kept_terms)),
            "n_removed": int(len(removed_terms)),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variances": variances_dict,
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=full_report_df,
        )
        return summary

    return [variance_filter_tool]
