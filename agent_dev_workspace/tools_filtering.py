"""Tools for feature selection and filtering."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    def _correlation_report(method: str, target_class: str, top_k: int) -> dict:
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "Document-term matrix or labels not built in session. Run build_dtm_tool first."}

        labels_list = list(session.labels)
        if target_class not in labels_list:
            return {"error": f"Target class '{target_class}' not found in session labels."}

        y = (np.array(labels_list) == target_class).astype(int)
        dense_matrix = session.feature_matrix.toarray()

        correlations = {}
        raw_corrs = []
        corr_func = pearsonr if method == "pearson" else spearmanr
        col_name = "pearson_r" if method == "pearson" else "spearman_r"

        for term, i in zip(session.feature_names, range(dense_matrix.shape[1])):
            col = dense_matrix[:, i]
            if np.all(col == col[0]):
                r_val = 0.0
            else:
                r, _ = corr_func(col, y)
                r_val = 0.0 if np.isnan(r) else float(r)

            r_rounded = float(round(r_val, 4))
            correlations[term] = r_rounded
            raw_corrs.append((term, r_val, r_rounded))

        sorted_by_abs = sorted(raw_corrs, key=lambda x: abs(x[1]), reverse=True)
        top_terms = [item[0] for item in sorted_by_abs[:top_k]]
        top_correlations = {term: correlations[term] for term in top_terms if term in correlations}

        report_data = [{"term": term, col_name: r_val} for term, r_val, _ in raw_corrs]
        full_report_df = pd.DataFrame(report_data).sort_values(by=col_name, ascending=False).reset_index(drop=True)

        tool_name = f"{method}_filter_tool"
        result_id = session.next_result_id(f"{method}_filter")

        # Full summary stored internally in session.results_store
        full_summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_terms": int(len(session.feature_names)),
            "correlations": correlations,
            "top_terms": top_terms,
        }

        session.store_result(
            tool_name,
            {"target_class": target_class, "top_k": top_k},
            full_summary,
            full_report=full_report_df,
        )

        # Truncated return summary for LLM context optimization
        return_summary = {
            "result_id": result_id,
            "target_class": target_class,
            "n_terms": int(len(session.feature_names)),
            "top_terms": top_terms,
            "correlations": top_correlations,
            "note": f"Correlations dictionary filtered to top {len(top_terms)} terms by absolute magnitude to optimize context size. Full report stored under result_id '{result_id}'.",
        }
        return return_summary

    @tool
    def variance_filter_tool(
        threshold: float = 0.01,
    ) -> dict:
        """Flags near-constant terms by variance for removal from the global document-term matrix.

        Args:
            threshold: Minimum population variance threshold. Terms below this threshold are flagged for removal. Defaults to 0.01.

        Returns:
            Dict containing result_id, threshold, n_terms, n_kept, n_removed, kept_terms, removed_terms, variances, and note.
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
        report_data = [
            {"term": term, "variance": float(var)}
            for term, var in zip(session.feature_names, raw_variances)
        ]
        full_report_df = pd.DataFrame(report_data).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        
        # Store full summary in session
        full_summary = {
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
            full_summary,
            full_report=full_report_df,
        )

        # Prepare truncated response for LLM context
        truncated_kept = kept_terms[:30]
        truncated_removed = removed_terms[:30]
        displayed_terms = set(truncated_kept + truncated_removed)
        truncated_variances = {term: variances_dict[term] for term in displayed_terms if term in variances_dict}

        is_truncated = len(kept_terms) > 30 or len(removed_terms) > 30
        note = (
            f"Lists truncated to first 30 terms ({len(truncated_kept)} kept, {len(truncated_removed)} removed displayed out of {len(kept_terms)} kept and {len(removed_terms)} total removed). Full details saved in session store under result_id '{result_id}'."
            if is_truncated
            else "All kept and removed terms displayed."
        )

        return_summary = {
            "result_id": result_id,
            "threshold": float(threshold),
            "n_terms": int(len(session.feature_names)),
            "n_kept": int(len(kept_terms)),
            "n_removed": int(len(removed_terms)),
            "kept_terms": truncated_kept,
            "removed_terms": truncated_removed,
            "variances": truncated_variances,
            "note": note,
        }
        return return_summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        top_k: int = 20,
    ) -> dict:
        """Computes Pearson linear correlation between term frequencies and a target class.

        Args:
            target_class: The target category label to correlate against.
            top_k: Number of top terms by absolute correlation to return in summary. Defaults to 20.

        Returns:
            Dict containing result_id, target_class, n_terms, correlations, top_terms, and note.
        """
        return _correlation_report("pearson", target_class, top_k)

    @tool
    def spearman_filter_tool(
        target_class: str,
        top_k: int = 20,
    ) -> dict:
        """Computes Spearman rank correlation between term frequencies and a target class.

        Args:
            target_class: The target category label to correlate against.
            top_k: Number of top terms by absolute correlation to return in summary. Defaults to 20.

        Returns:
            Dict containing result_id, target_class, n_terms, correlations, top_terms, and note.
        """
        return _correlation_report("spearman", target_class, top_k)

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
