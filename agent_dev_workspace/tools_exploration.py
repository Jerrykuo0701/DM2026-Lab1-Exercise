"""Tools for feature space exploration and similarity analysis."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_a: int,
        doc_index_b: int,
    ) -> dict:
        """Computes cosine similarity between two document vectors in the document-term matrix.

        Args:
            doc_index_a: Zero-based row index of the first document.
            doc_index_b: Zero-based row index of the second document.

        Returns:
            Dict containing result_id, doc_index_a, doc_index_b, and cosine_similarity.
        """
        if session.feature_matrix is None:
            return {"error": "Document-term matrix not built in session. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_index_a < 0 or doc_index_a >= n_docs:
            return {"error": f"doc_index_a ({doc_index_a}) is out of range [0, {n_docs - 1}]."}
        if doc_index_b < 0 or doc_index_b >= n_docs:
            return {"error": f"doc_index_b ({doc_index_b}) is out of range [0, {n_docs - 1}]."}

        a = session.feature_matrix[doc_index_a].toarray().ravel()
        b = session.feature_matrix[doc_index_b].toarray().ravel()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(a, b) / (norm_a * norm_b))

        sim_rounded = float(round(similarity, 4))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_a": int(doc_index_a),
            "doc_index_b": int(doc_index_b),
            "cosine_similarity": sim_rounded,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_a": doc_index_a, "doc_index_b": doc_index_b},
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20,
    ) -> dict:
        """Computes feature correlation heatmap over top terms by variance in the global document-term matrix.

        Args:
            top_k: Maximum number of top variance terms to include in the correlation matrix. Defaults to 20.

        Returns:
            Dict containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        raw_variances = np.var(dense_matrix, axis=0, ddof=0)

        # Sort term indices by variance descending
        sorted_indices = np.argsort(-raw_variances)
        k = min(top_k, len(session.feature_names))
        top_indices = sorted_indices[:k]

        top_terms = [session.feature_names[i] for i in top_indices]
        sub_matrix = dense_matrix[:, top_indices]

        # Compute correlation matrix across selected term columns
        if sub_matrix.shape[1] > 1:
            raw_corr = np.corrcoef(sub_matrix, rowvar=False)
        else:
            raw_corr = np.ones((sub_matrix.shape[1], sub_matrix.shape[1]))

        # Replace NaNs with 0.0 for constant columns and JSON safety
        cleaned_corr = np.nan_to_num(raw_corr, nan=0.0)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cleaned_corr, annot=True, fmt=".2f", xticklabels=top_terms, yticklabels=top_terms, ax=ax)
        ax.set_title(f"Top-{k} Term Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Convert to nested list of rounded floats
        corr_matrix_list = [
            [float(round(val, 4)) for val in row]
            for row in cleaned_corr
        ]

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_matrix_list,
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_k": top_k},
            summary,
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
