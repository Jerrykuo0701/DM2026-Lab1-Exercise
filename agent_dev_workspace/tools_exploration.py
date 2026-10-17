"""Tools for feature space exploration and similarity analysis."""

import numpy as np
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

    return [cosine_similarity_tool]
