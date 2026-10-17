"""Tools for Document-Term Matrix (DTM) construction and exploration."""

from typing import List, Optional, Union
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: Union[int, float] = 1.0,
        min_df: Union[int, float] = 1,
        max_features: Optional[int] = None,
    ) -> dict:
        """Build a Document-Term Matrix (DTM) from document text using CountVectorizer.

        Stores the fitted matrix in session.feature_matrix, vocabulary in session.feature_names,
        and vectorizer artifact in session.artifacts['count_vectorizer']. Calculates and reports
        matrix dimensions and sparsity percentage.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams, e.g., [1, 1].
            max_df: Ignore terms with document frequency strictly higher than threshold.
            min_df: Ignore terms with document frequency strictly lower than threshold.
            max_features: Build vocabulary using top max_features ordered by term frequency.

        Returns:
            Dict containing result_id, n_documents, vocabulary_size, non_zero, total_elements,
            and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or missing 'text' column in session."}

        # Convert ngram_range list to tuple for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1])

        # Fill missing text values with empty strings
        texts = session.dataframe["text"].fillna("").tolist()

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].tolist())

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range}, summary)
        return summary

    return [build_dtm_tool]
