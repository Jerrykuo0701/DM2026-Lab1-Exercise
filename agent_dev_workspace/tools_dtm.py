"""Tools for Document-Term Matrix (DTM) construction and exploration."""

from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    @tool
    def term_frequency_tool(
        top_k: int = 10,
    ) -> dict:
        """Aggregates term frequencies across all documents in the Document-Term Matrix (DTM).

        Reads from session.feature_matrix and session.feature_names, sums frequencies across
        all documents, and stores full term frequency breakdown.

        Args:
            top_k: Number of top frequent terms to include in the returned summary dictionary.

        Returns:
            Dict containing result_id, total_terms, total_word_count, and top_terms mapping.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix (DTM) not built yet. Run build_dtm_tool first."}

        # Sum column counts across all documents and flatten to 1-D array
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_names = list(session.feature_names)

        # Create DataFrame sorted descending by frequency
        df_freq = pd.DataFrame({
            "term": term_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Store artifact mapping
        freq_dict = dict(zip(df_freq["term"], df_freq["frequency"].astype(int)))
        session.artifacts["term_frequencies"] = freq_dict

        total_terms = len(term_names)
        total_word_count = int(counts.sum())

        # Extract top-K preview
        top_terms = dict(zip(df_freq["term"].head(top_k), df_freq["frequency"].head(top_k).astype(int)))

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": total_terms,
            "total_word_count": total_word_count,
            "top_terms": top_terms,
        }

        session.store_result("term_frequency_tool", {"top_k": top_k}, summary, full_report=df_freq)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> dict:
        """Visualizes a raw positional slice of the Document-Term Matrix (DTM) as a heatmap.

        Reads positional slice from session.feature_matrix and session.feature_names without
        sorting or scoring, constructs an annotated heatmap, sets session.pending_figure, and
        returns the matrix slice and labels.

        Args:
            n_terms: Number of terms (columns) to include from index 0.
            n_documents: Number of documents (rows) to include from index 0.

        Returns:
            Dict containing result_id, n_documents_sliced, n_terms_sliced, term_labels,
            document_labels, and sliced_matrix nested list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-Term Matrix (DTM) not built yet. Run build_dtm_tool first."}

        # Raw positional slice cast to int
        slice_matrix = session.feature_matrix[0:n_documents, 0:n_terms].toarray().astype(int)
        term_labels = list(session.feature_names[0:n_terms])
        doc_labels = [f"Doc_{i}" for i in range(slice_matrix.shape[0])]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(6, len(term_labels) * 0.8), max(4, len(doc_labels) * 0.5)))
        sns.heatmap(
            slice_matrix,
            xticklabels=term_labels,
            yticklabels=doc_labels,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"DTM Slice Heatmap ({slice_matrix.shape[0]} Docs x {slice_matrix.shape[1]} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents_sliced": slice_matrix.shape[0],
            "n_terms_sliced": slice_matrix.shape[1],
            "term_labels": term_labels,
            "document_labels": doc_labels,
            "sliced_matrix": slice_matrix.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
