"""Tools for dimensionality reduction and label binarization."""

from typing import List, Optional
import pandas as pd
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes document category labels into a binary matrix artifact.

        Converts session.dataframe['category_name'] into a binary indicator DataFrame
        stored in session.artifacts['binarized_labels'], returning class summary metrics
        and sample rows.

        Returns:
            Dict containing result_id, classes, n_documents, class_counts, and first_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' not found in session dataframe."}

        category_series = session.dataframe["category_name"]
        binarized_df = pd.get_dummies(category_series, dtype=int)

        session.artifacts["binarized_labels"] = binarized_df

        classes = sorted([str(col) for col in binarized_df.columns])
        n_documents = len(binarized_df)
        class_counts = {str(col): int(binarized_df[col].sum()) for col in binarized_df.columns}

        first_rows = [
            {str(col): int(val) for col, val in row.items()}
            for _, row in binarized_df.head(3).iterrows()
        ]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "n_documents": n_documents,
            "class_counts": class_counts,
            "first_rows": first_rows,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [binarize_labels_tool]
