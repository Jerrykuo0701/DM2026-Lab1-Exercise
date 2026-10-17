"""Tools for data loading, inspection, and cleaning."""

from typing import Optional
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def check_missing_tool(
        text_column: str = "text",
    ) -> dict:
        """Checks the working DataFrame in session for missing or empty text values.

        Identifies rows where text is null/NaN or empty/whitespace-only, returning the count
        and specific row indices.

        Args:
            text_column: Name of the column containing document text to check.

        Returns:
            Dict containing result_id, total_rows, missing_count, missing_indices, and has_missing.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session dataframe."}

        df = session.dataframe
        mask = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")
        missing_count = int(mask.sum())
        missing_indices = df.index[mask].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "has_missing": missing_count > 0,
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        subset_column: Optional[str] = None,
        drop: bool = False,
    ) -> dict:
        """Checks for (and optionally drops) duplicate rows in the working DataFrame.

        Args:
            subset_column: Optional column name to check for duplicates. If None, considers all columns.
            drop: If True, drops duplicate rows from session.dataframe using keep=False.

        Returns:
            Dict containing result_id, initial_rows, duplicate_count, duplicate_indices,
            dropped, and remaining_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        subset = [subset_column] if subset_column is not None else None
        initial_rows = len(df)

        duplicate_count = int(df.duplicated(subset=subset, keep="first").sum())
        duplicate_indices = df.index[df.duplicated(subset=subset, keep=False)].tolist()

        if drop:
            session.dataframe = df.drop_duplicates(subset=subset, keep=False).reset_index(drop=True)

        remaining_rows = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "initial_rows": initial_rows,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "remaining_rows": remaining_rows,
        }

        session.store_result(
            "check_duplicates_tool",
            {"subset_column": subset_column, "drop": drop},
            summary,
        )
        return summary

    return [check_missing_tool, check_duplicates_tool]
