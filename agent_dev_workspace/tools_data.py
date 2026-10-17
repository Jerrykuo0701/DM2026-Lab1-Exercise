"""Tools for data loading, inspection, and cleaning."""

import os
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
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

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name",
    ) -> dict:
        """Computes text length statistics overall and per category, building a box plot visualization.

        Args:
            text_column: Name of column containing document text. Defaults to "text".
            category_column: Name of column containing category labels. Defaults to "category_name".

        Returns:
            Dict containing result_id, overall_stats, and category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in session dataframe."}

        text_lengths = df[text_column].fillna("").astype(str).apply(len)

        def get_stats_dict(series: pd.Series) -> dict:
            desc = series.describe()
            return {
                "count": int(desc["count"]),
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
            }

        overall_stats = get_stats_dict(text_lengths)

        category_stats = {}
        if category_column in df.columns:
            for cat, group in df.groupby(category_column):
                cat_lengths = group[text_column].fillna("").astype(str).apply(len)
                category_stats[str(cat)] = get_stats_dict(cat_lengths)

        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in df.columns:
            sns.boxplot(x=df[category_column], y=text_lengths, ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=text_lengths, ax=ax)
            ax.set_xlabel("Overall")

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Text Length Distribution by Category")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "category_stats": category_stats,
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "category_column": category_column},
            summary,
        )
        return summary

    @tool
    def tokenize_tool(
        text_column: str = "text",
    ) -> dict:
        """Tokenizes document text into lowercased unigrams.

        Args:
            text_column: Name of column containing document text. Defaults to "text".

        Returns:
            Dict containing result_id, n_documents, total_tokens, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in session dataframe."}

        tokens_list = [
            word_tokenize(str(text).lower()) if pd.notna(text) else []
            for text in df[text_column]
        ]

        total_tokens = int(sum(len(toks) for toks in tokens_list))
        session.artifacts["tokens"] = tokens_list
        session.dataframe["unigrams"] = tokens_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(tokens_list),
            "total_tokens": total_tokens,
            "sample_tokens": tokens_list[:3],
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def list_files_tool(
        subdirectory: str = "",
    ) -> dict:
        """Lists directories and files inside a relative subdirectory of the repository root.

        Args:
            subdirectory: Relative path from repo root to list files/directories from. Defaults to "".

        Returns:
            Dict containing result_id, directories, and files (or an error dict if path is invalid).
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_path = os.path.normpath(os.path.join(repo_root, subdirectory))

        if not os.path.exists(target_path):
            return {"error": f"Path '{subdirectory}' does not exist."}

        if not os.path.isdir(target_path):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        entries = os.listdir(target_path)
        directories = sorted([e for e in entries if os.path.isdir(os.path.join(target_path, e))])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(target_path, e))])

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: int = 100,
        random_state: int = 42,
    ) -> dict:
        """Subsamples rows from the working DataFrame in session.

        Args:
            n: Target number of rows to sample. Defaults to 100.
            random_state: Random seed for reproducible sampling. Defaults to 42.

        Returns:
            Dict containing result_id, n_before, n_after, and sampled_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        n_before = len(df)
        sample_n = min(n, n_before)

        sampled_df = df.sample(n=sample_n, random_state=random_state)
        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()]

        session.dataframe = sampled_df

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_before": n_before,
            "n_after": len(sampled_df),
            "sampled_indices": sampled_indices,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
    ) -> dict:
        """Inspects the first n_rows of the working DataFrame in session.

        Args:
            n_rows: Number of rows to display from top of DataFrame. Defaults to 5.

        Returns:
            Dict containing result_id, n_rows_shown, columns, and rows with text truncated to 100 characters.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        head_df = df.head(n_rows)

        columns = [str(col) for col in head_df.columns]
        rows = []

        for _, row in head_df.iterrows():
            row_dict = {}
            for col in head_df.columns:
                val = row[col]
                if pd.isna(val):
                    val = ""
                elif isinstance(val, (int, float, str, bool)):
                    val = val
                elif hasattr(val, "item"):
                    val = val.item()
                else:
                    val = str(val)

                if col == "text" and isinstance(val, str):
                    val = val[:100]

                row_dict[str(col)] = val
            rows.append(row_dict)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_rows_shown": len(rows),
            "columns": columns,
            "rows": rows,
        }

        session.store_result(
            "inspect_data_tool",
            {"n_rows": n_rows},
            summary,
        )
        return summary

    return [check_missing_tool, check_duplicates_tool, describe_data_tool, tokenize_tool, list_files_tool, sample_data_tool, inspect_data_tool]
