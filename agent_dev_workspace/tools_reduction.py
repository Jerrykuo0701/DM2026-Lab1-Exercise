"""Tools for dimensionality reduction and label binarization."""

from typing import List, Literal, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduces document-term matrix dimensions via PCA, t-SNE, or UMAP, producing a scatter plot.

        Args:
            method: Dimensionality reduction technique ("pca", "tsne", or "umap"). Defaults to "pca".
            n_components: Number of target dimensions. Must be >= 2. Defaults to 2.
            random_state: Seed for reproducibility. Defaults to 42.
            perplexity: Perplexity parameter for t-SNE only. Defaults to 30.0.
            n_neighbors: Number of neighbors parameter for UMAP only. Defaults to 15.

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (PCA only).
        """
        if session.feature_matrix is None or session.labels is None:
            return {"error": "Document-term matrix or labels not built in session."}

        if n_components < 2:
            return {"error": "n_components must be >= 2."}

        X = session.feature_matrix.toarray()
        labels = np.array(session.labels)

        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = [
                round(float(v), 4) for v in model.explained_variance_ratio_
            ]
        elif method == "tsne":
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
        elif method == "umap":
            import umap
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=int(n_neighbors),
                random_state=random_state,
            )
            coords = model.fit_transform(X)
        else:
            return {"error": f"Unknown method '{method}'."}

        coords_list = [
            [round(float(val), 4) for val in row] for row in coords
        ]
        session.artifacts["reduced_coordinates"] = coords_list

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = sorted(list(set(labels)))
        for cat in unique_labels:
            mask = labels == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(cat), alpha=0.7)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend()
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "coordinates": coords_list,
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
