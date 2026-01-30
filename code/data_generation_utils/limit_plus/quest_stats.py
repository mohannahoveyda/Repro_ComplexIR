"""
Statistics and visualization for limit_quest_queries.jsonl.

After limit_quest_queries.jsonl is generated, use this module to:
1. Print/save a statistics table: samples per template, mean num_relevant_docs per template.
2. Generate a figure: overall distribution of num_relevant_docs + one subfigure per template.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

DEFAULT_QUERIES_PATH = os.path.join(
    os.path.dirname(__file__), "limit_data", "limit_quest_queries.jsonl"
)


def load_limit_quest_queries(path: str) -> List[dict]:
    """Load limit_quest_queries.jsonl into a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Queries file not found: {path}")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def compute_stats(samples: List[dict]) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Compute per-template num_docs lists and overall num_docs list.

    Returns:
        template2num_docs: template name -> list of num_docs for each sample
        all_num_docs: list of num_docs for all samples
    """
    template2num_docs: Dict[str, List[int]] = defaultdict(list)
    all_num_docs: List[int] = []

    for s in samples:
        num_docs = s.get("num_docs", len(s.get("docs", [])))
        all_num_docs.append(num_docs)
        template = s.get("metadata", {}).get("template", "?")
        template2num_docs[template].append(num_docs)

    return dict(template2num_docs), all_num_docs


def get_stats_table(
    template2num_docs: Dict[str, List[int]],
    all_num_docs: List[int],
    template_order: Optional[List[str]] = None,
) -> List[Tuple[str, int, float]]:
    """
    Build statistics table: (template, count, mean_num_relevant_docs).

    If template_order is None, templates are sorted by name.
    """
    if template_order is None:
        template_order = DEFAULT_TEMPLATE_ORDER

    rows = []
    for t in template_order:
        vals = template2num_docs.get(t, [])
        n = len(vals)
        mean_val = sum(vals) / n if n else 0.0
        rows.append((t, n, mean_val))

    # Optional total row
    total_n = len(all_num_docs)
    total_mean = sum(all_num_docs) / total_n if total_n else 0.0
    rows.append(("** Total **", total_n, total_mean))

    return rows


def print_stats_table(rows: List[Tuple[str, int, float]]) -> None:
    """Print the stats table to stdout."""
    col_tmpl = "Template"
    col_count = "Samples"
    col_mean = "Avg num_relevant_docs"
    max_tmpl_len = max(len(col_tmpl), max(len(r[0]) for r in rows))
    max_count_len = max(len(col_count), len(str(max(r[1] for r in rows))))
    fmt = f"{{:<{max_tmpl_len}}}  {{:>{max_count_len}}}  {{:>20}}"
    print(fmt.format(col_tmpl, col_count, col_mean))
    print("-" * (max_tmpl_len + 2 + max_count_len + 2 + 20))
    for t, n, mean_val in rows:
        print(fmt.format(t, n, f"{mean_val:.2f}"))


def save_stats_table_csv(
    rows: List[Tuple[str, int, float]],
    out_path: str,
) -> None:
    """Save the stats table as CSV (excluding the total row in a separate line if desired)."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("template,samples,avg_num_relevant_docs\n")
        for t, n, mean_val in rows:
            # Escape template name if it contains commas
            t_esc = f'"{t}"' if "," in t else t
            f.write(f"{t_esc},{n},{mean_val:.4f}\n")


def plot_num_relevant_docs_distribution(
    template2num_docs: Dict[str, List[int]],
    all_num_docs: List[int],
    template_order: Optional[List[str]] = None,
    fig_path: Optional[str] = None,
    fig_title: str = "Distribution of num_relevant_docs",
) -> bool:
    """
    Create a figure with:
    - One main subplot: distribution of num_relevant_docs across all queries.
    - Seven subplots: distribution for each template.

    Saves to fig_path if provided (e.g. limit_data/limit_quest_num_docs_dist.png).
    Returns True if figure was saved, False if matplotlib/numpy are not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if fig_path:
            print("Skipping figure (matplotlib not installed). Install with: pip install matplotlib numpy")
        return False

    if template_order is None:
        template_order = DEFAULT_TEMPLATE_ORDER

    # 1 main + 7 template subplots
    n_templates = len(template_order)
    n_sub = 1 + n_templates  # first is overall, then one per template
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes_flat = axes.flatten()

    # Overall distribution (first subplot)
    ax0 = axes_flat[0]
    ax0.hist(all_num_docs, bins=min(50, len(set(all_num_docs)) or 1), edgecolor="black", alpha=0.7)
    ax0.set_title("All queries")
    ax0.set_xlabel("num_relevant_docs")
    ax0.set_ylabel("Count")

    # One subplot per template
    for i, template in enumerate(template_order):
        ax = axes_flat[i + 1]
        vals = template2num_docs.get(template, [])
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(template[:30] + "..." if len(template) > 30 else template)
            continue
        bins = min(30, len(set(vals)) or 1)
        ax.hist(vals, bins=bins, edgecolor="black", alpha=0.7)
        short_title = template[:35] + "..." if len(template) > 35 else template
        ax.set_title(short_title, fontsize=9)
        ax.set_xlabel("num_relevant_docs")
        ax.set_ylabel("Count")

    # Hide unused subplots (we have 8 axes: 1 overall + 7 templates)
    for j in range(1 + n_templates, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(fig_title, fontsize=12, y=1.02)
    plt.tight_layout()

    if fig_path:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {fig_path}")
    plt.close(fig)
    return True


def compute_and_display_limit_quest_stats(
    queries_path: str = DEFAULT_QUERIES_PATH,
    output_dir: Optional[str] = None,
    save_table: bool = True,
    save_figure: bool = True,
    template_order: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Load limit_quest_queries.jsonl, compute statistics, print table, optionally save CSV and figure.

    Args:
        queries_path: Path to limit_quest_queries.jsonl.
        output_dir: If set, save table CSV and figure here; otherwise same dir as queries_path.
        save_table: If True, save stats table as CSV.
        save_figure: If True, save distribution figure as PNG.
        template_order: Order of templates for table and subplots. Default: sorted by name.

    Returns:
        Dict with keys: samples, template2num_docs, all_num_docs, table_rows.
    """
    samples = load_limit_quest_queries(queries_path)
    template2num_docs, all_num_docs = compute_stats(samples)

    if template_order is None:
        template_order = DEFAULT_TEMPLATE_ORDER

    table_rows = get_stats_table(template2num_docs, all_num_docs, template_order)

    print("=" * 60)
    print("LIMIT-QUEST queries statistics")
    print("=" * 60)
    print_stats_table(table_rows)
    print()

    if output_dir is None:
        output_dir = os.path.dirname(queries_path)
    os.makedirs(output_dir, exist_ok=True)

    if save_table:
        table_path = os.path.join(output_dir, "limit_quest_stats.csv")
        save_stats_table_csv(table_rows, table_path)
        print(f"Saved stats table to: {table_path}")

    if save_figure:
        fig_path = os.path.join(output_dir, "limit_quest_num_docs_dist.png")
        plot_num_relevant_docs_distribution(
            template2num_docs,
            all_num_docs,
            template_order=template_order,
            fig_path=fig_path,
        )

    return {
        "samples": samples,
        "template2num_docs": template2num_docs,
        "all_num_docs": all_num_docs,
        "table_rows": table_rows,
    }


# Default template order matching generate.TEMPLATES (7 templates)
DEFAULT_TEMPLATE_ORDER = [
    "_",
    "_ or _",
    "_ or _ or _",
    "_ that are also _",
    "_ that are also both _ and _",
    "_ that are also _ but not _",
    "_ that are not _",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute and display LIMIT-QUEST query statistics.")
    parser.add_argument(
        "--queries",
        default=DEFAULT_QUERIES_PATH,
        help="Path to limit_quest_queries.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV and figure output (default: same as queries file)",
    )
    parser.add_argument("--no-table", action="store_true", help="Do not save CSV table.")
    parser.add_argument("--no-figure", action="store_true", help="Do not save distribution figure.")
    args = parser.parse_args()

    compute_and_display_limit_quest_stats(
        queries_path=args.queries,
        output_dir=args.output_dir,
        save_table=not args.no_table,
        save_figure=not args.no_figure,
        template_order=DEFAULT_TEMPLATE_ORDER,
    )


if __name__ == "__main__":
    main()
