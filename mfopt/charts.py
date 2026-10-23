"""Matplotlib charts saved to docs/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .intents import CATEGORIES  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
LEVEL_COLORS = {"low": "#2a78d6", "medium": "#eb6834", "high": "#1baf7a"}
METHOD_COLORS = {"greedy_set_cover": "#2a78d6", "cost_aware_ranking": "#eb6834",
                 "random_mean": "#52514e"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#2a78d6", "#123a6b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width=7.5, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def accuracy_chart(report, path):
    metrics = [("exact_match", "Exact match"), ("intent_accuracy", "Intent accuracy"),
               ("slot_f1", "Slot F1")]
    levels = report["levels"]
    fig, ax = _figure()
    width = 0.8 / len(levels)
    for i, lv in enumerate(levels):
        xs = [m + (i - (len(levels) - 1) / 2) * width for m in range(len(metrics))]
        vals = [100 * report["summaries"][lv][key] for key, _ in metrics]
        ax.bar(xs, vals, width * 0.92, color=LEVEL_COLORS[lv], label=lv)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=TEXT_2)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent (slot F1 x 100)", color=TEXT_2, fontsize=9)
    ax.set_title(f"Accuracy by fidelity level ({report['config']['n_tests']} tests)",
                 loc="left", color=TEXT, fontsize=11)
    ax.legend(frameon=False, fontsize=9, title="Level", title_fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def failure_heatmap(report, tests, results, path):
    levels = report["levels"]
    n_by_cat = {c: sum(1 for t in tests if t["tags"]["category"] == c) for c in CATEGORIES}
    rates, counts = [], []
    for cat in CATEGORIES:
        row_r, row_c = [], []
        for lv in levels:
            fails = sum(1 for t in tests if t["tags"]["category"] == cat
                        and not results[lv][t["id"]]["passed"])
            row_r.append(fails / n_by_cat[cat])
            row_c.append(fails)
        rates.append(row_r)
        counts.append(row_c)
    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(rates, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i, cat in enumerate(CATEGORIES):
        for j in range(len(levels)):
            color = "white" if rates[i][j] > 0.55 else TEXT
            ax.text(j, i, f"{counts[i][j]}/{n_by_cat[cat]}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Failed tests per category and level", loc="left", color=TEXT, fontsize=11)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Failure rate", color=TEXT_2, fontsize=9)
    bar.ax.tick_params(colors=TEXT_2, labelsize=8)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def cost_coverage_chart(report, path):
    curves = report["curves"]
    xs = curves["cost_pct"]
    fig, ax = _figure()
    ax.fill_between(xs, curves["random_min"], curves["random_max"], color="#52514e", alpha=0.12,
                    linewidth=0, label="random, min to max (20 seeds)")
    labels = {"greedy_set_cover": "greedy set cover", "cost_aware_ranking": "cost-aware ranking",
              "random_mean": "random, mean (20 seeds)"}
    # cost-aware is drawn wider and below greedy, so both stay visible where they overlap
    for key, style, width in (("random_mean", "--", 2), ("cost_aware_ranking", "-", 4),
                              ("greedy_set_cover", "-", 1.6)):
        ax.plot(xs, curves[key], style, color=METHOD_COLORS[key], linewidth=width, label=labels[key])
    for key, marker in (("greedy_set_cover", "s"), ("cost_aware_ranking", "o")):
        m = report["selection"][key]["method"]
        x = 100 * (1 - m["cost_saved"])
        y = 100 * m.get("coverage_fresh", m["coverage_calibration"])
        ax.plot([x], [y], marker, markersize=8, color=METHOD_COLORS[key],
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{labels[key]}: final subset (coverage + sample)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Expensive-level cost used (% of running all tests)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Failure categories detected (%)", color=TEXT_2, fontsize=9)
    where = "mean over fresh runs" if report["fresh_runs_failure_categories"] else "calibration run"
    ax.set_title(f"Cost vs coverage of failure categories ({where})",
                 loc="left", color=TEXT, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def make_all(report, tests, results, out_dir):
    accuracy_chart(report, out_dir / "accuracy_by_fidelity.png")
    failure_heatmap(report, tests, results, out_dir / "failure_heatmap.png")
    cost_coverage_chart(report, out_dir / "cost_vs_coverage.png")
