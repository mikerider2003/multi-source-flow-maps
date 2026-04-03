import json
import matplotlib.pyplot as plt


def plot_separate_boxplots(data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    metrics = [
        ("distance_score", "Distance"),
        ("bundling_score", "Bundling"),
        ("crossings", "Crossings")
    ]

    for i, (metric, title) in enumerate(metrics):
        values = []
        labels = []

        for exp in data["experiments"]:
            k = exp["config"]["n_clusters"]
            vals = [c[metric] for c in exp["clusters"]]
            values.append(vals)
            labels.append(f"k={k}")

        bp = axes[i].boxplot(
            values,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black')
        )

        # Simple color styling
        for box in bp['boxes']:
            box.set_facecolor("#F4C947")
            box.set_edgecolor("black")

        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color("black")

        axes[i].set_title(title)
        axes[i].set_xticklabels(labels)

    # Simple legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Median'),
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='Mean')
    ]
    fig.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_json("Experiments/experiments.json")
    plot_separate_boxplots(data)