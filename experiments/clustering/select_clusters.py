import json

JSON_PATH = "./experiments/clustering/experiments.json"


def compute_score(cluster, avg_dist, avg_bundle, avg_cross):
    """
    Normalized distance from average (lower = more representative)
    """
    dist = cluster["distance_score"]
    bundle = cluster["bundling_score"]
    cross = cluster["crossings"]

    return (
        abs(dist - avg_dist) / (avg_dist + 1e-6)
        + abs(bundle - avg_bundle) / (avg_bundle + 1e-6)
        + abs(cross - avg_cross) / (avg_cross + 1e-6)
    )


def find_clusters(data):
    results = {}

    for exp in data["experiments"]:
        k = exp["config"]["n_clusters"]

        avg_dist = exp["metrics"]["avg_distance_score"]
        avg_bundle = exp["metrics"]["avg_bundling_score"]
        avg_cross = exp["metrics"]["avg_crossings"]

        best_cluster = None
        best_score = float("inf")

        worst_cluster = None
        worst_score = -float("inf")

        for cluster in exp["clusters"]:
            score = compute_score(cluster, avg_dist, avg_bundle, avg_cross)

            # Representative (closest to average)
            if score < best_score:
                best_score = score
                best_cluster = cluster

            # Worst (furthest from average)
            if score > worst_score:
                worst_score = score
                worst_cluster = cluster

        # Best visual (min crossings, max bundling, max distance)
        best_visual_cluster = min(
            exp["clusters"],
            key=lambda c: (
                c["crossings"],             
                -c["bundling_score"],       
                -c["distance_score"]        
            )
        )

        results[k] = {
            "representative": best_cluster,
            "worst": worst_cluster,
            "best_visual": best_visual_cluster,
        }

    return results


def print_results(results):
    print("\n=== Cluster Selection Results ===")

    for k, info in results.items():
        print(f"\n--- k = {k} ---")

        # Representative
        print("\nRepresentative (closest to average):")
        print(f"  cluster_id: {info['representative']['cluster_id']}")
        print(f"  metrics: dist={info['representative']['distance_score']:.2f}, "
              f"bundle={info['representative']['bundling_score']:.2f}, "
              f"cross={info['representative']['crossings']}")
        print(f"  countries: {info['representative']['countries']}")

        # Best visual
        print("\nBest visual (low crossings + high bundling + good separation):")
        print(f"  cluster_id: {info['best_visual']['cluster_id']}")
        print(f"  metrics: dist={info['best_visual']['distance_score']:.2f}, "
              f"bundle={info['best_visual']['bundling_score']:.2f}, "
              f"cross={info['best_visual']['crossings']}")
        print(f"  countries: {info['best_visual']['countries']}")

        # Worst
        print("\nWorst (most extreme):")
        print(f"  cluster_id: {info['worst']['cluster_id']}")
        print(f"  metrics: dist={info['worst']['distance_score']:.2f}, "
              f"bundle={info['worst']['bundling_score']:.2f}, "
              f"cross={info['worst']['crossings']}")
        print(f"  countries: {info['worst']['countries']}")


def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    results = find_clusters(data)
    print_results(results)


if __name__ == "__main__":
    main()