import json
import sys

JSON_PATH = "./experiments/clustering/experiments.json"


def compute_score(cluster, avg_dist, avg_bundle, avg_cross, use_normalized=False):
    """
    Normalized distance from average (lower = more representative)
    
    Parameters
    ----------
    cluster : dict
        Cluster data
    avg_dist, avg_bundle, avg_cross : float
        Average metrics for comparison
    use_normalized : bool
        If True, use normalized (per-edge) metrics
    """
    if use_normalized:
        dist = cluster["normalized_distance_score"]
        bundle = cluster["normalized_bundling_score"]
        cross = cluster["normalized_crossings"]
    else:
        dist = cluster["distance_score"]
        bundle = cluster["bundling_score"]
        cross = cluster["crossings"]

    return (
        abs(dist - avg_dist) / (avg_dist + 1e-6)
        + abs(bundle - avg_bundle) / (avg_bundle + 1e-6)
        + abs(cross - avg_cross) / (avg_cross + 1e-6)
    )


def find_clusters(data, use_normalized=False):
    """
    Find representative, worst, and best visual clusters for each k.
    
    Parameters
    ----------
    data : dict
        Experiment data
    use_normalized : bool
        If True, use normalized (per-edge) metrics
    """
    results = {}

    for exp in data["experiments"]:
        k = exp["config"]["n_clusters"]

        if use_normalized:
            avg_dist = exp["metrics"]["normalized_distance_score"]
            avg_bundle = exp["metrics"]["normalized_bundling_score"]
            avg_cross = exp["metrics"]["normalized_crossings"]
        else:
            avg_dist = exp["metrics"]["avg_distance_score"]
            avg_bundle = exp["metrics"]["avg_bundling_score"]
            avg_cross = exp["metrics"]["avg_crossings"]

        best_cluster = None
        best_score = float("inf")

        worst_cluster = None
        worst_score = -float("inf")

        for cluster in exp["clusters"]:
            score = compute_score(cluster, avg_dist, avg_bundle, avg_cross, use_normalized)

            # Representative (closest to average)
            if score < best_score:
                best_score = score
                best_cluster = cluster

            # Worst (furthest from average)
            if score > worst_score:
                worst_score = score
                worst_cluster = cluster

        # Best visual (min crossings, max bundling, max distance)
        if use_normalized:
            best_visual_cluster = min(
                exp["clusters"],
                key=lambda c: (
                    c["normalized_crossings"],             
                    -c["normalized_bundling_score"],       
                    -c["normalized_distance_score"]        
                )
            )
        else:
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


def print_results(results, use_normalized=False):
    """
    Print cluster selection results.
    
    Parameters
    ----------
    results : dict
        Results from find_clusters
    use_normalized : bool
        If True, display normalized metrics
    """
    print("\n=== Cluster Selection Results ===")
    if use_normalized:
        print("(Using NORMALIZED per-edge metrics)")
    else:
        print("(Using NON-NORMALIZED metrics)")

    for k, info in results.items():
        print(f"\n--- k = {k} ---")

        # Representative
        print("\nRepresentative (closest to average):")
        print(f"  cluster_id: {info['representative']['cluster_id']}")
        if use_normalized:
            print(f"  normalized metrics: dist={info['representative']['normalized_distance_score']:.4f}, "
                  f"bundle={info['representative']['normalized_bundling_score']:.4f}, "
                  f"cross={info['representative']['normalized_crossings']:.4f}")
        else:
            print(f"  metrics: dist={info['representative']['distance_score']:.2f}, "
                  f"bundle={info['representative']['bundling_score']:.2f}, "
                  f"cross={info['representative']['crossings']}")
        print(f"  countries: {info['representative']['countries']}")

        # Best visual
        print("\nBest visual (low crossings + high bundling + good separation):")
        print(f"  cluster_id: {info['best_visual']['cluster_id']}")
        if use_normalized:
            print(f"  normalized metrics: dist={info['best_visual']['normalized_distance_score']:.4f}, "
                  f"bundle={info['best_visual']['normalized_bundling_score']:.4f}, "
                  f"cross={info['best_visual']['normalized_crossings']:.4f}")
        else:
            print(f"  metrics: dist={info['best_visual']['distance_score']:.2f}, "
                  f"bundle={info['best_visual']['bundling_score']:.2f}, "
                  f"cross={info['best_visual']['crossings']}")
        print(f"  countries: {info['best_visual']['countries']}")

        # Worst
        print("\nWorst (most extreme):")
        print(f"  cluster_id: {info['worst']['cluster_id']}")
        if use_normalized:
            print(f"  normalized metrics: dist={info['worst']['normalized_distance_score']:.4f}, "
                  f"bundle={info['worst']['normalized_bundling_score']:.4f}, "
                  f"cross={info['worst']['normalized_crossings']:.4f}")
        else:
            print(f"  metrics: dist={info['worst']['distance_score']:.2f}, "
                  f"bundle={info['worst']['bundling_score']:.2f}, "
                  f"cross={info['worst']['crossings']}")
        print(f"  countries: {info['worst']['countries']}")


def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Check if --normalized flag is passed
    use_normalized = "--normalized" in sys.argv or "-n" in sys.argv
    
    if not use_normalized:
        print("\nNote: Use --normalized or -n flag to use normalized (per-edge) metrics\n")

    results = find_clusters(data, use_normalized=use_normalized)
    print_results(results, use_normalized=use_normalized)


if __name__ == "__main__":
    main()