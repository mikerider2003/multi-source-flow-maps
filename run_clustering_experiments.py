import os
import sys
import shutil
import json
import re
from io import StringIO
from datetime import datetime

from main import main_clustered


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "clustering")


def run_experiment_1(
    n_clusters,
    show_intra=False,
    multiple_bundle_points=True,
    bundle_radius=0,
    split_radius=0,
    experiment_name=None,
    output_dir=OUTPUT_DIR,
):
    """
    Wrapper that runs main_clustered, captures output, and returns results
    """

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name or f'n_clusters={n_clusters}'}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        main_clustered(
            n_clusters=n_clusters,
            show_intra=show_intra,
            multiple_bundle_points=multiple_bundle_points,
            bundle_radius=bundle_radius,
            split_radius=split_radius,
            print_table=False,
        )
    finally:
        sys.stdout = old_stdout

    output = captured_output.getvalue()
    print(output)


    distance_match = re.search(r'AVG BUNDLE-SPLIT DISTANCE SCORE:\s+([\d.]+)', output)
    bundling_match = re.search(r'AVG EDGE BUNDLING SCORE:\s+([\d.]+)', output)
    crossings_match = re.search(r'AVG NR OF EDGE CROSSINGS:\s+([\d.]+)', output)
    edges_match = re.search(r'AVG NR OF EDGES:\s+([\d.]+)', output)
    
    # Capture normalized metrics
    norm_distance_match = re.search(r'NORMALIZED BUNDLE-SPLIT DISTANCE SCORE:\s+([\d.]+)', output)
    norm_bundling_match = re.search(r'NORMALIZED EDGE BUNDLING SCORE:\s+([\d.]+)', output)
    norm_crossings_match = re.search(r'NORMALIZED NR OF EDGE CROSSINGS:\s+([\d.]+)', output)

    metrics = {}
    if distance_match:
        metrics['avg_distance_score'] = float(distance_match.group(1))
    if bundling_match:
        metrics['avg_bundling_score'] = float(bundling_match.group(1))
    if crossings_match:
        metrics['avg_crossings'] = float(crossings_match.group(1))
    if edges_match:
        metrics['avg_edges'] = float(edges_match.group(1))
    
    # Add normalized metrics
    if norm_distance_match:
        metrics['normalized_distance_score'] = float(norm_distance_match.group(1))
    if norm_bundling_match:
        metrics['normalized_bundling_score'] = float(norm_bundling_match.group(1))
    if norm_crossings_match:
        metrics['normalized_crossings'] = float(norm_crossings_match.group(1))


    cluster_pattern = re.findall(
        r"Source cluster (\d+) countries: (.*?)\n.*?"
        r"BUNDLE/SPLIT DISTANCE SCORE \(Q3\):\s+([\d.]+)\n"
        r"EDGE CROSSINGS \(Q1\):\s+([\d.]+)\n"
        r"BUNDLING SCORE \(Q2\):\s+([\d.]+)\n"
        r"NR OF EDGES:\s+([\d.]+)",
        output,
        re.DOTALL
    )

    clusters = []

    for match in cluster_pattern:
        cluster_id = int(match[0])
        countries_raw = match[1]
        distance = float(match[2])
        crossings = float(match[3])
        bundling = float(match[4])
        edges = float(match[5])

        countries = [c.strip() for c in countries_raw.split(",")]
        
        # Calculate normalized per-edge metrics for this cluster
        normalized_distance = distance / edges if edges > 0 else 0
        normalized_bundling = bundling / edges if edges > 0 else 0
        normalized_crossings = crossings / edges if edges > 0 else 0

        clusters.append({
            "cluster_id": cluster_id,
            "countries": countries,
            "distance_score": distance,
            "bundling_score": bundling,
            "crossings": crossings,
            "edges": edges,
            "normalized_distance_score": normalized_distance,
            "normalized_bundling_score": normalized_bundling,
            "normalized_crossings": normalized_crossings
        })


    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name or f"n_clusters_{n_clusters}",
        "config": {
            "n_clusters": n_clusters,
            "show_intra": show_intra,
            "multiple_bundle_points": multiple_bundle_points,
            "bundle_radius": bundle_radius,
            "split_radius": split_radius,
        },
        "metrics": metrics,
        "clusters": clusters
    }


    if experiment_name:
        new_filename = os.path.join(output_dir, f"map_{experiment_name}.png")
    else:
        new_filename = os.path.join(output_dir, f"map_{n_clusters}.png")


    map_path = os.path.join(BASE_DIR, "map.png")

    if os.path.exists(map_path):
        shutil.move(map_path, new_filename)
        print(f"Saved map to {new_filename}")

    print(f"\n{'='*60}\n")

    return results


def run_batch_experiment_1():
    """
    Run experiments for different cluster numbers (3, 7, 12)
    Saves all results to experiments/clustering/experiments.json
    """

    output_dir = OUTPUT_DIR

    print("\n" + "="*70)
    print("BATCH EXPERIMENT 1: Testing cluster numbers 3, 7, 12")
    print("="*70)
    print("\nOutput:")
    print(f"   - Metrics → {output_dir}/experiments.json")
    print(f"   - Images  → {output_dir}/map_{{3,7,12}}.png")
    print("="*70 + "\n")

    cluster_values = [3, 7, 12]
    all_results = []

    for n_clusters in cluster_values:
        result = run_experiment_1(
            n_clusters=n_clusters,
            show_intra=False,
            multiple_bundle_points=True,
            bundle_radius=0,
            split_radius=0,
            experiment_name=str(n_clusters),
            output_dir=output_dir,
        )
        all_results.append(result)

    combined_results = {
        "batch_timestamp": datetime.now().isoformat(),
        "description": "Cluster analysis experiment with n_clusters = 3, 7, 12",
        "experiments": all_results
    }

    json_path = os.path.join(output_dir, "experiments.json")
    with open(json_path, "w") as f:
        json.dump(combined_results, f, indent=4)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nSummary Table (Non-Normalized - LaTeX format):")
    print("-" * 70)

    for result in all_results:
        n = result['config']['n_clusters']
        metrics = result['metrics']
        crossings = metrics.get('avg_crossings', 0)
        bundling = metrics.get('avg_bundling_score', 0)
        distance = metrics.get('avg_distance_score', 0)

        print(f"{n:2d} & {crossings:6.2f} & {bundling:6.2f} & {distance:6.2f} \\\\")

    print("-" * 70)
    print("\nSummary Table (Normalized Per-Edge - LaTeX format):")
    print("-" * 70)

    for result in all_results:
        n = result['config']['n_clusters']
        metrics = result['metrics']
        norm_crossings = metrics.get('normalized_crossings', 0)
        norm_bundling = metrics.get('normalized_bundling_score', 0)
        norm_distance = metrics.get('normalized_distance_score', 0)

        print(f"{n:2d} & {norm_crossings:6.4f} & {norm_bundling:6.4f} & {norm_distance:6.4f} \\\\")
    
    print("-" * 70)
    print(f"\nSaved JSON → {json_path}")
    print(f"Saved maps → {output_dir}/map_{{3,7,12}}.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_batch_experiment_1()