import geopandas as gpd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

from modules.baseline import main_baseline    
from modules.edge_bundling import matplotlib_map_bundled as mmb1
from modules.edge_bundling_multiple import matplotlib_map_bundled as mmb2
import modules.clustering as clustering



def main_clustered(n_clusters=None, show_intra=None, multiple_bundle_points=True, bundle_radius=3.0, split_radius=1.5):
    """
    Partitions countries into clusters, then generates a flow map with bundled edges between clusters.

    Parameters
    ----------
    n_clusters : int or None
        The number of clusters to partition the countries into.
        If None, will ask the user in a CLI dialog.
    show_intra : bool or None
        Whether to show export edges within a source cluster.
        If None, will ask the user in a CLI dialog.
    multiple_bundle_points : bool
        Whether to use on bundle point *per target cluster* (True) or share one point for *all target clusters* (False)
    bundle_radius : float
        Radius for feasible regions around source countries to place bundle points in.
        If 0, will ignore feasible regions and place bundling points based on a different cost function.
    split_radius : float
        Radius for feasible regions around destination countries to place splitting points in.
        If 0, will ignore feasible regions and place splitting points based on a different cost function.
        
    Returns
    -------
    map.png
        Saves an image containing one flow map for each cluster with that cluster as the source.
    """

    # Load the GeoJSON file
    gdf = gpd.read_file("data/geo.json")

    # Load country centroid table
    centroid_table = pd.read_csv("data/centroids.csv")

    # Always use the full dataset for clustering
    data = pd.read_csv("data/EU_trade_data_full.csv", sep=';', thousands='.', header=0, index_col=0)

    countries = list(data.index)

    # Ask the user how many clusters
    if n_clusters == None:
        while True:
            try:
                n_clusters = int(input(f"How many clusters? (2-{len(countries)}): "))
                if 2 <= n_clusters <= len(countries):
                    break
            except ValueError:
                pass
            print("Please enter a valid number.")

    clusters = clustering.cluster_countries(centroid_table, countries, n_clusters)
    
    if show_intra == None:
        show_intra = input("Show intra-cluster edges (within the source cluster)? [y/n]: ").strip().lower() != 'n'
    
    n_keys = len(clusters)
    cols = min(3, n_keys)
    rows = math.ceil(n_keys / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(36 * cols, 30 * rows))
    axes = axes.flatten()

    for i, m in enumerate(sorted(clusters.keys())):
        ax = axes[i]
        source_countries = clusters[m]
        print(f"\nSource cluster {m} countries: {', '.join(sorted(source_countries))}")

        # Filter data to only flows originating from the selected cluster
        filtered = clustering.filter_data_by_sources(data, source_countries)
        print(f"Showing exports from {len(filtered)} source countries to {len(filtered.columns)} destinations.\n")

        ax.set_title(f"Cluster {m}")

        if multiple_bundle_points:
            mmb2(gdf, filtered, centroid_table, clusters, show_intra=show_intra, ax=ax, bundle_radius=bundle_radius, split_radius=split_radius)
        else:
            mmb1(gdf, filtered, centroid_table, clusters, show_intra=show_intra, ax=ax, radius=bundle_radius)
            
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("map.png")
    # plt.show()


if __name__ == "__main__":
    # Set mode to "clustered" for interactive clustering, or a dataset name for a pre-made subset
    mode = "clustered"  # Options: "clustered", "full", "distant", "close", "2_clusters", "3_clusters", "5_clusters"

    if mode == "clustered":
        main_clustered(n_clusters = 7, show_intra=False, multiple_bundle_points=True, bundle_radius=0, split_radius=0)
    else:
        main_baseline(mode)
