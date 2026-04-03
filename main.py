import geopandas as gpd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

from modules.baseline import main_baseline    
from modules.edge_bundling import matplotlib_map_bundled as mmb1
from modules.edge_bundling_multiple import matplotlib_map_bundled as mmb2
import modules.clustering as clustering

COUNTRY_ISO2 = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czechia': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
    'Hungary': 'HU', 'Ireland': 'IE', 'Italy': 'IT', 'Latvia': 'LV',
    'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT', 'Netherlands': 'NL',
    'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Slovakia': 'SK',
    'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE',
}


def main_clustered(n_clusters=None, show_intra=None, multiple_bundle_points=True, bundle_radius=3.0, split_radius=1.5, q2_weight=0.3, q3_weight=0.15, output_file="map.png"):
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

    estimated_exports = {} # Estimated export for every pair of countries. For countries A and B belonging respectively to clusters X and Y, the estimated export is
    # fraction of total export from X that is due to A * fraction of total export from X that goes to B * total export from X to Y, which can equivalently be calculated as
    # sum([export[A, d] for d in destination_countries]) * sum([export[s, B] for s in source_countries]) / sum([[export[s,d] for s in source_countries] for d in destination_countries])

    for i, m in enumerate(sorted(clusters.keys())):
        ax = axes[i]
        source_countries = clusters[m]
        print(f"\nSource cluster {m} countries: {', '.join(sorted(source_countries))}")

        # Filter data to only flows originating from the selected cluster
        filtered = clustering.filter_data_by_sources(data, source_countries)
        print(f"Showing exports from {len(filtered)} source countries to {len(filtered.columns)} destinations.\n")

        ax.set_title(f"Cluster {m}")

        if multiple_bundle_points:
            mmb2(gdf, filtered, centroid_table, clusters, show_intra=show_intra, ax=ax, bundle_radius=bundle_radius, split_radius=split_radius, estimated_exports=estimated_exports, q2_weight=q2_weight, q3_weight=q3_weight)
        else:
            mmb1(gdf, filtered, centroid_table, clusters, show_intra=show_intra, ax=ax, radius=bundle_radius)
            
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Print estimated export table in LaTeX format
    countries.sort()

    print()
    print("----- ESTIMATED EXPORTS -----")
    print()
    print(f"Min: {min(estimated_exports.values())}")
    print(f"Max: {max(estimated_exports.values())}")
    print()

    header_str = "Exporter "
    for country in countries:
        header_str += f"& {COUNTRY_ISO2[country]} "
    header_str += "\\\\"
    print(header_str)
    print("\\hline")

    for src_country in countries:
        line_str = f"{src_country} "
        for dst_country in countries:
            try:
                # Color cell based on value
                # Simple gradient: < 0.5 is red, 0.5-0.8 is yellow, 0.8-1.2 is green, 1.2-1.5 is yellow again, and >1.5 is red again.
                value = estimated_exports[src_country, dst_country]
                if value < 0.5 or value > 1.5:
                    colorcmd = "\\cellcolor{red}"
                elif value < 0.8 or value > 1.2:
                    colorcmd = "\\cellcolor{yellow}"
                else:
                    colorcmd = "\\cellcolor{green}"
                line_str += f"& {colorcmd}{value:.2f} "
            except KeyError: # If there is no flow between the two countries (e.g. if src = dst), write a "-"
                line_str += "& - "
        line_str += "\\\\"
        print(line_str)

    # Save image
    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()


if __name__ == "__main__":
    # Set mode to "clustered" for interactive clustering, or a dataset name for a pre-made subset
    mode = "clustered"  # Options: "clustered", "full", "distant", "close", "2_clusters", "3_clusters", "5_clusters"

    if mode == "clustered":
        main_clustered(n_clusters = 7, show_intra=False, multiple_bundle_points=True, bundle_radius=0, split_radius=0)
    else:
        main_baseline(mode)
