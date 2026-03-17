import geopandas as gpd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from modules.baseline import main_baseline    
from modules.edge_bundling import matplotlib_map_bundled as mmb1
from modules.edge_bundling_multiple import matplotlib_map_bundled as mmb2
import modules.clustering as clustering



def main_clustered():
    """Interactive mode: cluster countries geographically and let the user pick a source cluster."""

    # Load the GeoJSON file
    gdf = gpd.read_file("data/geo.json")

    # Load country centroid table
    centroid_table = pd.read_csv("data/centroids.csv")

    # Always use the full dataset for clustering
    data = pd.read_csv("data/EU_trade_data_full.csv", sep=';', thousands='.', header=0, index_col=0)

    countries = list(data.index)

    # TODO: REMOVE TESTING: Hard-code values
    TESTING = True
    # Number of clusters
    n = 7
    # Fixed cluster selection
    m = 2
    # Show intra-cluster edges
    show_intra = False  

    # Ask the user how many clusters
    # TODO: REMOVE TESTING
    if not TESTING:
        while True:
            try:
                n = int(input(f"How many clusters? (2-{len(countries)}): "))
                if 2 <= n <= len(countries):
                    break
            except ValueError:
                pass
            print("Please enter a valid number.")

    clusters = clustering.cluster_countries(centroid_table, countries, n)
    
    # TODO: REMOVE TESTING
    if not TESTING:
        source_countries = clustering.select_source_cluster(clusters)
    else:
        source_countries = clusters[m]

    print(f"\nSource countries: {', '.join(sorted(source_countries))}")

    # Filter data to only flows originating from the selected cluster
    filtered = clustering.filter_data_by_sources(data, source_countries)
    print(f"Showing exports from {len(filtered)} source countries to {len(filtered.columns)} destinations.\n")

    # TODO: REMOVE TESTING
    if not TESTING:
        show_intra = input("Show intra-cluster edges (within the source cluster)? [y/n]: ").strip().lower() != 'n'

    multiple_bundle_points = True
    if multiple_bundle_points == True:
        mmb2(gdf, filtered, centroid_table, clusters, show_intra=show_intra)
    else:
        mmb1(gdf, filtered, centroid_table, clusters, show_intra=show_intra)



if __name__ == "__main__":
    # Set mode to "clustered" for interactive clustering, or a dataset name for a pre-made subset
    mode = "clustered"  # Options: "clustered", "full", "distant", "close", "2_clusters", "3_clusters", "5_clusters"

    if mode == "clustered":
        main_clustered()
    else:
        main_baseline(mode)
