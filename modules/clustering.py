# Helper functions related to country clustering

import numpy as np
from scipy.cluster.vq import kmeans2



def cluster_countries(centroid_table, countries, n_clusters):
    """Cluster countries by their geographic coordinates using KMeans.
    Returns a dict mapping cluster_id -> list of country names."""
    # Build a dataframe of only the countries present in the dataset
    geo = centroid_table[centroid_table['name'].isin(countries)].copy()
    coords = geo[['lat', 'lon']].values
    names = geo['name'].values

    rng = np.random.default_rng(42)
    init_idx = rng.choice(len(coords), size=n_clusters, replace=False)
    _, labels = kmeans2(coords.astype(float), coords[init_idx], iter=100, minit='matrix')

    clusters = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)
    return clusters


def select_source_cluster(clusters):
    """Print the clusters and let the user pick one as the source."""
    print("\n=== Country Clusters ===")
    for cid in sorted(clusters):
        members = ", ".join(sorted(clusters[cid]))
        print(f"  Cluster {cid}: {members}")

    while True:
        try:
            choice = int(input(f"\nSelect a source cluster (0-{len(clusters)-1}): "))
            if choice in clusters:
                return clusters[choice]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def filter_data_by_sources(data, source_countries):
    """Keep only rows (sources) that belong to the selected cluster."""
    valid_sources = [c for c in source_countries if c in data.index]
    return data.loc[valid_sources]