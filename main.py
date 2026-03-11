import geopandas as gpd
import matplotlib.pyplot as plt
import folium

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2

def get_centroid(centroid_table, country_name):
    # Get the centroid (longitude, latitude) of a country from a table with (name, centroid) pairs
    # Note: column names are based on current version of centroids.csv; if this file gets replaced, also change the column names here
    row = centroid_table.loc[centroid_table['name'] == country_name]
    try:
        return (row['lon'].iloc[0], row['lat'].iloc[0])
    except:
        print(f"Couldn't find centroid of {country_name}!")
        return (0, 0) # Substitute (0,0) if a country cannot be found



def matplotlib_map(gdf, data, centroid_table):
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot the geometries
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5)

    # Get country centroids
    centroids = {}
    for _, row in gdf.iterrows():
        name = row.get('name')          
        
        # NEW: Get centroid from table (which is pre-computed in a nicer way)
        centroids[name] = get_centroid(centroid_table, name)

        # OLD: Compute centroid based on geodata
        #if name and pd.notna(row['geometry']) and not row['geometry'].is_empty:
            #centroid = row['geometry'].centroid
            #centroids[name] = (centroid.x, centroid.y)   

    # Get highest quantity in whole dataset (for scaling line width)
    max_q = data.max(numeric_only=True).max()

    # Draw each flow as a curved arrow
    for src, row in data.iterrows():
        for dst, qty in row.items():
            # Ignore flow to self and flow to locations not found in the centroids list
            if qty != 0 and src in centroids and dst in centroids:
                src_lon, src_lat = centroids[src]
                dst_lon, dst_lat = centroids[dst]
                lw = 0.5 + (qty / max_q) * 4   

                # Curved arrow annotation
                ax.annotate(
                    "", 
                    xy=(dst_lon, dst_lat), 
                    xytext=(src_lon, src_lat),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=lw,
                        color='red',
                        connectionstyle="arc3,rad=0.2"  
                    )
                )

    # Zoom in on Europe
    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()



def eu_map_folium(gdf, data):
    """Interactive Folium version"""
    
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Create a map centered on Europe
    m = folium.Map(
        location=[52, 10],  
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add the GeoJSON data with tooltips
    folium.GeoJson(
        gdf,
        name='Countries',
        style_function=lambda x: {
            'fillColor': 'lightblue',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'pop_est', 'gdp_md'],
            aliases=['Country:', 'Population:', 'GDP (millions):'],
            localize=True
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save to HTML file
    output_file = 'europe_map.html'
    m.save(output_file)
    print(f"Interactive map saved to {output_file}")
    print("Open this file in your web browser to interact with the map!")
    
    return m



def eu_map_plotly(gdf, data):
    """Interactive Plotly version"""
    
    # Ensure data is in the right CRS for Plotly
    gdf_plotly = gdf.copy()
    if gdf_plotly.crs != "EPSG:4326":
        gdf_plotly = gdf_plotly.to_crs("EPSG:4326")
    
    geojson = gdf_plotly.__geo_interface__

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=gdf_plotly.index,
        z=[0] * len(gdf_plotly),  
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        marker_line_color='black',
        marker_line_width=0.5,
    ))

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.show()
    
    return fig



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


def main(dataset):

    # Load the GeoJSON file
    gdf = gpd.read_file("geo.json")

    # Load country centroid table
    # Current version is taken from https://github.com/360-info/country-centroids?tab=readme-ov-file
    # A few country names have been edited to match the names in geo.json (e.g. Vatican City -> Vatican)
    # Also, a few non-country territories (like Gibraltar or Guernsey) from geo.json are missing in centroids.csv, but I doubt this'll be a problem
    centroid_table = pd.read_csv("centroids.csv")

    # Load the flow map data
    # This is a copy of sheet 2 of EU_trade_data.xlsx (so export in euros), with the total intra-eu line removed
    data = pd.read_csv(f"EU_trade_data_{dataset}.csv", sep=';', thousands='.', header=0, index_col=0)
    print("IMPORTED DATASET: ")
    print(data.head())
    #print(data['Belgium']['Czechia']) # For some reason the first key refers to the column and the second to the row?
                                      # So data[A][B] is the value in euros that B exported to A (or equivalently, that A imported from B)

    matplotlib_map(gdf, data, centroid_table)
    #eu_map_folium(gdf, data)
    #eu_map_plotly(gdf, data)


def main_clustered():
    """Interactive mode: cluster countries geographically and let the user pick a source cluster."""

    # Load the GeoJSON file
    gdf = gpd.read_file("geo.json")

    # Load country centroid table
    centroid_table = pd.read_csv("centroids.csv")

    # Always use the full dataset for clustering
    data = pd.read_csv("EU_trade_data_full.csv", sep=';', thousands='.', header=0, index_col=0)

    countries = list(data.index)

    # Ask the user how many clusters
    while True:
        try:
            n = int(input(f"How many clusters? (2-{len(countries)}): "))
            if 2 <= n <= len(countries):
                break
        except ValueError:
            pass
        print("Please enter a valid number.")

    clusters = cluster_countries(centroid_table, countries, n)
    source_countries = select_source_cluster(clusters)

    print(f"\nSource countries: {', '.join(sorted(source_countries))}")

    # Filter data to only flows originating from the selected cluster
    filtered = filter_data_by_sources(data, source_countries)
    print(f"Showing exports from {len(filtered)} source countries to {len(filtered.columns)} destinations.\n")

    matplotlib_map(gdf, filtered, centroid_table)


if __name__ == "__main__":
    # Set mode to "clustered" for interactive clustering, or a dataset name for a pre-made subset
    mode = "clustered"  # Options: "clustered", "full", "distant", "close", "2_clusters", "3_clusters", "5_clusters"

    if mode == "clustered":
        main_clustered()
    else:
        main(mode)
