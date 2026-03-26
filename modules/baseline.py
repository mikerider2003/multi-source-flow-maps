# Our initial/baseline flow map plot using matplotlib

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from modules.centroids import get_centroid



def matplotlib_map(gdf, data, centroid_table):
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(40, 40))
    
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


def main_baseline(dataset):
    # Load the GeoJSON file
    gdf = gpd.read_file("data/geo.json")

    # Load country centroid table
    # Current version is taken from https://github.com/360-info/country-centroids?tab=readme-ov-file
    # A few country names have been edited to match the names in geo.json (e.g. Vatican City -> Vatican)
    # Also, a few non-country territories (like Gibraltar or Guernsey) from geo.json are missing in centroids.csv, but I doubt this'll be a problem
    centroid_table = pd.read_csv("data/centroids.csv")

    # Load the flow map data
    # This is a copy of sheet 2 of EU_trade_data.xlsx (so export in euros), with the total intra-eu line removed
    data = pd.read_csv(f"data/EU_trade_data_{dataset}.csv", sep=';', thousands='.', header=0, index_col=0)
    print("IMPORTED DATASET: ")
    print(data.head())
    
    matplotlib_map(gdf, data, centroid_table)