import geopandas as gpd
import matplotlib.pyplot as plt
import folium

import plotly.graph_objects as go
import pandas as pd

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
    


if __name__ == "__main__":
    dataset = "3_clusters" # Options: "full", "distant", "close", "2_clusters", "3_clusters" or "5_clusters". For descriptions, see the README.
    main(dataset)
