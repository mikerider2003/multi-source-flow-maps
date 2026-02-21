import geopandas as gpd
import matplotlib.pyplot as plt
import folium

import plotly.graph_objects as go
import pandas as pd

def matplotlib_map(gdf, data):
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot the geometries
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5)


    # Zoom in on Europe
    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])
    
    # Remove axis ticks 
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Show the plot
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


def main():

    # Load the GeoJSON file
    gdf = gpd.read_file("geo.json")

    # Load the flow_map data
    data = pd.read_csv("data.csv")

    matplotlib_map(gdf, data)
    # eu_map_folium(gdf, data)
    # eu_map_plotly(gdf, data)
    


if __name__ == "__main__":
    main()
