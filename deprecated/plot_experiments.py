# Initial experiments we did with folium and plotly, before settling on matplotlib

import pandas as pd
import geopandas as gpd
import folium
import plotly.graph_objects as go

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
    gdf = gpd.read_file("data/geo.json")

    # Load the flow map data
    # This is a copy of sheet 2 of EU_trade_data.xlsx (so export in euros), with the total intra-eu line removed
    data = pd.read_csv(f"data/EU_trade_data_{dataset}.csv", sep=';', thousands='.', header=0, index_col=0)
    print("IMPORTED DATASET: ")
    print(data.head())
    
    #eu_map_folium(gdf, data)
    eu_map_plotly(gdf, data)


if __name__ == "__main__":
    dataset = "full"  # Options: "full", "distant", "close", "2_clusters", "3_clusters", "5_clusters"
    main(dataset)