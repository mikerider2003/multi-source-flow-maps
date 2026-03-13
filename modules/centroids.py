# Helper functions related to country centroids

def get_centroid(centroid_table, country_name):
    # Get the centroid (longitude, latitude) of a country from a table with (name, centroid) pairs
    # Note: column names are based on current version of centroids.csv; if this file gets replaced, also change the column names here
    row = centroid_table.loc[centroid_table['name'] == country_name]
    try:
        return (row['lon'].iloc[0], row['lat'].iloc[0])
    except:
        print(f"Couldn't find centroid of {country_name}!")
        return (0, 0) # Substitute (0,0) if a country cannot be found