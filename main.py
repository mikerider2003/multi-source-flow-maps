import geopandas as gpd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from modules.baseline import main_baseline    
from modules.edge_bundling import matplotlib_map_bundled
import modules.clustering as clustering



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



# ── Edge-bundling helpers ─────────────────────────────────────────────────

def compute_cost_weighted_mean(positions, weights):
    """Default cost function: weighted mean of positions.

    Minimises  sum_i  w_i * ||p - x_i||^2.
    You can swap this out for any function with the same signature:
        (positions: dict[str, (x,y)], weights: dict[str, float]) -> (x, y)
    """
    total_w = sum(weights.values())
    if total_w == 0:
        xs = [positions[c][0] for c in weights]
        ys = [positions[c][1] for c in weights]
        return (float(np.mean(xs)), float(np.mean(ys)))
    x = sum(weights[c] * positions[c][0] for c in weights) / total_w
    y = sum(weights[c] * positions[c][1] for c in weights) / total_w
    return (x, y)


def compute_cluster_bundle_points(centroids, clusters):
    """Compute a single bundle point per cluster as the mean of all countries.

    Returns
    -------
    dict : cluster_id -> (x, y)
    """
    bundle_points = {}
    for cid, countries in clusters.items():
        valid = [c for c in countries if c in centroids]
        if valid:
            xs = [centroids[c][0] for c in valid]
            ys = [centroids[c][1] for c in valid]
            bundle_points[cid] = (float(np.mean(xs)), float(np.mean(ys)))
    return bundle_points


def compute_bundle_split_points(data, centroids, clusters, cost_fn=None):
    """Compute bundle and split points for every (src_cluster, dst_cluster) pair.

    Bundle points are precomputed once per source cluster (mean of countries).
    Split points are computed per destination for each pair.

    Returns
    -------
    dict : (src_cid, dst_cid) -> {
        'bundle': (x, y),
        'split': (x, y),
        'src_weights': {country: flow_total},
        'dst_weights': {country: flow_total},
        'total_flow': float
    }
    """
    if cost_fn is None:
        cost_fn = compute_cost_weighted_mean

    # Precompute bundle points per source cluster
    bundle_points = compute_cluster_bundle_points(centroids, clusters)

    result = {}
    for src_cid in clusters:
        for dst_cid in clusters:
            if src_cid == dst_cid:
                continue

            src_weights = {}
            for s in clusters[src_cid]:
                if s not in data.index or s not in centroids:
                    continue
                w = sum(data.loc[s, d] for d in clusters[dst_cid] if d in data.columns)
                if w > 0:
                    src_weights[s] = w

            dst_weights = {}
            for d in clusters[dst_cid]:
                if d not in data.columns or d not in centroids:
                    continue
                w = sum(data.loc[s, d] for s in clusters[src_cid] if s in data.index)
                if w > 0:
                    dst_weights[d] = w

            if not src_weights or not dst_weights:
                continue

            result[(src_cid, dst_cid)] = {
                'bundle': bundle_points[src_cid],
                'split':  cost_fn(centroids, dst_weights),
                'src_weights': src_weights,
                'dst_weights': dst_weights,
                'total_flow': sum(src_weights.values()),
            }
    return result


def _smooth_curve(p0, p1, offset=0.08, n=40):
    """Return a gentle cubic-spline arc between two points.

    Positive offset curves the arc one way, negative curves the other.
    This sign is exploited by _compute_trunk_offsets to separate crossing trunks.
    """
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ctrl = (mid[0] - dy * offset, mid[1] + dx * offset)
    pts = np.array([p0, ctrl, p1])
    t = np.array([0, 0.5, 1.0])
    t_smooth = np.linspace(0, 1, n)
    cs_x = CubicSpline(t, pts[:, 0], bc_type='clamped')
    cs_y = CubicSpline(t, pts[:, 1], bc_type='clamped')
    return cs_x(t_smooth), cs_y(t_smooth)


def _segments_cross(a, b, c, d):
    """Return True if segment a→b properly (not at endpoints) intersects c→d."""
    def _cross2d(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1 = _cross2d(c, d, a)
    d2 = _cross2d(c, d, b)
    d3 = _cross2d(a, b, c)
    d4 = _cross2d(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _compute_trunk_offsets(bs_points, base_offset=0.15):
    """Assign an arc-offset sign to every trunk so that crossing pairs arc
    in opposite directions, visually separating them.

    This is the crossing-minimization cost function: for each pair of trunk
    segments whose straight-line extents intersect, one gets +base_offset and
    the other gets -base_offset.  Swap or extend this logic to implement a
    different crossing-cost criterion (e.g. a continuous penalty minimized via
    scipy.optimize).
    """
    keys = list(bs_points.keys())
    offsets = {k: base_offset for k in keys}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            if _segments_cross(
                bs_points[ki]['bundle'], bs_points[ki]['split'],
                bs_points[kj]['bundle'], bs_points[kj]['split'],
            ):
                # Arc the two crossing trunks in opposite directions
                offsets[kj] = -offsets[ki]
    return offsets


def _draw_curve_with_arrow(ax, xs, ys, color, lw, alpha):
    """Draw a precomputed curve with a filled arrowhead at its tip."""
    split = max(2, len(xs) - 5)
    # Curve body
    ax.plot(xs[:split], ys[:split], color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=3)
    # Final short segment + arrowhead
    ax.annotate(
        "",
        xy=(xs[-1], ys[-1]),
        xytext=(xs[split - 1], ys[split - 1]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            alpha=alpha,
            mutation_scale=6 + lw * 2,
        ),
        zorder=4,
    )


def matplotlib_map_bundled(gdf, data, centroid_table, clusters, show_intra=True):
    """Draw a flow map with edge bundling between clusters.

    Rendering per (src_cluster, dst_cluster):
      1. Thin edges: each source country  →  bundle point
      2. One thick edge: bundle point  →  split point   (merged trunk)
      3. Thin edges: split point  →  each destination country
    Intra-cluster flows are drawn as direct curved arrows if show_intra=True.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5)

    # Build centroid lookup
    centroids = {}
    for _, row in gdf.iterrows():
        name = row.get('name')
        centroids[name] = get_centroid(centroid_table, name)

    # Country -> cluster mapping
    country_to_cluster = {}
    for cid, members in clusters.items():
        for c in members:
            country_to_cluster[c] = cid

    bs = compute_bundle_split_points(data, centroids, clusters)

    # Colour palette
    n_clusters = len(clusters)
    cmap = plt.cm.get_cmap('Set1', max(n_clusters, 3))
    cluster_colors = {cid: cmap(i) for i, cid in enumerate(sorted(clusters))}

    max_q = data.max(numeric_only=True).max()

    # Compute arc-offset signs for trunks to minimise visual crossings
    trunk_offsets = _compute_trunk_offsets(bs)

    # ── Draw inter-cluster bundled flows ──
    for (src_cid, dst_cid), info in bs.items():
        bundle_pt = info['bundle']
        split_pt  = info['split']
        color = cluster_colors.get(src_cid, 'red')
        trunk_offset = trunk_offsets[(src_cid, dst_cid)]

        # 1) Thin legs: each source → bundle point  (arrow at bundle end)
        for country, w in info['src_weights'].items():
            if country not in centroids:
                continue
            lw = 0.3 + (w / max_q) * 3
            xs, ys = _smooth_curve(centroids[country], bundle_pt)
            _draw_curve_with_arrow(ax, xs, ys, color, lw, alpha=0.55)

        # 2) Thick trunk: bundle point → split point  (crossing-minimising arc)
        trunk_lw = 0.5 + (info['total_flow'] / max_q) * 5
        xs, ys = _smooth_curve(bundle_pt, split_pt, offset=trunk_offset)
        _draw_curve_with_arrow(ax, xs, ys, color, trunk_lw, alpha=0.75)

        # 3) Thin legs: split point → each destination  (arrow at destination)
        for country, w in info['dst_weights'].items():
            if country not in centroids:
                continue
            lw = 0.3 + (w / max_q) * 3
            xs, ys = _smooth_curve(split_pt, centroids[country])
            _draw_curve_with_arrow(ax, xs, ys, color, lw, alpha=0.55)

    # ── Draw intra-cluster flows as direct arcs ──
    if show_intra:
        for src, row_data in data.iterrows():
            src_cid = country_to_cluster.get(src)
            if src_cid is None or src not in centroids:
                continue
            for dst, qty in row_data.items():
                if qty == 0 or dst not in centroids or src == dst:
                    continue
                dst_cid = country_to_cluster.get(dst)
                if dst_cid is None or src_cid != dst_cid:
                    continue
                lw = 0.3 + (qty / max_q) * 3
                color = cluster_colors.get(src_cid, 'red')
                ax.annotate(
                    "", xy=centroids[dst], xytext=centroids[src],
                    arrowprops=dict(
                        arrowstyle="-|>", lw=lw,
                        color=color, alpha=0.55,
                        connectionstyle="arc3,rad=0.2",
                        mutation_scale=6 + lw * 2,
                    )
                )

    # ── Draw country markers ──
    # Countries that export (source cluster)
    source_countries = set(data.index)  
    # All destination countries
    dest_countries = set(data.columns)  

    # Source countries: squares, sized by total exports
    for country in source_countries:
        if country not in centroids:
            continue
        cid = country_to_cluster.get(country)
        if cid is None:
            continue

        # Total exports from this country
        total_export = data.loc[country].sum()
        marker_size = (total_export / max_q) * 5
        color = cluster_colors.get(cid, 'gray')

        ax.plot(*centroids[country], marker='s', markersize=marker_size, color=color, markeredgecolor='black', markeredgewidth=1, zorder=6, alpha=0.7)

    # Destination countries: circles, unit size
    for country in dest_countries:
        if country not in centroids or country in source_countries:
            # Skip if not found or already drawn as source
            continue
        cid = country_to_cluster.get(country)
        if cid is None:
            continue

        color = cluster_colors.get(cid, 'gray')
        ax.plot(*centroids[country], marker='o', markersize=5,
                color=color, markeredgecolor='black', markeredgewidth=0.5,
                zorder=6, alpha=0.7)

    # Debug markers: bundle (●) and split (■) points
    for (src_cid, dst_cid), info in bs.items():
        ax.plot(*info['bundle'], 'o', color='black', markersize=5, zorder=5)
        ax.plot(*info['split'],  's', color='black', markersize=5, zorder=5)

    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# ── Clustering helpers ────────────────────────────────────────────────────

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
    gdf = gpd.read_file("data/geo.json")

    # Load country centroid table
    centroid_table = pd.read_csv("data/centroids.csv")

    # Always use the full dataset for clustering
    data = pd.read_csv("data/EU_trade_data_full.csv", sep=';', thousands='.', header=0, index_col=0)

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

    clusters = clustering.cluster_countries(centroid_table, countries, n)
    source_countries = clustering.select_source_cluster(clusters)

    print(f"\nSource countries: {', '.join(sorted(source_countries))}")

    # Filter data to only flows originating from the selected cluster
    filtered = clustering.filter_data_by_sources(data, source_countries)
    print(f"Showing exports from {len(filtered)} source countries to {len(filtered.columns)} destinations.\n")

    show_intra = input("Show intra-cluster edges (within the source cluster)? [y/n]: ").strip().lower() != 'n'

    matplotlib_map_bundled(gdf, filtered, centroid_table, clusters, show_intra=show_intra)


if __name__ == "__main__":
    # Set mode to "clustered" for interactive clustering, or a dataset name for a pre-made subset
    mode = "clustered"  # Options: "clustered", "full", "distant", "close", "2_clusters", "3_clusters", "5_clusters"

    if mode == "clustered":
        main_clustered()
    else:
        main_baseline(mode)
