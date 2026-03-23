# Helper functions related to edge bundling

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from modules.centroids import get_centroid



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

def compute_cluster_bundle_point(data, centroids, clusters, radius=3.0):
    """
    Compute a single bundle point per cluster. Uses optimization for source 
    clusters, or falls back to geographic mean for others.
    """
    bundle_points = {}
    
    # Pre-calculate simple means first because the optimizer needs them!
    # (The optimizer needs 'other' bundle points to measure distance to)
    temp_points = {}
    for cid, countries in clusters.items():
        valid = [c for c in countries if c in centroids]
        if valid:
            xs = [centroids[c][0] for c in valid]
            ys = [centroids[c][1] for c in valid]
            temp_points[cid] = (float(np.mean(xs)), float(np.mean(ys)))
            
    # Optimize
    for cid in clusters:
        is_source = any(c in data.index for c in clusters[cid])
        if is_source:
             opt_pt = _find_optimal_bundle_point_src(cid, data, centroids, clusters, radius, temp_points)
             if opt_pt:
                 bundle_points[cid] = opt_pt
             else:
                 bundle_points[cid] = temp_points.get(cid) 
        else:
             bundle_points[cid] = temp_points.get(cid)
             
    return bundle_points

def compute_bundle_split_points(data, centroids, clusters, cost_fn=None, bundle_points=None):
    """Compute bundle and split points for every (src_cluster, dst_cluster) pair.

    Bundle points are precomputed once per source cluster (mean of countries).
    Split points are optimized within feasible areas around destination countries.

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

    # Precompute bundle points per source cluster if not provided
    if bundle_points is None:
        bundle_points = compute_cluster_bundle_point(data, centroids, clusters)

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

            bundle_pt = bundle_points[src_cid]
            # Optimize split point within feasible areas around destination countries
            split_pt = _find_optimal_split_point_for_pair(
                bundle_pt, dst_cid, centroids, clusters, radius=1.5, dst_weights=dst_weights
            )

            result[(src_cid, dst_cid)] = {
                'bundle': bundle_pt,
                'split': split_pt,
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
    """Return True if segment a->b properly (not at endpoints) intersects c->d."""
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
    the other gets -base_offset. 
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

def _feasible_areas(source_points, radius=3.0):
    """
    Returns a list of feasible regions as circles, each defined by a center
    and a radius. Currently all circles have the same radius, but this can be
    extended later (e.g., different radii per source point).
    """
    src_arr = np.array(source_points)
    areas = [{'center': src, 'radius': radius} for src in src_arr]
    return areas

def _find_optimal_split_point_for_pair(bundle_pt, dst_cid, centroids, clusters, radius=1.5, dst_weights=None):
    """
    Finds the optimal split point for a destination cluster.
    Minimizes distance from split point to bundle point while staying within feasible areas.
    
    Parameters
    ----------
    bundle_pt : tuple
        The bundle point (x, y) coordinates.
    dst_cid : int/str
        The destination cluster ID.
    centroids : dict
        Mapping of country name -> (x, y) coordinates.
    clusters : dict
        Mapping of cluster_id -> list of country names.
    radius : float
        Radius of feasible areas around destination countries.
    dst_weights : dict, optional
        Weights for destination countries (for fallback).
        
    Returns
    -------
    tuple
        Optimal split point coordinates.
    """
    if dst_cid not in clusters:
        return bundle_pt
    
    dst_countries = [c for c in clusters[dst_cid] if c in centroids]
    if not dst_countries:
        return bundle_pt
    
    dst_points = np.array([centroids[c] for c in dst_countries])
    bundle_pt_arr = np.array(bundle_pt)
    
    # Get feasible regions as circles
    areas = _feasible_areas(dst_points, radius=radius)
    
    def cost_function(p):
        point = np.array([p[0], p[1]])
        # Minimize distance to bundle point
        dist_to_bundle = np.sqrt((point[0] - bundle_pt_arr[0])**2 + (point[1] - bundle_pt_arr[1])**2)
        return dist_to_bundle
    
    def project_to_feasible(point):
        """Project point into feasible region (closest point in any circle)."""
        point = np.array(point)
        
        # Check if already in feasible region
        for a in areas:
            center = np.array(a['center'])
            r = a['radius']
            if np.linalg.norm(point - center) <= r:
                return point
        
        # Find closest point on boundary of any circle
        best_point = None
        best_dist = float('inf')
        
        for a in areas:
            center = np.array(a['center'])
            r = a['radius']
            direction = point - center
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                closest = center + (direction / direction_norm) * r
            else:
                closest = center + np.array([r, 0])
            
            dist = np.linalg.norm(point - closest)
            if dist < best_dist:
                best_dist = dist
                best_point = closest
        
        return best_point if best_point is not None else point
    
    # Start from weighted mean of destination countries or bundle point
    if dst_weights:
        initial_guess = compute_cost_weighted_mean(centroids, dst_weights)
    else:
        initial_guess = np.mean(dst_points, axis=0)
    
    result = minimize(cost_function, initial_guess, method='Nelder-Mead')
    final_point = project_to_feasible(result.x)
    return tuple(final_point)

def _compute_distances_from_source(src_cid, bundle_points):
    """
    Calculate the distance between the source cluster's bundle point 
    and all other clusters' bundle points.
    
    Parameters
    ----------
    src_cid : int/str
        The ID of the source cluster.
    bundle_points : dict
        Mapping of cluster_id -> (x, y) coordinates of bundle points.
        
    Returns
    -------
    dict
        Mapping of dst_cid -> float (Euclidean distance)
    """
    distances = {}
    if src_cid not in bundle_points:
        return distances
        
    p_src = bundle_points[src_cid]
    
    for dst_cid, p_dst in bundle_points.items():
        if src_cid == dst_cid:
            continue
            
        # Euclidean distance between the two points
        dist = np.sqrt((p_src[0] - p_dst[0])**2 + (p_src[1] - p_dst[1])**2)
        distances[dst_cid] = dist
        
    return distances

def _compute_exports_from_source(src_cid, data, clusters):
    """
    Calculate the total export amount from the source cluster to all other clusters.
    
    Parameters
    ----------
    src_cid : int/str
        The ID of the source cluster.
    data : pd.DataFrame
        Trade matrix where index is origin (exports) and columns are destinations.
    clusters : dict
        Mapping of cluster_id -> list of countries.
        
    Returns
    -------
    dict
        Mapping of dst_cid -> float (total export volume)
    """
    weights = {}
    if src_cid not in clusters:
        return weights
        
    # Get valid source countries available in the dataframe index
    src_countries = [c for c in clusters[src_cid] if c in data.index]
    if not src_countries:
        return weights
        
    for dst_cid, dst_countries_list in clusters.items():
        if src_cid == dst_cid:
            continue
            
        # Get valid destination countries available in the dataframe columns
        dst_countries = [c for c in dst_countries_list if c in data.columns]
        
        if not dst_countries:
            continue
            
        # Sum the exports from all source countries to all destination countries
        total_export = data.loc[src_countries, dst_countries].sum().sum()
        weights[dst_cid] = float(total_export)
            
    return weights

def _find_optimal_bundle_point_src(src_cid, data, centroids, clusters, radius, temp_bundle_points):
    # TODO: Im not 100% happy with this func will need to change this
    """
    Finds the optimal bundling point for a source cluster.
    """
    if src_cid not in clusters:
        return None

    cluster_exports = _compute_exports_from_source(src_cid, data, clusters)
    src_countries = [c for c in clusters[src_cid] if c in data.index and c in centroids]
    if not src_countries:
        return None
        
    src_points = np.array([centroids[c] for c in src_countries])
    
    # Get feasible regions as circles (center + radius)
    areas = _feasible_areas(src_points, radius=radius)

    def cost_function(p):
        cost = 0
        point = (p[0], p[1])
        
        # We need a copy so we don't mess up other optimizers running
        test_points = temp_bundle_points.copy()
        test_points[src_cid] = point
        
        dists = _compute_distances_from_source(src_cid, test_points)
        
        max_export = max(cluster_exports.values()) if cluster_exports else 1
        
        for dst_cid, dist in dists.items():
            weight = cluster_exports.get(dst_cid, 1) / max_export
            cost += dist * weight
            
        # Constraint: point must be inside at least one feasible circle
        in_any_area = False
        for a in areas:
            center = a['center']
            r = a['radius']
            if np.linalg.norm(point - center) <= r:
                in_any_area = True
                break
                
        if not in_any_area:
            cost += 9999999
            
        return cost

    initial_guess = np.mean(src_points, axis=0)
    result = minimize(cost_function, initial_guess, method='Nelder-Mead')
    return tuple(result.x)

def matplotlib_map_bundled(gdf, data, centroid_table, clusters, show_intra=True, ax=None):
    """Draw a flow map with edge bundling between clusters.

    Rendering per (src_cluster, dst_cluster):
      1. Thin edges: each source country  ->  bundle point
      2. One thick edge: bundle point  ->  split point   (merged trunk)
      3. Thin edges: split point  ->  each destination country
    Intra-cluster flows are drawn as direct curved arrows if show_intra=True.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        return_fig = True
    else:
        return_fig = False
        
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

    # Pre-calculate bundle points, distances and exports so they're only computed once
    bundle_points = compute_cluster_bundle_point(data, centroids, clusters)
    
    bs = compute_bundle_split_points(data, centroids, clusters, bundle_points=bundle_points)

    # Colour palette
    n_clusters = len(clusters)
    cmap = plt.cm.get_cmap('Set1', max(n_clusters, 3))
    cluster_colors = {cid: cmap(i) for i, cid in enumerate(sorted(clusters))}

    max_q = data.max(numeric_only=True).max()

    # Compute arc-offset signs for trunks to minimise visual crossings
    trunk_offsets = _compute_trunk_offsets(bs)

    max_cluster_export = max(info['total_flow'] for info in bs.values()) if bs else 1
    
    # Cache to store distances and exports
    cluster_dists = {}
    cluster_exports = {}
    
    for src_cid in clusters:
        cluster_dists[src_cid] = _compute_distances_from_source(src_cid, bundle_points)
        cluster_exports[src_cid] = _compute_exports_from_source(src_cid, data, clusters)

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

        # TODO: For debuging only REMOVE LATER Plot distance and text to map
        dists = cluster_dists.get(src_cid, {})
        exports = cluster_exports.get(src_cid, {})
        
        if dst_cid in dists and dst_cid in exports:
            euclidean_dist = dists[dst_cid]
            
            # SCALING MATH: (current_export / max_cluster_export) * 100
            scaled_export = (exports[dst_cid] / max_cluster_export) * 100

            # Format the text box 
            label_text = f"Dist: {euclidean_dist:.1f}\nExp: {scaled_export:.0f}"
            
            # Draw text near the split point
            ax.text(split_pt[0] + 0.5, split_pt[1] + 0.5, label_text, 
                    fontsize=7, 
                    color='black', 
                    family='monospace',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                    zorder=10)

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

    # TODO: REMOVE Vis after debugging
    # ── Draw Feasible Areas as Circles (Debugging) ──
    # Gather ALL source points globally to ensure exactly 1 circle per source
    global_source_pts = [centroids[c] for c in source_countries if c in centroids]

    if global_source_pts:
        # Calculate feasible areas – now returns circles directly
        areas = _feasible_areas(global_source_pts, radius=3.0) 
        
        for area in areas:
            center = area['center']
            circle_radius = area['radius']
            # Create and add a single green circle patch per source point
            circle = Circle(
                (center[0], center[1]), circle_radius,
                facecolor='green', alpha=0.3, edgecolor='green', zorder=2
            )
            #ax.add_patch(circle)

    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])
    ax.set_xticks([])
    ax.set_yticks([])
    
    if return_fig:
        plt.tight_layout()
        plt.savefig("map.png")
        
    return ax
