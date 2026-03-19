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

def compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=3.0):
    """
    Compute optimal bundle points for each (src_cluster, dst_cluster) pair.
    Each bundle point minimizes distance from source cluster mean to destination cluster mean.
    
    Returns
    -------
    dict : (src_cid, dst_cid) -> (x, y)
    """
    bundle_points = {}
    
    # Precompute cluster means for all clusters
    cluster_means = {}
    for cid, countries in clusters.items():
        valid = [c for c in countries if c in centroids]
        if valid:
            xs = [centroids[c][0] for c in valid]
            ys = [centroids[c][1] for c in valid]
            cluster_means[cid] = (float(np.mean(xs)), float(np.mean(ys)))
    
    # Optimize bundle point for each source->destination pair
    for src_cid in clusters:
        is_source = any(c in data.index for c in clusters[src_cid])
        if not is_source:
            continue
            
        for dst_cid in clusters:
            if src_cid == dst_cid:
                continue
            if dst_cid not in cluster_means:
                continue
                
            opt_pt = _find_optimal_bundle_point_for_pair(
                src_cid, dst_cid, centroids, clusters, radius, cluster_means
            )
            if opt_pt:
                bundle_points[(src_cid, dst_cid)] = opt_pt
    
    return bundle_points

def compute_bundle_split_points(data, centroids, clusters, cost_fn=None, bundle_points=None):
    """Compute bundle and split points for every (src_cluster, dst_cluster) pair.

    Bundle points are optimized per source-destination pair to minimize distance between cluster means.
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

    # Precompute bundle points per source-destination pair if not provided
    if bundle_points is None:
        bundle_points = compute_cluster_bundle_points_per_pair(data, centroids, clusters)

    result = {}
    for src_cid in clusters:
        for dst_cid in clusters:
            if src_cid == dst_cid:
                continue

            bundle_pt = bundle_points.get((src_cid, dst_cid))
            if bundle_pt is None:
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

def _find_optimal_bundle_point_for_pair(src_cid, dst_cid, centroids, clusters, radius, cluster_means, src_dst_weight_ratio = 1.1):
    """
    Finds the optimal bundling point for a source cluster targeting a destination cluster.
    Minimizes the distance from source cluster mean to destination cluster mean.
    
    Parameters
    ----------
    src_cid : int/str
        The source cluster ID.
    dst_cid : int/str
        The destination cluster ID.
    centroids : dict
        Mapping of country name -> (x, y) coordinates.
    clusters : dict
        Mapping of cluster_id -> list of country names.
    radius : float
        Radius of feasible areas around source points.
        If 0, do not compute or project to feasible regions; 
        instead use a different cost function that factors in distance to source countries as well.
    cluster_means : dict
        Precomputed cluster means: cluster_id -> (x, y).
    src_dst_weight_ratio : float
        Only used if radius = 0. 
        Measures how much we value placing the bundle point close to the source countries over the destination countries.
        Around 1.1-1.2 is a good ratio if source distance is not weighted by export.
        
        
    Returns
    -------
    tuple or None
        Optimal bundle point coordinates, or None if optimization fails.
    """
    if src_cid not in clusters or dst_cid not in clusters:
        return None
    
    src_countries = [c for c in clusters[src_cid] if c in centroids]
    if not src_countries:
        return None
    
    dst_mean = cluster_means.get(dst_cid)
    if not dst_mean:
        return None
    
    src_points = np.array([centroids[c] for c in src_countries])
    
    # Get feasible regions as circles (center + radius)
    areas = _feasible_areas(src_points, radius=radius)
    
    def cost_function(p):
        point = np.array([p[0], p[1]])
        
        ### OLD: Only distance from bundle point to destination cluster mean ###
        if radius > 0:
            cost = np.sqrt((point[0] - dst_mean[0])**2 + (point[1] - dst_mean[1])**2)

        ### NEW: Cost is dependent on distance to source countries (high weight) and destination countries (low weight) ###
        else:
            dist_to_src = np.sum([np.linalg.norm(point - src_point) for src_point in src_points]) / len(src_points) # Unweighted
            # Weighted by export; need to supply export data in function arguments
            #dist_to_src = np.sum([np.linalg.norm(point - src_points[i]) * data.loc[src_countries[i], clusters[dst_cid]].sum() for i in range(len(src_countries))]) / data.loc[src_countries, clusters[dst_cid]].sum().sum() 
            dist_to_dst = np.linalg.norm(point - dst_mean) # Could do sum of distances to all dst countries instead of only to the mean.
            cost = src_dst_weight_ratio * dist_to_src + dist_to_dst
        
        return cost
    
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
                # Project onto circle boundary
                closest = center + (direction / direction_norm) * r
            else:
                # Point is at center, pick any boundary point
                closest = center + np.array([r, 0])
            
            dist = np.linalg.norm(point - closest)
            if dist < best_dist:
                best_dist = dist
                best_point = closest
        
        return best_point if best_point is not None else point
    
    initial_guess = np.mean(src_points, axis=0)
    result = minimize(cost_function, initial_guess, method='Nelder-Mead')
    
    if radius > 0:  # OLD: Project result back into feasible region to ensure it's valid
        final_point = project_to_feasible(result.x)
    else:           # NEW: Don't project
        final_point = result.x

    return tuple(final_point)

def matplotlib_map_bundled(gdf, data, centroid_table, clusters, radius=3.0, show_intra=True, ax=None):
    """Draw a flow map with edge bundling between clusters.

    Rendering per (src_cluster, dst_cluster):
      1. Thin edges: each source country  →  bundle point
      2. One thick edge: bundle point  →  split point   (merged trunk)
      3. Thin edges: split point  →  each destination country
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

    # Pre-calculate bundle points for all source-destination pairs
    bundle_points = compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=radius)
    
    bs = compute_bundle_split_points(data, centroids, clusters, bundle_points=bundle_points)

    # Colour palette
    n_clusters = len(clusters)
    cmap = plt.cm.get_cmap('Set1', max(n_clusters, 3))
    cluster_colors = {cid: cmap(i) for i, cid in enumerate(sorted(clusters))}

    max_q = data.max(numeric_only=True).max()

    # Compute arc-offset signs for trunks to minimise visual crossings
    trunk_offsets = _compute_trunk_offsets(bs)
    
    max_cluster_export = max(info['total_flow'] for info in bs.values()) if bs else 1

    # ── Draw inter-cluster bundled flows ──
    for (src_cid, dst_cid), info in bs.items():
        bundle_pt = info['bundle']
        split_pt  = info['split']
        color = cluster_colors.get(dst_cid, 'red')
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
            ax.add_patch(circle)

    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])
    ax.set_xticks([])
    ax.set_yticks([])
    
    if return_fig:
        plt.tight_layout()
        plt.savefig("map.png")
    
    return ax
