# Helper functions related to edge bundling

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from modules.centroids import get_centroid



def compute_cost_weighted_mean(positions, weights):
    """Weighted mean of positions. Minimises sum_i w_i * ||p - x_i||^2.
    Can be swapped for any function with signature:
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
    """Compute optimal bundle points for each (src_cluster, dst_cluster) pair.
    Each bundle point minimizes distance from source cluster mean to destination cluster mean.

    Returns dict: (src_cid, dst_cid) -> (x, y)
    """
    bundle_points = {}

    cluster_means = {}
    for cid, countries in clusters.items():
        valid = [c for c in countries if c in centroids]
        if valid:
            xs = [centroids[c][0] for c in valid]
            ys = [centroids[c][1] for c in valid]
            cluster_means[cid] = (float(np.mean(xs)), float(np.mean(ys)))

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

def compute_bundle_split_points(data, centroids, clusters, cost_fn=None, bundle_points=None, bundle_radius=3.0, split_radius=1.5):
    """Compute bundle and split points for every (src_cluster, dst_cluster) pair.

    Bundle points are optimized per pair to minimize distance between cluster means.
    Split points are optimized within feasible areas around destination countries.

    Returns dict: (src_cid, dst_cid) -> {
        'bundle': (x, y),
        'split': (x, y),
        'src_weights': {country: flow_total},
        'dst_weights': {country: flow_total},
        'total_flow': float
    }
    """
    if cost_fn is None:
        cost_fn = compute_cost_weighted_mean

    if bundle_points is None:
        bundle_points = compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=bundle_radius)

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

            split_pt = _find_optimal_split_point_for_pair(
                bundle_pt, dst_cid, centroids, clusters, radius=split_radius, dst_weights=dst_weights
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
    """Cubic-spline arc between two points.
    Positive offset curves one way, negative the other.
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


def _cubic_bezier(p0, p1, p2, p3, n=50):
    """Evaluate a cubic Bezier curve from four control points.
    Returns (xs, ys) arrays of length n.
    """
    p0, p1, p2, p3 = (np.asarray(p) for p in (p0, p1, p2, p3))
    t = np.linspace(0, 1, n)[:, None]
    curve = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    return curve[:, 0], curve[:, 1]


def _smooth_curve_directed(p0, p1, tangent_out=None, tangent_in=None,
                           strength=0.35, n=50):
    """Cubic Bezier between p0 and p1 with optional tangent constraints.

    tangent_out: direction the curve leaves p0 (defaults to straight line p0 -> p1)
    tangent_in:  direction the curve arrives at p1 from (defaults to straight line p0 -> p1)
    strength:    how far control points sit from endpoints, as fraction of chord length
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    chord = np.linalg.norm(p1 - p0)
    arm = chord * strength

    if tangent_out is not None:
        t_out = np.asarray(tangent_out, float)
        t_out = t_out / (np.linalg.norm(t_out) + 1e-12)
    else:
        t_out = (p1 - p0) / (chord + 1e-12)

    if tangent_in is not None:
        t_in = np.asarray(tangent_in, float)
        t_in = t_in / (np.linalg.norm(t_in) + 1e-12)
    else:
        t_in = (p1 - p0) / (chord + 1e-12)

    cp1 = p0 + t_out * arm
    cp2 = p1 - t_in  * arm
    return _cubic_bezier(p0, cp1, cp2, p1, n=n)


def _segments_cross(a, b, c, d):
    """True if segment a->b properly intersects c->d (not at endpoints)."""
    def _cross2d(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1 = _cross2d(c, d, a)
    d2 = _cross2d(c, d, b)
    d3 = _cross2d(a, b, c)
    d4 = _cross2d(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _compute_trunk_offsets(bs_points, base_offset=0.15):
    """Assign arc-offset signs so crossing trunk pairs arc in opposite directions."""
    keys = list(bs_points.keys())
    offsets = {k: base_offset for k in keys}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            if _segments_cross(
                bs_points[ki]['bundle'], bs_points[ki]['split'],
                bs_points[kj]['bundle'], bs_points[kj]['split'],
            ):
                offsets[kj] = -offsets[ki]
    return offsets


def _draw_curve(ax, xs, ys, color, lw, alpha, arrow=True):
    """Draw a sampled curve, optionally with an arrowhead at the end."""
    if arrow:
        split = max(2, len(xs) - 5)
        ax.plot(xs[:split], ys[:split], color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=3)
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
    else:
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=3)

def _feasible_areas(source_points, radius=3.0):
    """Returns a list of feasible regions as circles (center + radius).
    Currently all circles share the same radius.
    """
    src_arr = np.array(source_points)
    areas = [{'center': src, 'radius': radius} for src in src_arr]
    return areas

def _find_optimal_split_point_for_pair(bundle_pt, dst_cid, centroids, clusters, radius=1.5, dst_weights=None, dst_src_weight_ratio=1.3, min_arc_length=2.0, min_arc_penalty=2.0):
    """Find the optimal split point for a destination cluster.
    Minimizes distance from split point to bundle point while staying in feasible areas.
    Also penalizes positions that would make any destination arc shorter than min_arc_length.

    Parameters
    ----------
    bundle_pt :             (x, y) coordinates of the bundle point.
    dst_cid :               destination cluster ID.
    centroids :             country name -> (x, y).
    clusters :              cluster_id -> list of country names.
    radius :                radius of feasible areas around destination countries.
    dst_weights :           weights for destination countries (optional, used as fallback).
    dst_src_weight_ratio :  only used if radius = 0.
                            How much we value proximity to sink countries vs source countries.
                            Around 1.2-1.3 works well.
    min_arc_length :        minimum desired distance (data coords) from the split point to
                            each destination country. Arcs shorter than this get penalized.
    min_arc_penalty :       strength of the penalty for short arcs.

    Returns (x, y) of the optimal split point.
    """
    if dst_cid not in clusters:
        return bundle_pt

    dst_countries = [c for c in clusters[dst_cid] if c in centroids]
    if not dst_countries:
        return bundle_pt

    dst_points = np.array([centroids[c] for c in dst_countries])
    bundle_pt_arr = np.array(bundle_pt)

    areas = _feasible_areas(dst_points, radius=radius)

    def cost_function(p):
        point = np.array([p[0], p[1]])

        if radius > 0:
            cost = np.linalg.norm(point - bundle_pt_arr)
        else:
            dist_to_dst = np.sum([np.linalg.norm(point - dst_point) for dst_point in dst_points]) / len(dst_points)
            dist_to_src = np.linalg.norm(point - bundle_pt_arr)
            cost = dst_src_weight_ratio * dist_to_dst + dist_to_src

        # Penalize being too close to any destination country
        for dst_point in dst_points:
            d = np.linalg.norm(point - dst_point)
            if d < min_arc_length:
                cost += min_arc_penalty * (min_arc_length - d) ** 2

        return cost

    def project_to_feasible(point):
        """Project point to closest position inside any feasible circle."""
        point = np.array(point)

        for a in areas:
            center = np.array(a['center'])
            r = a['radius']
            if np.linalg.norm(point - center) <= r:
                return point

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

    if dst_weights:
        initial_guess = compute_cost_weighted_mean(centroids, dst_weights)
    else:
        initial_guess = np.mean(dst_points, axis=0)

    result = minimize(cost_function, initial_guess, method='Nelder-Mead')

    if radius > 0:
        final_point = project_to_feasible(result.x)
    else:
        final_point = result.x

    return tuple(final_point)

def _find_optimal_bundle_point_for_pair(src_cid, dst_cid, centroids, clusters, radius, cluster_means, src_dst_weight_ratio=1.2, min_arc_length=2.0, min_arc_penalty=2.0):
    """Find the optimal bundling point for a source cluster targeting a destination cluster.
    Minimizes distance from source cluster mean to destination cluster mean.
    Also penalizes positions that would make any source arc shorter than min_arc_length.

    Parameters
    ----------
    src_cid :               source cluster ID.
    dst_cid :               destination cluster ID.
    centroids :             country name -> (x, y).
    clusters :              cluster_id -> list of country names.
    radius :                radius of feasible areas around source points.
                            If 0, uses a different cost function that also factors in
                            distance to source countries.
    cluster_means :         precomputed cluster_id -> (x, y).
    src_dst_weight_ratio :  only used if radius = 0.
                            How much we value proximity to source countries vs destinations.
                            Around 1.1-1.2 works well if source distance is unweighted.
    min_arc_length :        minimum desired distance (data coords) from the bundle point to
                            each source country. Arcs shorter than this get penalized.
    min_arc_penalty :       strength of the penalty for short arcs.

    Returns (x, y) or None.
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

    areas = _feasible_areas(src_points, radius=radius)

    def cost_function(p):
        point = np.array([p[0], p[1]])

        if radius > 0:
            cost = np.linalg.norm(point - dst_mean)
        else:
            dist_to_src = np.sum([np.linalg.norm(point - src_point) for src_point in src_points]) / len(src_points)
            # Could weight by export: need to supply export data in function arguments
            #dist_to_src = np.sum([np.linalg.norm(point - src_points[i]) * data.loc[src_countries[i], clusters[dst_cid]].sum() for i in range(len(src_countries))]) / data.loc[src_countries, clusters[dst_cid]].sum().sum()
            dist_to_dst = np.linalg.norm(point - dst_mean)
            cost = src_dst_weight_ratio * dist_to_src + dist_to_dst

        # Penalize being too close to any source country
        for src_point in src_points:
            d = np.linalg.norm(point - src_point)
            if d < min_arc_length:
                cost += min_arc_penalty * (min_arc_length - d) ** 2

        return cost

    def project_to_feasible(point):
        """Project point to closest position inside any feasible circle."""
        point = np.array(point)

        for a in areas:
            center = np.array(a['center'])
            r = a['radius']
            if np.linalg.norm(point - center) <= r:
                return point

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

    initial_guess = np.mean(src_points, axis=0)
    result = minimize(cost_function, initial_guess, method='Nelder-Mead')

    if radius > 0:
        final_point = project_to_feasible(result.x)
    else:
        final_point = result.x

    return tuple(final_point)

def matplotlib_map_bundled(gdf, data, centroid_table, clusters, bundle_radius=3.0, split_radius=1.5, show_intra=True, ax=None):
    """Draw a flow map with edge bundling between clusters.

    For each (src_cluster, dst_cluster):
      1. Thin branches from each source country to the bundle point
      2. Straight trunk from bundle point to split point
      3. Thin branches from split point to each destination country
    Intra-cluster flows are drawn as direct curved arrows if show_intra=True.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        return_fig = True
    else:
        return_fig = False

    gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5)

    # Set axis limits before drawing so transData gives correct pixel coords
    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])

    centroids = {}
    for _, row in gdf.iterrows():
        name = row.get('name')
        centroids[name] = get_centroid(centroid_table, name)

    country_to_cluster = {}
    for cid, members in clusters.items():
        for c in members:
            country_to_cluster[c] = cid

    bundle_points = compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=bundle_radius)
    bs = compute_bundle_split_points(data, centroids, clusters, bundle_points=bundle_points, bundle_radius=bundle_radius, split_radius=split_radius)

    n_clusters = len(clusters)
    cmap = plt.cm.get_cmap('Set1', max(n_clusters, 3))
    cluster_colors = {cid: cmap(i) for i, cid in enumerate(sorted(clusters))}

    max_q = data.max(numeric_only=True).max()

    # Force a draw so pixel extents are valid for transData
    ax.get_figure().canvas.draw()

    def _perp_offset(centre_data, offset_pts, perp_disp):
        """Shift centre_data by offset_pts (typographic points) along perp_disp
        (a unit vector in display/pixel space). Returns data coordinates."""
        dpi = ax.get_figure().dpi
        px = offset_pts * dpi / 72.0
        c_disp = ax.transData.transform(centre_data)
        c_disp = c_disp + perp_disp * px
        return ax.transData.inverted().transform(c_disp)

    # Draw inter-cluster bundled flows
    for (src_cid, dst_cid), info in bs.items():
        bundle_pt = np.array(info['bundle'], float)
        split_pt  = np.array(info['split'], float)
        color = cluster_colors.get(dst_cid, 'red')

        trunk_dir = split_pt - bundle_pt

        # Perpendicular in display (pixel) space so it accounts for axis scaling
        b_disp = ax.transData.transform(bundle_pt)
        s_disp = ax.transData.transform(split_pt)
        d_disp = s_disp - b_disp
        d_len  = np.linalg.norm(d_disp)
        d_unit = d_disp / (d_len + 1e-12)
        perp_disp = np.array([-d_unit[1], d_unit[0]])

        def _branch_lw(w):
            return 0.3 + (w / max_q) * 3

        src_branches = [(c, w, _branch_lw(w))
                        for c, w in info['src_weights'].items()
                        if c in centroids]
        dst_branches = [(c, w, _branch_lw(w))
                        for c, w in info['dst_weights'].items()
                        if c in centroids]

        # Sort branches by their projection onto the perpendicular so that
        # left-side countries get left slots and right-side countries get
        # right slots, preventing crossings near the junction.
        def _perp_proj(country):
            pt_disp = ax.transData.transform(centroids[country])
            return np.dot(pt_disp, perp_disp)

        src_branches.sort(key=lambda b: _perp_proj(b[0]))
        dst_branches.sort(key=lambda b: _perp_proj(b[0]))

        trunk_lw = sum(lw for _, _, lw in src_branches) if src_branches else 1.0

        # Source -> bundle: branches arrive side by side at the flat trunk head
        running_pts = 0.0
        for country, _, lw in src_branches:
            centre_pts = running_pts + lw / 2 - trunk_lw / 2
            running_pts += lw
            junction = _perp_offset(bundle_pt, centre_pts, perp_disp)
            xs, ys = _smooth_curve_directed(
                centroids[country], tuple(junction), tangent_in=trunk_dir)
            _draw_curve(ax, xs, ys, color, lw, alpha=0.55, arrow=False)

        # Trunk: straight line from bundle to split with flat ends
        ax.plot([bundle_pt[0], split_pt[0]], [bundle_pt[1], split_pt[1]],
                color=color, lw=trunk_lw, alpha=0.75,
                solid_capstyle='butt', zorder=3)

        # Split -> destination: branches depart side by side from the flat trunk head
        running_pts = 0.0
        for country, _, lw in dst_branches:
            centre_pts = running_pts + lw / 2 - trunk_lw / 2
            running_pts += lw
            junction = _perp_offset(split_pt, centre_pts, perp_disp)
            xs, ys = _smooth_curve_directed(
                tuple(junction), centroids[country], tangent_out=trunk_dir)
            _draw_curve(ax, xs, ys, color, lw, alpha=0.55, arrow=True)

    # Intra-cluster flows as direct arcs
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

    # Country markers
    source_countries = set(data.index)
    dest_countries = set(data.columns)

    # Source countries: squares, sized by total exports
    for country in source_countries:
        if country not in centroids:
            continue
        cid = country_to_cluster.get(country)
        if cid is None:
            continue
        total_export = data.loc[country].sum()
        marker_size = (total_export / max_q) * 5
        color = cluster_colors.get(cid, 'gray')
        ax.plot(*centroids[country], marker='s', markersize=marker_size, color=color, markeredgecolor='black', markeredgewidth=1, zorder=6, alpha=0.7)

    # Destination countries: circles, unit size
    for country in dest_countries:
        if country not in centroids or country in source_countries:
            continue
        cid = country_to_cluster.get(country)
        if cid is None:
            continue
        color = cluster_colors.get(cid, 'gray')
        ax.plot(*centroids[country], marker='o', markersize=5,
                color=color, markeredgecolor='black', markeredgewidth=0.5,
                zorder=6, alpha=0.7)

    # Debug markers for bundle and split points
    for (src_cid, dst_cid), info in bs.items():
        ax.plot(*info['bundle'], 'o', color='black', markersize=5, zorder=5)
        ax.plot(*info['split'],  's', color='black', markersize=5, zorder=5)

    # TODO: REMOVE after debugging
    global_source_pts = [centroids[c] for c in source_countries if c in centroids]
    if global_source_pts:
        areas = _feasible_areas(global_source_pts, radius=3.0)
        for area in areas:
            center = area['center']
            circle_radius = area['radius']
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
