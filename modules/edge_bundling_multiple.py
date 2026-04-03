# Helper functions related to edge bundling

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from modules.centroids import get_centroid


# Pre-defined colors 
CLUSTER_COLORS = [
    '#7F77DD',  # purple
    '#1D9E75',  # teal
    '#D85A30',  # coral/orange
    '#378ADD',  # blue
    '#D4537E',  # pink
    '#639922',  # green
    '#E8A735',  # amber
    '#8B5CF6',  # violet
    '#06B6D4',  # cyan
    '#F97316',  # orange
    '#14B8A6',  # teal-green
    '#A855F7',  # bright violet
    '#EF4444',  # red
    '#3B82F6',  # sky blue
    '#84CC16',  # lime
    '#F59E0B',  # yellow-amber
    '#EC4899',  # hot pink
    '#10B981',  # emerald
    '#6366F1',  # indigo
    '#F43F5E',  # rose
]

CLUSTER_EDGE_COLORS = [
    '#534AB7',
    '#0F6E56',
    '#993C1D',
    '#185FA5',
    '#993556',
    '#3B6D11',
    '#B8841A',
    '#6D28D9',
    '#0891B2',
    '#C2410C',
    '#0D9488',
    '#7E22CE',
    '#B91C1C',
    '#1D4ED8',
    '#4D7C0F',
    '#B45309',
    '#BE185D',
    '#047857',
    '#4338CA',
    '#BE123C',
]

# ISO 2-letter country codes for labeling
COUNTRY_ISO2 = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czechia': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
    'Hungary': 'HU', 'Ireland': 'IE', 'Italy': 'IT', 'Latvia': 'LV',
    'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT', 'Netherlands': 'NL',
    'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Slovakia': 'SK',
    'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE',
}


def compute_cost_weighted_mean(positions, weights):
    """Weighted mean of positions. Minimises sum_i w_i * ||p - x_i||^2.

    Returns (x, y) tuple.
    """
    total_w = sum(weights.values())
    if total_w == 0:
        xs = [positions[c][0] for c in weights]
        ys = [positions[c][1] for c in weights]
        return (float(np.mean(xs)), float(np.mean(ys)))
    x = sum(weights[c] * positions[c][0] for c in weights) / total_w
    y = sum(weights[c] * positions[c][1] for c in weights) / total_w
    return (x, y)

def compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=3.0, q2_weight=0.3, q3_weight=0.15):
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
                src_cid, dst_cid, centroids, clusters, radius, cluster_means,
                q2_weight=q2_weight, q3_weight=q3_weight
            )
            if opt_pt:
                bundle_points[(src_cid, dst_cid)] = opt_pt

    return bundle_points

def compute_bundle_split_points(data, centroids, clusters, cost_fn=None, bundle_points=None, bundle_radius=3.0, split_radius=1.5, estimated_exports=None, q2_weight=0.3, q3_weight=0.15, distance_scores=None):
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
    
    Also updates estimated exports table with data from the given source cluster. (Note that although the code may suggest that we're going over all countries,
    in reality "data" only contains export data for *one* source cluster)
    """
    if cost_fn is None:
        cost_fn = compute_cost_weighted_mean

    if bundle_points is None:
        bundle_points = compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=bundle_radius, q2_weight=q2_weight, q3_weight=q3_weight)

    result = {}
    sum_of_min_dist = 0 # Distance between bundle/split point and nearest country, summed over all clusters

    for src_cid in clusters:
        for dst_cid in clusters:
            if src_cid == dst_cid:
                continue

            bundle_pt = bundle_points.get((src_cid, dst_cid))
            if bundle_pt is None:
                continue

            total_cluster_export = 0 # Total export between these clusters (the sum([[export[s,d] for s in source_countries] for d in destination_countries]) from the pseudocode)

            src_weights = {}
            for s in clusters[src_cid]:
                if s not in data.index or s not in centroids:
                    continue
                w = sum(data.loc[s, d] for d in clusters[dst_cid] if d in data.columns) # This is sum([export[A, d] for d in destination_countries]) from the pseudocode above
                total_cluster_export += w                                               # Summing this over all A gives us sum([[export[s,d] for s in source_countries] for d in destination_countries])

                for d in clusters[dst_cid]:
                    if d not in data.columns or d not in centroids:
                        continue
                    estimated_exports[(s, d)] = w

                if w > 0:
                    src_weights[s] = w

            dst_weights = {}
            for d in clusters[dst_cid]:
                if d not in data.columns or d not in centroids:
                    continue
                w = sum(data.loc[s, d] for s in clusters[src_cid] if s in data.index) # This is sum([export[s, B] for s in source_countries]) from the pseudocode above

                for s in clusters[src_cid]:
                    if s not in data.index or s not in centroids:
                        continue
                    estimated_exports[(s,d)] *= w / total_cluster_export # total_cluster_export has also been fully calculated now
                    #print(f"Estimated export {s} to {d}: {estimated_exports[(s,d)]}")
                    estimated_exports[(s,d)] /= data.loc[s,d]  # Divide by actual export to get over/underestimation factor

                if w > 0:
                    dst_weights[d] = w

            if not src_weights or not dst_weights:
                continue

            split_pt = _find_optimal_split_point_for_pair(
                bundle_pt, dst_cid, centroids, clusters, radius=split_radius, dst_weights=dst_weights,
                q2_weight=q2_weight, q3_weight=q3_weight
            )

            sum_of_min_dist += compute_dist_to_closest_country(bundle_pt, src_cid, clusters, centroids)
            sum_of_min_dist += compute_dist_to_closest_country(split_pt, dst_cid, clusters, centroids)

            result[(src_cid, dst_cid)] = {
                'bundle': bundle_pt,
                'split': split_pt,
                'src_weights': src_weights,
                'dst_weights': dst_weights,
                'total_flow': sum(src_weights.values()),
            }

    print("BUNDLE/SPLIT DISTANCE SCORE (Q3): ", sum_of_min_dist)

    if distance_scores is not None:
        distance_scores.append(sum_of_min_dist)

    # --- Q1 refinement: reduce trunk crossings by perturbing bundle/split points ---
    result = _refine_crossing_reduction(result, centroids, clusters, bundle_radius, split_radius)

    return result


def _count_trunk_crossings(bs_points):
    """Count the total number of trunk-trunk crossings across all cluster pairs."""
    keys = list(bs_points.keys())
    crossings = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            if _segments_cross(
                bs_points[ki]['bundle'], bs_points[ki]['split'],
                bs_points[kj]['bundle'], bs_points[kj]['split'],
            ):
                crossings += 1
    return crossings


def _refine_crossing_reduction(result, centroids, clusters, bundle_radius, split_radius,
                                max_iters=20, perturb_scale=0.5):
    """Post-processing pass that perturbs bundle/split points to reduce trunk crossings.

    For each pair that participates in a crossing, we try small perturbations
    of its bundle and split points (perpendicular to the trunk direction) and
    keep the perturbation if it reduces total crossings without violating
    feasibility constraints.
    """
    initial_crossings = _count_trunk_crossings(result)
    if initial_crossings == 0:
        return result

    best_crossings = initial_crossings

    for _ in range(max_iters):
        improved = False
        keys = list(result.keys())

        for ki in keys:
            info_i = result[ki]
            bundle_i = np.array(info_i['bundle'], float)
            split_i = np.array(info_i['split'], float)
            trunk_dir = split_i - bundle_i
            trunk_len = np.linalg.norm(trunk_dir)
            if trunk_len < 1e-8:
                continue
            perp = np.array([-trunk_dir[1], trunk_dir[0]]) / trunk_len

            # Try perturbing bundle and split points perpendicular to trunk
            for sign in [1.0, -1.0]:
                for target in ['bundle', 'split']:
                    orig = np.array(info_i[target], float)
                    candidate = orig + sign * perturb_scale * perp

                    # Check feasibility: must be within radius of at least one country
                    src_cid, dst_cid = ki
                    if target == 'bundle':
                        cluster_countries = [c for c in clusters[src_cid] if c in centroids]
                        rad = bundle_radius
                    else:
                        cluster_countries = [c for c in clusters[dst_cid] if c in centroids]
                        rad = split_radius

                    if rad > 0:
                        in_feasible = any(
                            np.linalg.norm(candidate - np.array(centroids[c])) <= rad
                            for c in cluster_countries
                        )
                        if not in_feasible:
                            continue

                    # Test if this reduces crossings
                    old_val = info_i[target]
                    info_i[target] = tuple(candidate)
                    new_crossings = _count_trunk_crossings(result)

                    if new_crossings < best_crossings:
                        best_crossings = new_crossings
                        improved = True
                    else:
                        info_i[target] = old_val

            if best_crossings == 0:
                break

        if not improved or best_crossings == 0:
            break
        perturb_scale *= 0.8  # Decrease perturbation for finer refinement

    if best_crossings < initial_crossings:
        print(f"  Q1 refinement: trunk crossings {initial_crossings} -> {best_crossings}")

    return result

def compute_dist_to_closest_country(point, cid, clusters, centroids):
    """Compute distance between given (bundle/split) point and the closest country in the corresponding cluster (given by cid)"""

    min_dist = np.inf
    countries = [c for c in clusters[cid] if c in centroids]
    country_points = np.array([centroids[c] for c in countries])

    for c_point in country_points:
        min_dist = min(min_dist, np.linalg.norm(point - c_point))

    return min_dist



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


def _arc_length(xs, ys):
    """Compute the arc length of a sampled polyline given by xs, ys arrays."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))


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


def _curves_intersect(xs_a, ys_a, xs_b, ys_b):
    """Vectorised check whether two sampled polylines properly cross."""
    ax1, ay1 = xs_a[:-1, None], ys_a[:-1, None]
    ax2, ay2 = xs_a[1:, None],  ys_a[1:, None]
    bx1, by1 = xs_b[None, :-1], ys_b[None, :-1]
    bx2, by2 = xs_b[None, 1:],  ys_b[None, 1:]

    def _cross(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = _cross(bx1, by1, bx2, by2, ax1, ay1)
    d2 = _cross(bx1, by1, bx2, by2, ax2, ay2)
    d3 = _cross(ax1, ay1, ax2, ay2, bx1, by1)
    d4 = _cross(ax1, ay1, ax2, ay2, bx2, by2)
    return bool(np.any(((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))))


def _count_fan_crossings(fan_pt, trunk_dir, perp, trunk_w,
                         ordered_branches, centroids, is_source=False):
    """Count actual curve–curve crossings for *ordered_branches*.

    Generates Bézier curves at the junction positions implied by the
    branch order and trunk geometry, then checks every pair for a
    proper intersection.
    """
    fan_pt = np.asarray(fan_pt, float)
    curves = []
    running = 0.0
    for country, _, dw in ordered_branches:
        offset = running + dw / 2 - trunk_w / 2
        running += dw
        junction = tuple(fan_pt + perp * offset)
        dest = centroids[country]
        if is_source:
            xs, ys = _smooth_curve_directed(dest, junction,
                                            tangent_in=trunk_dir, n=20)
        else:
            xs, ys = _smooth_curve_directed(junction, dest,
                                            tangent_out=trunk_dir, n=20)
        curves.append((np.asarray(xs), np.asarray(ys)))

    crossings = 0
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            if _curves_intersect(curves[i][0], curves[i][1],
                                 curves[j][0], curves[j][1]):
                crossings += 1
    return crossings


def _minimize_fan_crossings(fan_pt, trunk_dir, branches, centroids,
                            trunk_w, is_source=False):
    """Return branches reordered to minimise actual curve crossings.

    1. Angular sort as a fast initial guess.
    2. Generate the real Bézier curves and count crossings.
    3. If crossings remain and n <= 7, try every permutation.
    4. For larger n, refine with adjacent-swap passes.

    Returns the best sorting of branches as well as the number of crossings achieved by this sorting
    """
    from itertools import permutations as _perms

    n = len(branches)
    if n <= 1:
        return branches

    fan_pt = np.asarray(fan_pt, float)
    trunk_dir = np.asarray(trunk_dir, float)
    trunk_unit = trunk_dir / (np.linalg.norm(trunk_dir) + 1e-12)
    perp = np.array([-trunk_unit[1], trunk_unit[0]])

    # initial guess: angular sort around the fan point 
    def _angle(branch):
        v = np.array(centroids[branch[0]]) - fan_pt
        return np.arctan2(np.dot(v, perp), np.dot(v, trunk_unit))

    best = sorted(branches, key=_angle)
    best_c = _count_fan_crossings(fan_pt, trunk_dir, perp, trunk_w,
                                  best, centroids, is_source)
    if best_c == 0:
        return (best, best_c)

    # exhaustive search for small n 
    if n <= 7:
        for perm in _perms(range(n)):
            candidate = [branches[i] for i in perm]
            c = _count_fan_crossings(fan_pt, trunk_dir, perp, trunk_w,
                                     candidate, centroids, is_source)
            if c < best_c:
                best, best_c = candidate, c
                if c == 0:
                    return (best, best_c)
        return (best, best_c)

    # adjacent-swap refinement for larger n 
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            swapped = best[:i] + [best[i + 1], best[i]] + best[i + 2:]
            c = _count_fan_crossings(fan_pt, trunk_dir, perp, trunk_w,
                                     swapped, centroids, is_source)
            if c < best_c:
                best, best_c = swapped, c
                improved = True
                if c == 0:
                    return (best, best_c)

    return (best, best_c)


def _curve_to_tapered_polygon(xs, ys, width_start, width_end):
    """Convert a sampled curve into a tapered polygon (wide end to narrow end).

    Parameters
    ----------
    xs, ys      : array-like of curve sample coordinates.
    width_start : polygon width (data coords) at the first sample.
    width_end   : polygon width (data coords) at the last sample.

    Returns (N, 2) array of polygon corners (left side forward, right side backward).
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    widths = np.linspace(width_start, width_end, n)

    left = np.empty((n, 2))
    right = np.empty((n, 2))

    for i in range(n):
        if i == 0:
            dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        elif i == n - 1:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
        else:
            dx, dy = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]

        length = np.hypot(dx, dy) + 1e-12
        nx, ny = -dy / length, dx / length

        w = widths[i] / 2
        left[i] = [xs[i] + nx * w, ys[i] + ny * w]
        right[i] = [xs[i] - nx * w, ys[i] - ny * w]

    return np.vstack([left, right[::-1]])


def _feasible_areas(source_points, radius=3.0):
    """Returns a list of feasible regions as circles (center + radius).
    Currently all circles share the same radius.
    """
    src_arr = np.array(source_points)
    areas = [{'center': src, 'radius': radius} for src in src_arr]
    return areas

def _find_optimal_split_point_for_pair(bundle_pt, dst_cid, centroids, clusters, radius=1.5,
                                       dst_weights=None, dst_src_weight_ratio=1.3,
                                       min_arc_length=2.0, min_arc_penalty=2.0,
                                       q2_weight=0.3, q3_weight=0.15):
    """Find the optimal split point for a destination cluster.

    Multi-objective cost function optimising:
      - Direction: split point should face the bundle point (minimise dist to bundle)
      - Q2 (bundling ratio): split point should be close to destination countries so
        that destination branches are short relative to the trunk.
      - Q3 (distance to nearest country): split point should not overlap with any
        destination country centroid, to keep individual flows distinguishable.

    Parameters
    ----------
    bundle_pt :             (x, y) coordinates of the bundle point.
    dst_cid :               destination cluster ID.
    centroids :             country name -> (x, y).
    clusters :              cluster_id -> list of country names.
    radius :                radius of feasible areas around destination countries.
    dst_weights :           weights for destination countries (optional, used as fallback).
    dst_src_weight_ratio :  weight for proximity to dest vs source (radius=0 mode).
    min_arc_length :        minimum desired distance from split to each destination country.
    min_arc_penalty :       strength of the hard-threshold penalty for short arcs.
    q2_weight :             weight for the bundling-ratio term (penalises long dest branches).
    q3_weight :             weight for the distance-to-nearest-country term (rewards separation).

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

        # --- Primary direction term ---
        if radius > 0:
            cost = np.linalg.norm(point - bundle_pt_arr)
        else:
            dist_to_dst = np.mean([np.linalg.norm(point - dp) for dp in dst_points])
            dist_to_src = np.linalg.norm(point - bundle_pt_arr)
            cost = dst_src_weight_ratio * dist_to_dst + dist_to_src

        # --- Q2: penalise long destination branches (mean dist to dest countries) ---
        # Shorter branches → larger bundling ratio
        mean_dst_dist = np.mean([np.linalg.norm(point - dp) for dp in dst_points])
        cost += q2_weight * mean_dst_dist

        # --- Q3: reward distance to nearest destination country centroid ---
        # Negative term: larger min-distance → lower cost
        min_dist = min(np.linalg.norm(point - dp) for dp in dst_points)
        cost -= q3_weight * min_dist

        # --- Hard threshold: penalise being too close to any destination country ---
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

def _find_optimal_bundle_point_for_pair(src_cid, dst_cid, centroids, clusters, radius, cluster_means,
                                        src_dst_weight_ratio=1.2, min_arc_length=2.0, min_arc_penalty=2.0,
                                        q2_weight=0.3, q3_weight=0.15):
    """Find the optimal bundling point for a source cluster targeting a destination cluster.

    Multi-objective cost function optimising:
      - Direction: bundle point should face the destination cluster (minimise dist to dst mean)
      - Q2 (bundling ratio): bundle point should be close to source countries so that
        source branches are short relative to the trunk.
      - Q3 (distance to nearest country): bundle point should not overlap with any
        source country centroid, to keep individual flows distinguishable.

    Parameters
    ----------
    src_cid, dst_cid :      cluster IDs.
    centroids :             country name -> (x, y).
    clusters :              cluster_id -> list of country names.
    radius :                radius of feasible areas around source points.
    cluster_means :         precomputed cluster_id -> (x, y).
    src_dst_weight_ratio :  weight for proximity to source vs destination (radius=0 mode).
    min_arc_length :        minimum desired distance from bundle point to each source country.
    min_arc_penalty :       strength of the hard-threshold penalty for short arcs.
    q2_weight :             weight for the bundling-ratio term (penalises long source branches).
    q3_weight :             weight for the distance-to-nearest-country term (rewards separation).

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
    dst_mean_arr = np.array(dst_mean)

    areas = _feasible_areas(src_points, radius=radius)

    def cost_function(p):
        point = np.array([p[0], p[1]])

        # --- Primary direction term ---
        if radius > 0:
            cost = np.linalg.norm(point - dst_mean_arr)
        else:
            dist_to_src = np.mean([np.linalg.norm(point - sp) for sp in src_points])
            dist_to_dst = np.linalg.norm(point - dst_mean_arr)
            cost = src_dst_weight_ratio * dist_to_src + dist_to_dst

        # --- Q2: penalise long source branches (mean dist to source countries) ---
        # Shorter branches → larger bundling ratio
        mean_src_dist = np.mean([np.linalg.norm(point - sp) for sp in src_points])
        cost += q2_weight * mean_src_dist

        # --- Q3: reward distance to nearest source country centroid ---
        # Negative term: larger min-distance → lower cost
        min_dist = min(np.linalg.norm(point - sp) for sp in src_points)
        cost -= q3_weight * min_dist

        # --- Hard threshold: penalise being too close to any source country ---
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

def matplotlib_map_bundled(gdf, data, centroid_table, clusters, bundle_radius=3.0, split_radius=1.5, show_intra=True, ax=None, estimated_exports=None, q2_weight=0.3, q3_weight=0.15, crossing_scores=None, distance_scores=None, bundling_scores=None):
    """Draw a flow map with edge bundling between clusters using tapered polygons.

    For each (src_cluster, dst_cluster):
      1. Tapered branches from each source country to the bundle point
      2. Uniform-width trunk from bundle point to split point
      3. Tapered branches from split point to each destination country
    Intra-cluster flows are drawn as tapered arcs if show_intra=True.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(40, 33))
        return_fig = True
    else:
        return_fig = False

    # Base map (z order 0-1)
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, zorder=0)

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

    bundle_points = compute_cluster_bundle_points_per_pair(data, centroids, clusters, radius=bundle_radius, q2_weight=q2_weight, q3_weight=q3_weight)
    bs = compute_bundle_split_points(data, centroids, clusters, bundle_points=bundle_points, bundle_radius=bundle_radius, split_radius=split_radius, estimated_exports=estimated_exports, q2_weight=q2_weight, q3_weight=q3_weight, distance_scores=distance_scores)

    # Custom colour palette
    cluster_colors = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                      for i, cid in enumerate(sorted(clusters))}
    cluster_edge_colors = {cid: CLUSTER_EDGE_COLORS[i % len(CLUSTER_EDGE_COLORS)]
                           for i, cid in enumerate(sorted(clusters))}

    max_total_flow = max((info['total_flow'] for info in bs.values()), default=1)

    # Force a draw so transData pixel transforms are valid
    ax.get_figure().canvas.draw()

    # Display-space helpers (for branch sorting only)
    def _disp_perp(pt_a, pt_b):
        a_d = ax.transData.transform(pt_a)
        b_d = ax.transData.transform(pt_b)
        d = b_d - a_d
        d_len = np.linalg.norm(d)
        d_u = d / (d_len + 1e-12)
        return np.array([-d_u[1], d_u[0]])

    def _perp_proj(country, perp_disp):
        pt_d = ax.transData.transform(centroids[country])
        return np.dot(pt_d, perp_disp)

    # Collect all rendered curves for global crossing count (Q1)
    all_curves = []

    # Per-flow bundling ratio computation (Q2) per report formula:
    #   r(s,j) = l(bj->pj) / (l(s->bj) + l(bj->pj) + l(pj->d*))
    #   where d* is the destination with the longest branch
    bundling_ratios = []

    # Tapered inter-cluster flows
    nr_crossings = 0   # Total number of crossings

    for (src_cid, dst_cid), info in bs.items():
        bundle_pt = np.array(info['bundle'], float)
        split_pt  = np.array(info['split'], float)
        color = cluster_colors.get(dst_cid, 'red')

        total_flow = info['total_flow']
        # Validity constraint 3: trunk width proportional to F(Ksrc, Kj)
        # Use sqrt scaling for visual clarity, but ensure proportionality
        # between different trunks: w ∝ sqrt(F(Ksrc, Kj))
        trunk_w = 0.4 + 1.1 * np.sqrt(total_flow / max_total_flow)  # data-coord width

        trunk_dir = split_pt - bundle_pt
        trunk_unit = trunk_dir / (np.linalg.norm(trunk_dir) + 1e-12)
        perp = np.array([-trunk_unit[1], trunk_unit[0]])

        # Branch widths proportional to their share of the trunk
        src_branches = [(c, w, trunk_w * w / total_flow)
                        for c, w in info['src_weights'].items() if c in centroids]
        dst_branches = [(c, w, trunk_w * w / total_flow)
                        for c, w in info['dst_weights'].items() if c in centroids]

        # Minimise crossings at the bundle and split fan-out points by
        # testing actual Bézier curves and searching for the best order
        src_branches, c1 = _minimize_fan_crossings(bundle_pt, trunk_dir, src_branches,
                                                    centroids, trunk_w, is_source=True)
        dst_branches, c2 = _minimize_fan_crossings(split_pt, trunk_dir, dst_branches,
                                                    centroids, trunk_w, is_source=False)
        
        nr_crossings += c1 + c2

        # Track per-source-country branch lengths for Q2 computation
        src_branch_lengths = {}  # country -> length of source branch (s -> bj)

        # Track per-source-country branch lengths for Q2 computation
        src_branch_lengths = {}  # country -> length of source branch (s -> bj)

        # Source to bundle: branches arrive side by side at the flat trunk head
        running = 0.0
        for country, _, dw in src_branches:
            offset = running + dw / 2 - trunk_w / 2
            running += dw
            junction = bundle_pt + perp * offset

            xs, ys = _smooth_curve_directed(
                centroids[country], tuple(junction), tangent_in=trunk_dir)
            # Store curve for global crossing count (Q1)
            all_curves.append((np.asarray(xs), np.asarray(ys)))
            # Compute arc length along the rendered curve
            src_branch_lengths[country] = _arc_length(xs, ys)

            corners = _curve_to_tapered_polygon(xs, ys, dw * 1.5, dw)
            ax.add_patch(MplPolygon(corners, closed=True, facecolor=color,
                                    edgecolor='none', alpha=0.45, zorder=2))

        # Trunk: uniform-width rectangle (validity constraint 3: width ∝ F(Ksrc, Kj))
        trunk_length = np.linalg.norm(bundle_pt - split_pt)

        # Store trunk as a polyline for crossing count
        trunk_xs = np.array([bundle_pt[0], split_pt[0]])
        trunk_ys = np.array([bundle_pt[1], split_pt[1]])
        all_curves.append((trunk_xs, trunk_ys))

        w_half = trunk_w / 2
        trunk_corners = np.array([
            bundle_pt + perp * w_half,
            bundle_pt - perp * w_half,
            split_pt  - perp * w_half,
            split_pt  + perp * w_half,
        ])
        ax.add_patch(MplPolygon(trunk_corners, closed=True, facecolor=color,
                                edgecolor='none', alpha=0.55, zorder=2))

        # Split to destination: tapered (wide at split, narrow at destination)
        dst_branch_lengths = {}  # country -> length of destination branch (pj -> d)
        running = 0.0
        for country, _, dw in dst_branches:
            offset = running + dw / 2 - trunk_w / 2
            running += dw
            junction = split_pt + perp * offset

            xs, ys = _smooth_curve_directed(
                tuple(junction), centroids[country], tangent_out=trunk_dir)
            # Store curve for global crossing count (Q1)
            all_curves.append((np.asarray(xs), np.asarray(ys)))
            # Compute arc length along the rendered curve
            dst_branch_lengths[country] = _arc_length(xs, ys)

            narrow = min(max(dw * 0.3, 0.2), dw)
            corners = _curve_to_tapered_polygon(xs, ys, dw, narrow)
            ax.add_patch(MplPolygon(corners, closed=True, facecolor=color,
                                    edgecolor='none', alpha=0.45, zorder=2))

        # Compute per-flow bundling ratio (Q2) per report formula:
        #   r(s, j) = l(bj->pj) / l_total(s, j)
        #   l_total(s, j) = l(s->bj) + l(bj->pj) + l(pj->d*)
        #   where d* is the destination country with the longest branch
        if dst_branch_lengths:
            max_dst_branch = max(dst_branch_lengths.values())
        else:
            max_dst_branch = 0
        for country in src_branch_lengths:
            l_src = src_branch_lengths[country]
            l_total = l_src + trunk_length + max_dst_branch
            if l_total > 0:
                bundling_ratios.append(trunk_length / l_total)

    # ---- Quality metrics ----
    # Q1: Total edge crossings across all rendered curves
    total_crossings = 0
    for i in range(len(all_curves)):
        for j in range(i + 1, len(all_curves)):
            if _curves_intersect(all_curves[i][0], all_curves[i][1],
                                 all_curves[j][0], all_curves[j][1]):
                total_crossings += 1

    # Q2: Sum of bundling ratios (higher = better bundling)
    bundling_score = sum(bundling_ratios) if bundling_ratios else 0

    avg_bundling = bundling_score / len(bundling_ratios) if bundling_ratios else 0

    print("EDGE CROSSINGS (Q1): ", total_crossings)
    print("BUNDLING SCORE (Q2): ", avg_bundling)

    if crossing_scores is not None:
        crossing_scores.append(total_crossings)
    if bundling_scores is not None:
        bundling_scores.append(avg_bundling)

    # Intra-cluster flows as tapered arcs 
    if show_intra:
        max_q = data.max(numeric_only=True).max()
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

                t_intra = np.sqrt(qty / max_q)
                w_intra = 0.5 + t_intra * 0.8
                color = cluster_colors.get(src_cid, 'red')

                xs, ys = _smooth_curve(centroids[src], centroids[dst], offset=0.15)
                corners = _curve_to_tapered_polygon(xs, ys, w_intra, w_intra * 0.15)
                ax.add_patch(MplPolygon(corners, closed=True, facecolor=color,
                                        edgecolor='none', alpha=0.4, zorder=2))

    # Country markers (zorder 5)
    # Circles for exporters (scaled by squareroot of export), diamonds for destinations.
    source_countries = set(data.index)
    all_countries = source_countries | set(data.columns)

    max_export = max((data.loc[c].sum() for c in source_countries
                      if c in centroids), default=1)

    for country in all_countries:
        if country not in centroids:
            continue
        cid = country_to_cluster.get(country)
        if cid is None:
            continue

        fill = cluster_colors.get(cid, 'gray')
        edge = cluster_edge_colors.get(cid, 'black')

        if country in source_countries:
            total_export = data.loc[country].sum()
            size = 12 + 24 * np.sqrt(total_export / max_export)
            marker = 'o'
        else:
            size = 14
            marker = 'D'

        ax.plot(*centroids[country], marker, color=fill,
                markeredgecolor=edge, markeredgewidth=1.2,
                markersize=size, zorder=5, alpha=0.85)

    # Country labels: ISO 2-letter codes 
    for country in all_countries:
        if country not in centroids:
            continue
        iso2 = COUNTRY_ISO2.get(country, '')
        if iso2:
            ax.annotate(iso2, xy=centroids[country], xytext=(6, 6),
                        textcoords='offset points', fontsize=8,
                        fontweight='bold', color='#333333', zorder=6)

    # Legend
    # Determine which cluster is the source cluster
    src_cluster_ids = {country_to_cluster[c] for c in source_countries
                       if c in country_to_cluster}

    cluster_handles = []
    for cid in sorted(clusters):
        members = sorted(clusters[cid])
        is_src = cid in src_cluster_ids
        marker = 'o' if is_src else 'D'
        color = cluster_colors[cid]
        edge = cluster_edge_colors[cid]

        label = ', '.join(COUNTRY_ISO2.get(c, c[:3]) for c in members[:4])
        if len(members) > 4:
            label += f' +{len(members) - 4}'
        if is_src:
            label += '  (export)'

        cluster_handles.append(
            Line2D([0], [0], marker=marker, color='none',
                   markerfacecolor=color, markeredgecolor=edge,
                   markeredgewidth=2.0, markersize=32,
                   alpha=0.85, label=label))

    leg = ax.legend(handles=cluster_handles, loc='upper left',
                    title='Clusters', fontsize=28,
                    title_fontsize=32, framealpha=0.9)
    ax.add_artist(leg)

    ax.annotate('Wide end = source country\nNarrow end = destination',
                xy=(0.98, 0.02), xycoords='axes fraction', fontsize=6,
                ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim([-15, 45])
    ax.set_ylim([30, 75])
    ax.set_xticks([])
    ax.set_yticks([])

    if return_fig:
        plt.tight_layout()
        plt.savefig("map.png", dpi=150)

    return ax
