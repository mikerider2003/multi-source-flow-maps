# Helper functions related to edge bundling

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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
    # plt.show()

    plt.savefig("map.png")
