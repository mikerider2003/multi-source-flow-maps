"""
Experiment runner: vary Q2 (bundling) and Q3 (distance) weights and save
maps with descriptive filenames.

Each experiment generates the full k=7 flow map grid and saves it as:
    experiments/map_q2={q2_weight}_q3={q3_weight}.png

Console output includes the metric values for each run.
"""

import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for batch runs

from main import main_clustered

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "weights")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Weight configurations to test:
#   (q2_weight, q3_weight)
#   q2 controls bundling-ratio optimisation (higher -> shorter branches, longer trunk)
#   q3 controls distance-to-nearest-country (higher -> bundle/split farther from centroids)
EXPERIMENTS = [
    # Baseline: no Q2/Q3 optimisation
    (0.0, 0.0),
    # Q2-only variants
    (0.15, 0.0),
    (0.3, 0.0),
    (0.6, 0.0),
    # Q3-only variants
    (0.0, 0.08),
    (0.0, 0.15),
    (0.0, 0.30),
    # Combined (default)
    (0.3, 0.15),
    # Higher combined
    (0.6, 0.30),
    # Imbalanced: heavy bundling, light distance
    (0.6, 0.08),
    # Imbalanced: light bundling, heavy distance
    (0.15, 0.30),
]

for q2, q3 in EXPERIMENTS:
    filename = os.path.join(OUTPUT_DIR, f"map_q2={q2:.2f}_q3={q3:.2f}.png")
    print("=" * 70)
    print(f"EXPERIMENT: q2_weight={q2}, q3_weight={q3}")
    print(f"  Output: {filename}")
    print("=" * 70)

    main_clustered(
        n_clusters=7,
        show_intra=False,
        multiple_bundle_points=True,
        bundle_radius=0,
        split_radius=0,
        q2_weight=q2,
        q3_weight=q3,
        output_file=filename,
    )

    # Close all figures to free memory between runs
    import matplotlib.pyplot as plt
    plt.close('all')

    print()

print(f"All experiments complete. Results saved in {OUTPUT_DIR}")