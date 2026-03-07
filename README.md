# Visualizing Multi-Source Flow Maps

This repository is for Group 5's code for the Algorithms for Geovisualization course. We aim to develop a suitable algorithm for visualizing a flow map with multiple sources and sinks, while avoiding clutter as much as possible.

### How to run the code
Currently, the algorithm can simply be run by running `main.py`.

### Data
Our dataset, `EU_trade_data.xlsx`, is taken from [https://ec.europa.eu/eurostat/databrowser/view/ds-045409__custom_20227384/default/table](Eurostat) and shows the export between EU countries in 2024. For testing purposes, we selected several subsets of this data, indicated by their postfix (right before the `.csv`):
- `_full`: the full dataset of all 27 countries in the EU.
- `_distant`: 4 countries that are spaced far apart.
- `_close`: 4 countries that are close together. Together with the previous dataset, this is mostly for testing if our algorithm produces good visualizations on distant scales.
- `_2_clusters`: 2 distant clusters of 3 nearby countries each. Meant for initial testing of edge-bundling techniques.
- `_3_clusters`: 3 evenly spaced clusters of 2 countries each.
- `_5_clusters`: 5 somewhat evenly spaced clusters of 2 countries each. Meant for stress-testing our edge bundling before applying it to the full dataset.