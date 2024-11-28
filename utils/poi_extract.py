import sys
sys.path.append('..')

import osmium as o
# import folium
import pandas as pd

class OSMHandler(o.SimpleHandler):
    def __init__(self):
        super(OSMHandler, self).__init__()
        self.nodes = []
        self.ways = []

    def node(self, n):
        # Check for presence of high-level POI tags without considering specific values
        for tag in ['amenity', 'shop', 'tourism', 'leisure', 'historic', 'office']:
            if tag in n.tags:
                # Store only the category of the tag
                self.nodes.append({
                    'id': n.id,
                    'lat': n.location.lat,
                    'lon': n.location.lon,
                    'category': tag  # Store the tag name as the category
                })
                break  # Exit after the first relevant tag is found

    def way(self, w):
        # Process ways similarly, but here we're just storing references
        nodes = [{'id': node.ref, 'lat': 0, 'lon': 0} for node in w.nodes]
        self.ways.append({'id': w.id, 'nodes': nodes})


# Processing the OSM file
handler = OSMHandler()
handler.apply_file('../data/porto_data.osm')

# Convert lists to DataFrames
nodes_df = pd.DataFrame(handler.nodes)
ways_df = pd.DataFrame(handler.ways)

# Save nodes and ways as pickle files
nodes_df.to_pickle('../data/porto_20200_new_nodes.pkl')
ways_df.to_pickle('../data/porto_20200_new_ways.pkl')
print("Nodes and ways have been saved as pickle files.")

# Initialize a Folium map at the first node location
# if not nodes_df.empty:
#     m = folium.Map(location=[nodes_df.iloc[0]['lat'], nodes_df.iloc[0]['lon']], zoom_start=15)
#
#     # Define a color map for different POI types
#     poi_types = nodes_df['type'].unique()
#     colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
#               'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
#               'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
#               'gray', 'black', 'lightgray']
#     color_map = {poi: colors[i % len(colors)] for i, poi in enumerate(poi_types)}
#
#     # Add nodes to the map with colors based on POI type
#     for index, row in nodes_df.iterrows():
#         folium.CircleMarker(
#             location=(row['lat'], row['lon']),
#             radius=5,
#             color=color_map[row['type']],
#             fill=True,
#             fill_color=color_map[row['type']],
#             fill_opacity=0.7,
#             popup=row['type']
#         ).add_to(m)
#
#     # Save the map as an HTML file
#     m.save('./maps/porto_poi.html')
#     print("Map has been saved as map.html")
# else:
#     print("No data available to plot on the map.")
