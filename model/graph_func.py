import sys
sys.path.append('..')
import os
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import logging
import pickle5 as pickle
from functools import partial
from collections import Counter, defaultdict
import plotly.express as px
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv, GAE
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import SAGEConv
# from torch_geometric.nn import GATConv
from torch.nn.utils.rnn import pad_sequence
import requests
import json
from datetime import datetime
from tqdm import tqdm


from config import Config
from utils import tool_funcs
from utils.data_loader import read_traj_dataset
from utils.traj import *
from utils import tool_funcs
from config import Config


def prepare_graph(Config, cellspace):
    # Convert connectivities to edge indices
    edges = []
    nodes_map = {}  # Map to track node indices: coordinates to indices
    # index_to_nodes = {}  # Reverse map to track indices to coordinates
    node_index = 0

    features = []

    for x in range(cellspace.x_size):
        for y in range(cellspace.y_size):
            # Fetch the actual coordinates (cell_x, cell_y) from the CellSpace
            cell_x, cell_y = cellspace.get_point_by_xyidx(x, y)  # Modify according to your actual method
            features.append([x, y, cell_x, cell_y])

    features = torch.tensor(features, dtype=torch.float).to(Config.device)

    # Node mapping
    for x in range(cellspace.x_size):
        for y in range(cellspace.y_size):
            nodes_map[(x, y)] = node_index
            node_index += 1

    for x in range(cellspace.x_size):
        for y in range(cellspace.y_size):
            node_id = nodes_map[(x, y)]
            # Define neighbors: right, left, bottom, top, and four diagonals
            neighbors = [
                (x, y + 1),  # right
                (x, y - 1),  # left
                (x + 1, y),  # bottom
                (x - 1, y),  # top
                (x + 1, y + 1),  # bottom-right
                (x + 1, y - 1),  # bottom-left
                (x - 1, y + 1),  # top-right
                (x - 1, y - 1)  # top-left
            ]

            # Add edges for valid neighbors within grid bounds
            for nx, ny in neighbors:
                if 0 <= nx < cellspace.x_size and 0 <= ny < cellspace.y_size:  # Ensure the neighbor is within grid bounds
                    neighbor_id = nodes_map[(nx, ny)]
                    edges.append((node_id, neighbor_id))
                    edges.append((neighbor_id, node_id))  # Add both directions if undirected graph

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(Config.device)

    # Create a simple feature matrix with one-hot encoding of nodes
    # num_nodes = len(nodes_map)
    # print(num_nodes)
    # features = torch.eye(num_nodes).to(Config.device)
    return features, edge_index, nodes_map


    # Create the graph data structure
    # graph_data = Data(x=features, edge_index=edge_index).to(Config.device)
    # num_features = 2  # Assuming each node has 2 features for simplicity
    # hidden_dim = Config.hidden_dim // 2
    # output_dim = Config.seq_embedding_dim
    # gsage = GraphSAGE(num_features, hidden_dim, output_dim).to(Config.device)
    #
    # grid = gsage(features, edge_index)
    # # return graph_data, nodes_map
    # return grid, nodes_map

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True, aggregation='mean')
        # self.conv2 = SAGEConv(hidden_channels, out_channels, normalize=True, aggregation='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        return x


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        # Encoder with two GCN layers
        self.encoder1 = GCNConv(num_features, hidden_channels)
        self.encoder2 = GCNConv(hidden_channels, out_channels)

        # Decoder that maps back to the original feature space
        self.decoder1 = torch.nn.Linear(out_channels, hidden_channels)
        self.decoder2 = torch.nn.Linear(hidden_channels, num_features)

    def encode (self, x, edge_index):
        z = F.relu(self.encoder1(x, edge_index))  # First layer
        z = F.relu(self.encoder2(z, edge_index))  # Second layer
        return z
    def forward(self, x, edge_index):
        # Encode
        z = F.relu(self.encoder1(x, edge_index))  # First layer
        z = F.relu(self.encoder2(z, edge_index))  # Second layer

        # Decode
        z = F.relu(self.decoder1(z))  # First decoder layer
        x_hat = self.decoder2(z)  # Second decoder layer
        return x_hat


def get_8_neighbors(x, y, cellspace):
    """
    get 8 neighbors according to cell (x,y).
    """
    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cellspace.x_size and 0 <= ny < cellspace.y_size:
            neighbors.append((nx, ny))
    return neighbors


def get_adj_matrix(traj_cell, cellspace, embs, B, max_num_points, inv_paddings):
    adjacency_matrices = torch.zeros((B, max_num_points, Config.num_neighbours), dtype=torch.float)
    adjacency = torch.zeros((B, max_num_points, Config.num_neighbours, Config.cell_embedding_dim), dtype=torch.float)

    for b_idx, traj in enumerate(traj_cell):
        pad = inv_paddings[b_idx]
        for i in range(len(traj)):
            if pad[i] != 0:
                x, y = cellspace.get_xyidx_by_cellid(traj[i])
                adjacency_matrices[b_idx, i, 0] = 1  # First column for self-connection
                adjacency[b_idx, i, 0] = torch.tensor(embs[traj[i]])
                neighbors = get_8_neighbors(x, y, cellspace)
                for j, (dx, dy) in enumerate(neighbors):
                    n_x, n_y = x + dx, y + dy
                    # Check if the neighbor is within grid boundaries
                    if 0 <= n_x < cellspace.x_size and 0 <= n_y < cellspace.y_size:
                        neighbor_id = cellspace.get_cellid_by_xyidx(n_x, n_y)
                        adjacency_matrices[b_idx, i, j + 1] = neighbor_id  # j+1 because 0 is the self-connection
                        adjacency[b_idx, i, j + 1] = torch.tensor(embs[neighbor_id])

    return adjacency_matrices, adjacency


def retrieve_poi():
    node_file = Config.poi_node_file
    way_file = Config.poi_way_file

    poi_nodes = pickle.load(open(node_file, 'rb'))
    print(poi_nodes.shape)
    # print(poi_nodes.head())

    type_counts = poi_nodes.groupby('category').size()
    # print(type_counts)
    cutoff_date = datetime(2014, 7, 1)

    batch_size = 100  # Number of IDs to query at once
    total = poi_nodes.shape[0]
    batches = [poi_nodes.iloc[i:i + batch_size] for i in range(0, total, batch_size)]


    filtered_rows = []
    overpass_url = "http://overpass-api.de/api/interpreter"
    count = 0
    for batch in tqdm(batches, desc="Processing batches of POIs"):
        node_ids = ','.join(batch['id'].astype(str))
        # Define the query to fetch the history of the node
        query = f"""
                [output:json];
                node(id:{node_ids});
                out meta;
                """

        # Send the request
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()

        # Print the result
        # print(json.dumps(data, indent=2))
        if 'elements' not in data:
            print("no data found")
        else:
            for element in data['elements']:
                timestamp = element.get('timestamp', 'No timestamp available')
                tags = element.get('tags', {})
                print(f"Timestamp: {timestamp}, Tags: {tags}")
                date_string = timestamp[:-1]
                date_obj = datetime.fromisoformat(date_string)
                if date_obj < cutoff_date:
                    row = batch[batch['id'] == element['id']].iloc[0]
                    filtered_rows.append(row)
                    print("POI added before July 1, 2014")
                    count += 1


    print(f"Number of POIs filtered: {count}")

    filtered_pois = pd.DataFrame(filtered_rows)
    filtered_pois.to_pickle('./filtered_poi.pkl')
    print("Filtered POIs saved to './filtered_poi.pkl'")


def _collate(trajs, cellspace, embs):
    # traj_cell, point = zip(*[merc2cell2(t[:, :2], cellspace) for t in trajs])
    traj_cell = []
    traj_offsets = []
    ts = []
    for t in trajs:
        # print(t.shape)
        # Process the first two columns of 't' through 'merc2cell'
        cells, traj_o, time = merc2cell(t, cellspace)

        # Append the results to the respective lists
        traj_cell.append(cells)
        traj_offsets.append(traj_o)
        # print(f"cell: {cells}")
        # print(f"traj_o: {traj_o}")
        # print(f"time: {time}")
        # print(t[:, -1])
        # Directly append the last column of 't' to 'last_elements'
        ts.append(time)

    traj_emb_cell = [embs[list(t)] for t in traj_cell]
    num_points = torch.tensor(list(map(len, traj_cell)), dtype=torch.long, device=Config.device)
    # max_num_points = num_points.max().item()
    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True).to(Config.device)
    traj_offsets = [torch.tensor(np.stack(list(traj_o))) for traj_o in traj_offsets]
    # print(traj_offsets)
    traj_offsets = pad_sequence(traj_offsets, batch_first=True).to(Config.device)
    ts = [torch.tensor(list(t)) for t in ts]
    ts = pad_sequence(ts, batch_first=True).to(Config.device)
    # paddings = torch.arange(max_num_points, device=Config.device)[None, :] >= num_points[:, None]
    # inv_paddings = ~paddings

    # B = traj_emb_cell.shape[1]
    # get trajectory adjacency matrix.
    # adj_m = get_adj_matrix(traj_cell, cellspace, B, max_num_points, inv_paddings)
    return traj_emb_cell, traj_offsets.float() , ts.float() , num_points

def visit_density():
    train_dataset, _, _ = read_traj_dataset(Config.dataset_file)
    # print(len(train_dataset))
    cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))

    # traj_cell, point = zip(*[merc2cell2(t[:, :2], cellspace) for t in train_dataset])

    train_dataloader = DataLoader(train_dataset,
                                       batch_size=Config.trajcl_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=True,
                                       collate_fn=partial(_collate, cellspace=cellspace))

    cell_id_counts = Counter()
    for i, batch in enumerate(train_dataloader):
        traj_cell, point, t = batch


        for traj in traj_cell:
            # Update the Counter with the cell IDs in this trajectory
            # Note: traj is assumed to be a list or tuple of cell IDs here
            cell_id_counts.update(traj)

    # normalize to 0~1.
    max_count = max(cell_id_counts.values())
    normalized_counts = {cell_id: count / max_count for cell_id, count in cell_id_counts.items()}

    # print(normalized_counts)

    with open("density.pkl","wb") as f:
        pickle.dump(normalized_counts, f)

    return cell_id_counts, normalized_counts, cellspace



def add_transitions(trajectories, transitions):
    # iteratively add transition data in the dict.
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            current_cell = trajectory[i]
            next_cell = trajectory[i + 1]
            # print(current_cell, next_cell)
            # Increment the transition count from current_cell to next_cell
            transitions[current_cell][next_cell] += 1


def normalize_adj_matrix(adjacency_matrix):
    # Find the maximum value in the matrix
    max_value = adjacency_matrix.max()

    # Avoid division by zero in case the matrix is all zeros
    if max_value > 0:
        # Normalize the entire matrix by the maximum value
        normalized_matrix = adjacency_matrix / max_value
    else:
        normalized_matrix = adjacency_matrix  # Keep original if max is 0
    return normalized_matrix

def visit_transition():
    train_dataset, _, _ = read_traj_dataset(Config.dataset_file)
    # print(len(train_dataset))
    cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=Config.trajcl_batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=True,
                                  collate_fn=partial(_collate, cellspace=cellspace))

    adjacency_matrix = np.zeros((cellspace.size(), cellspace.size()), dtype=float)
    # Dictionary to track transitions for each cell
    transitions = defaultdict(lambda: defaultdict(float))

    for i, batch in enumerate(train_dataloader):
        traj_cell, point, t = batch

        add_transitions(traj_cell, transitions)

    # Build the adjacency matrix from the transitions
    for source_cell, destinations in transitions.items():
        for destination_cell, weight in destinations.items():
            adjacency_matrix[source_cell, destination_cell] = weight
    # print(adjacency_matrix)
    normalized_matrix = normalize_adj_matrix(adjacency_matrix)
    # print(normalized_matrix)
    with open("transition.pkl", "wb") as f:
        pickle.dump(normalized_matrix, f)

    return normalized_matrix


def vis_visit_density(normalized_counts, cs):
    density_map = {}
    # get the density map in lon and lat.
    for cell, count in normalized_counts.items():
        cell_x, cell_y = cs.get_xyidx_by_cellid(cell)
        cell_merc = cs.get_point_by_xyidx(cell_x, cell_y)
        lon, lat = tool_funcs.meters2lonlat(cell_merc[0], cell_merc[1])
        # print(lon, lat)
        # break
        density_map[(lon, lat)] = count

    # visualize in a heatmap.
    # Convert the density_map to a DataFrame
    data = {'longitude': [], 'latitude': [], 'count': []}
    for (lon, lat), count in density_map.items():
        data['longitude'].append(lon)
        data['latitude'].append(lat)
        data['count'].append(count)
    df = pd.DataFrame(data)

    # Create a scatter mapbox plot with Plotly Express
    fig = px.scatter_mapbox(df,
                            lat="latitude",
                            lon="longitude",
                            size="count",  # Use normalized count for bubble size
                            color="count",  # Use normalized count for bubble color
                            color_continuous_scale=px.colors.cyclical.IceFire,  # Color scale
                            size_max=30,
                            zoom=10,
                            mapbox_style="carto-positron")  # Map style

    # fig.show()
    fig.write_html("./vis_density.html")

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'),
                                    logging.StreamHandler()]
                        )
    Config.dataset = 'porto'
    Config.post_value_updates()

    # cell_id_counts, normalized_counts, cellspace = visit_density()
    # vis_visit_density(normalized_counts, cellspace)
    # norm_mat = visit_transition()
    # vis_transition()
    retrieve_poi()
    # cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # print(cellspace.size())
    # features, edge_index, nodes_map = prepare_graph(Config, cellspace)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GraphAutoencoder(num_features=4, hidden_channels=128, out_channels=256).to(Config.device)

    # optimizer = torch.optim.Adam([
    #     dict(params=model.encoder1.parameters(), weight_decay=5e-4),
    #     dict(params=model.encoder2.parameters(), weight_decay=0)
    # ], lr=0.01)  # Only perform weight-decay on first convolution.

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # loss_fn = torch.nn.MSELoss()
    #
    # model.train()
    # # Training loop (simplified)
    # for epoch in range(400):
    #     model.train()
    #     optimizer.zero_grad()
    #     x_hat = model(features, edge_index)
    #     loss = loss_fn(x_hat, features)  # Minimizing the reconstruction error
    #     loss.backward()
    #     optimizer.step()
    #     print(f'Epoch {epoch}: Loss {loss.item()}')
    #
    # model.eval()
    # with torch.no_grad():  # Disable gradient computation for inference
    #     embeddings = model.encode(features, edge_index)
    #     embeddings = embeddings.detach().cpu().numpy()
    #
    #     # Save embeddings to a file
    #     with open('GCN_embeddings.pkl', 'wb') as f:
    #         pickle.dump(embeddings, f)

