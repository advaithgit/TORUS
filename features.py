# features.py
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from .network import Packet, TorusNetwork
from .simulator import EnhancedNetworkSimulator
from .config import NETWORK_CONFIG  # Import network configuration

class FeatureEngineer:
    def __init__(self, simulator: EnhancedNetworkSimulator):
        self.simulator = simulator
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.pca = PCA(n_components=6)
        self.path_padder = 20  # Max path length for padding
        self.edge_feat_names = [
            'utilization', 'congestion', 'queue_len',
            'status', 'contentions', 'x_dir'
        ]
        self.is_fitted = False  # Track if scalers are trained
        self.buffer_capacity = NETWORK_CONFIG['buffer_capacity']
        self.processing_rate = NETWORK_CONFIG['processing_rate']

    def create_ml_dataset(self) -> pd.DataFrame:
        """Convert raw simulation data to ML-ready dataset"""
        if not self.simulator.packet_log:
            raise ValueError("No packet log data available for feature engineering")

        # Initial pass to fit scalers
        if not self.is_fitted:
            self._fit_scalers()

        ml_data = []
        for packet_log in self.simulator.packet_log:
            state = self._find_nearest_state(packet_log['timestamp'])
            node_feats = self._process_node_features(state['nodes'])
            edge_feats = self._process_edge_features(state['edges'])
            path_data = self._process_path(packet_log['route'])

            ml_data.append({
                'load_level': packet_log['load_level'],
                'node_features': node_feats,
                'edge_features': edge_feats,
                'path_directions': path_data['directions'],
                'path_mask': path_data['mask'],
                'contention': packet_log['contention_count'] > 0,
                'transmission_time': packet_log['transmission_time'],
                'priority_scheme': packet_log['priority_scheme'],
                'routing_algorithm': packet_log['routing_algorithm'],
                'success': packet_log['success']
            })
        return pd.DataFrame(ml_data)

    def _fit_scalers(self):
        """Fit all scalers using first batch of data"""
        sample_states = [s for s in self.simulator.network_state_history[:100]]

        # Fit node scaler
        node_features = []
        for state in sample_states:
            node_features.extend(self._raw_node_features(state['nodes']))
        self.node_scaler.fit(node_features)

        # Fit edge scaler and PCA
        edge_features = []
        for state in sample_states:
            edge_features.extend(self._raw_edge_features(state['edges']))
        self.edge_scaler.fit(edge_features)
        self.pca.fit(edge_features)
        self.is_fitted = True

    def _raw_node_features(self, node_states: List[Dict]) -> List[List[float]]:
        """Extract raw node features without scaling"""
        return [
            [
                node['buffer_occ'] / self.buffer_capacity,
                node['throughput'] / self.processing_rate,
                np.log1p(node['contention']),
                node['drops'] / 100.0
            ]
            for node in sorted(node_states, key=lambda x: x['id'])
        ]

    def _process_node_features(self, node_states: List[Dict]) -> np.ndarray:
        """Process and normalize node features"""
        raw_features = self._raw_node_features(node_states)
        return self.node_scaler.transform(raw_features)

    def _raw_edge_features(self, edge_states: List[Dict]) -> List[List[float]]:
        """Extract raw edge features without scaling"""
        return [
            [
                edge['util'],
                edge['congestion'],
                np.sqrt(edge['queue_len']),
                edge['status'],
                np.log1p(edge['contentions']),
                (edge['src'] % 4) - (edge['dst'] % 4),
                edge['dst'] // 4 - edge['src'] // 4
            ]
            for edge in edge_states
        ]

    def _process_edge_features(self, edge_states: List[Dict]) -> np.ndarray:
        """Process and compress edge features"""
        raw_features = self._raw_edge_features(edge_states)
        scaled = self.edge_scaler.transform(raw_features)
        return self.pca.transform(scaled)

    def _process_path(self, route: List[int]) -> Dict:
        """Convert node path to directional sequence with validation"""
        if not route:
            raise ValueError("Empty route provided for path processing")

        directions = []
        for i in range(1, len(route)):
            curr = route[i-1]
            next_node = route[i]
            dx = (next_node % 4 - curr % 4) % 4
            dy = (next_node // 4 - curr // 4) % 4

            if dx == 1:
                directions.append(0)  # E
            elif dx == 3:
                directions.append(1)  # W
            elif dy == 1:
                directions.append(2)  # S
            elif dy == 3:
                directions.append(3)  # N
            else:
                raise ValueError(f"Invalid movement from {curr} to {next_node}")

        # Padding and masking
        padded = np.zeros(self.path_padder, dtype=np.int64)
        mask = np.zeros(self.path_padder, dtype=np.bool_)
        length = min(len(directions), self.path_padder)
        padded[:length] = directions[:length]
        mask[:length] = True
        return {'directions': padded, 'mask': mask}

    def _find_nearest_state(self, timestamp: float) -> Dict:
        """Find closest network state snapshot with validation"""
        if not self.simulator.network_state_history:
            raise ValueError("No network state history available")

        times = [s['timestamp'] for s in self.simulator.network_state_history]
        idx = np.argmin(np.abs(np.array(times) - timestamp))
        return self.simulator.network_state_history[idx]

class DatasetSplitter:
    def __init__(self, dataset: pd.DataFrame):
        if dataset.empty:
            raise ValueError("Cannot split empty dataset")
        self.dataset = dataset

    def load_stratified_split(self, ratios: Dict = {'train':0.7, 'val':0.2, 'test':0.1}):
        """Improved stratified split preserving load level distribution"""
        # Initial split
        train, test = train_test_split(
            self.dataset,
            test_size=ratios['test'],
            stratify=self.dataset['load_level'],
            random_state=42
        )

        # Secondary split
        train, val = train_test_split(
            train,
            test_size=ratios['val']/(1 - ratios['test']),
            stratify=train['load_level'],
            random_state=42
        )

        return {
            'train': train.reset_index(drop=True),
            'val': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)
        }

class EnhancedPacketLogger:
    def __init__(self, network: TorusNetwork):  # Added network reference
        self.minimal_path_cache = {}
        self.reroute_threshold = NETWORK_CONFIG['buffer_capacity'] * 0.8
        self.network = network  # Store network reference

    def calculate_rerouting(self, packet: Packet) -> Dict:
        """Calculate re-routing metrics with buffer awareness"""
        actual_length = len(packet.route) - 1
        minimal_length = self.minimal_distance(packet.src, packet.dst)
        deviation = actual_length - minimal_length

        # Access node buffer utilization via network
        buffer_penalty = 1.2 if any(
            self.network.nodes[node_id].buffer_utilization > self.reroute_threshold
            for node_id in packet.route
        ) else 1.0

        return {
            'reroute_percent': (deviation / minimal_length) * 100 * buffer_penalty,
            'actual_hops': actual_length,
            'minimal_hops': minimal_length,
            'deviation': deviation
        }

    def minimal_distance(self, src: int, dst: int) -> int:
        """Cache-aware minimal path calculation"""
        if (src, dst) not in self.minimal_path_cache:
            x1, y1 = src % 4, src // 4
            x2, y2 = dst % 4, dst // 4
            dx = min(abs(x1 - x2), 4 - abs(x1 - x2))
            dy = min(abs(y1 - y2), 4 - abs(y1 - y2))
            self.minimal_path_cache[(src, dst)] = dx + dy
        return self.minimal_path_cache[(src, dst)]

def create_dgl_graph(node_feats: np.ndarray, edge_feats: np.ndarray,
                     edges: List[Tuple[int, int]]) -> dgl.DGLGraph:
    """Convert processed features to DGL graph with torus topology"""
    g = dgl.DGLGraph()
    g.add_nodes(len(node_feats))

    # Add bidirectional edges with corrected features
    src_nodes = [e[0] for e in edges]
    dst_nodes = [e[1] for e in edges]
    
    # Add forward and reverse edges
    g.add_edges(src_nodes, dst_nodes)
    g.add_edges(dst_nodes, src_nodes)

    # Assign node features
    g.ndata['features'] = torch.FloatTensor(node_feats)

    # Generate reverse edge features by flipping x_dir and y_dir
    reverse_edge_feats = []
    for feat in edge_feats:
        reversed_feat = feat.copy()
        reversed_feat[5] = -feat[5]  # Reverse x_dir (src-dst becomes dst-src)
        reversed_feat[6] = -feat[6]  # Reverse y_dir
        reverse_edge_feats.append(reversed_feat)
    
    # Combine forward and reverse features
    all_edge_feats = np.vstack([edge_feats, reverse_edge_feats])
    g.edata['features'] = torch.FloatTensor(all_edge_feats)

    return g