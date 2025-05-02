import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict
from .network import TorusNetwork

class BaseRouter:
    """Base class for all routing implementations"""
    
    def __init__(self, network: TorusNetwork):
        self.network = network

    def minimal_distance(self, src: int, dst: int) -> int:
        """Calculate minimal wrap-aware Manhattan distance"""
        x1, y1 = src % 4, src // 4
        x2, y2 = dst % 4, dst // 4
        dx = min(abs(x1 - x2), 4 - abs(x1 - x2))
        dy = min(abs(y1 - y2), 4 - abs(y1 - y2))
        return dx + dy

class XYRouter(BaseRouter):
    """Dimension-ordered routing (X then Y)"""
    
    def route(self, current: int, dest: int) -> int:
        current_x = current % 4
        current_y = current // 4
        dest_x = dest % 4
        dest_y = dest // 4

        # X-direction first
        if current_x != dest_x:
            dx = (dest_x - current_x) % 4
            next_x = (current_x + (1 if dx < 2 else -1)) % 4
            return current_y * 4 + next_x
        # Y-direction second
        else:
            dy = (dest_y - current_y) % 4
            next_y = (current_y + (1 if dy < 2 else -1)) % 4
            return next_y * 4 + current_x

class YXRouter(BaseRouter):
    """Dimension-ordered routing (Y then X)"""
    def route(self, current: int, dest: int) -> int:
        current_x = current % 4
        current_y = current // 4
        dest_x = dest % 4
        dest_y = dest // 4

        # Y-direction first
        if current_y != dest_y:
            dy = (dest_y - current_y) % 4
            next_y = (current_y + (1 if dy < 2 else -1)) % 4
            return next_y * 4 + current_x
        
        # X-direction second
        dx = (dest_x - current_x) % 4
        next_x = (current_x + (1 if dx < 2 else -1)) % 4
        return current_y * 4 + next_x

class AdaptiveRouter(BaseRouter):
    """Congestion-aware adaptive routing"""
    
    def route(self, current: int, dest: int) -> int:
        neighbors = self.network.get_neighbors(current)
        if not neighbors:  # Fallback if no neighbors
            return current

        min_cost = float('inf')
        best_neighbor = current
        for neighbor in neighbors:
            node = self.network.nodes[neighbor]
            link = self.network.links[(current, neighbor)]
            distance = self.minimal_distance(neighbor, dest)
            total_cost = (0.4 * node.buffer_utilization + 
                          0.3 * link.congestion_factor + 
                          0.3 * (distance / 4))  # Normalized
            if total_cost < min_cost:
                min_cost = total_cost
                best_neighbor = neighbor
        return best_neighbor

class MinimalRouter(BaseRouter):
    """Shortest path routing with wrap-around"""
    def route(self, current: int, dest: int) -> int:
        neighbors = self.network.get_neighbors(current)
        min_dist = float('inf')
        best_neighbor = current
        
        for neighbor in neighbors:
            dist = self.minimal_distance(neighbor, dest)
            if dist < min_dist and self.network.links[(current, neighbor)].status == 1:
                min_dist = dist
                best_neighbor = neighbor
                
        return best_neighbor

class QRouter(BaseRouter):
    """Q-Learning based adaptive routing"""
    
    def __init__(self, network: TorusNetwork):
        super().__init__(network)
        self.q_table = defaultdict(lambda: 1.0)  # Optimistic initialization
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

    def route(self, current: int, dest: int) -> int:
        neighbors = self.network.get_neighbors(current)
        if not neighbors:  # Edge case handling
            return current
        q_values = [self.q_table[(current, n, dest)] for n in neighbors]
        return neighbors[np.argmax(q_values)]

    def update_q_value(self, current: int, next_node: int, dest: int, reward: float):
        neighbors = self.network.get_neighbors(next_node)
        if not neighbors:  # Avoid KeyError
            return
        max_future_q = max([self.q_table[(next_node, nn, dest)] for nn in neighbors])
        old_q = self.q_table[(current, next_node, dest)]
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(current, next_node, dest)] = new_q

class AttentionRouter(BaseRouter, nn.Module):
    """Neural attention-based routing"""
    
    def __init__(self, network: TorusNetwork):
        BaseRouter.__init__(self, network)
        nn.Module.__init__(self)
        self.node_enc = nn.Linear(4, 64)  # 4 node features
        self.edge_enc = nn.Linear(6, 64)  # 6 edge features
        self.attention = nn.MultiheadAttention(64, 4)
        self.decoder = nn.Linear(64, 4)  # 4 directions

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        node_emb = self.node_enc(node_feats)
        edge_emb = self.edge_enc(edge_feats)
        attn_out, _ = self.attention(
            node_emb.unsqueeze(1),  # Add sequence dimension
            edge_emb.unsqueeze(1), 
            node_emb.unsqueeze(1)
        )
        return self.decoder(attn_out.squeeze(1))

    def route(self, current: int, dest: int) -> int:
        self.current_dest = dest
        with torch.no_grad():
            node_feats = self._get_node_features(current)
            edge_feats = self._get_edge_features(current)
            logits = self.forward(node_feats, edge_feats)
            neighbors = self.network.get_neighbors(current)
            if not neighbors:  # Safety check
                return current
            return neighbors[torch.argmax(logits).item()]

    def _get_node_features(self, current: int) -> torch.Tensor:
        node = self.network.nodes[current]
        return torch.FloatTensor([
            node.buffer_utilization,
            node.throughput,
            node.contention_counter / 100,
            self.minimal_distance(current, self.current_dest) / 4
        ])

    def _get_edge_features(self, current: int) -> torch.Tensor:
        features = []
        for neighbor in self.network.get_neighbors(current):
            link = self.network.links[(current, neighbor)]
            features.append([
                link.current_utilization,
                link.congestion_factor,
                len(link.priority_queue) / 100,
                link.status,
                self.minimal_distance(neighbor, self.current_dest) / 4,
                len(link.contention_history) / 10
            ])
        return torch.FloatTensor(features)