import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class EnhancedTorusNet(nn.Module):
    """Main GNN architecture for torus network routing"""
    def __init__(self, node_feat_dim=4, edge_feat_dim=6, hidden_dim=128):
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # GNN layers with directional attention
        self.gnn_layers = nn.ModuleList([
            TorusGNNLayer(hidden_dim) for _ in range(3)
        ])
        
        # Output heads
        self.path_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 4)  # N,S,E,W directions
        )
        
        self.contention_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus()
        )

        # Load-conditioned normalization
        self.load_norm = LoadAdaptiveNorm(hidden_dim)

    def forward(self, g, load_level):
        # Encode features
        g.ndata['h'] = self.node_encoder(g.ndata['features'])
        g.edata['e'] = self.edge_encoder(g.edata['features'])
        
        # Apply load-conditioned normalization
        g.ndata['h'] = self.load_norm(g.ndata['h'], load_level)
        
        # GNN message passing
        for layer in self.gnn_layers:
            g = layer(g)
        
        # Path prediction (per node)
        path_logits = self.path_head(g.ndata['h'])
        
        # Contention prediction (per edge)
        g.apply_edges(lambda edges: {'contention': self.contention_head(
            torch.cat([edges.src['h'], edges.dst['h']], dim=1))})
        
        # Time prediction (graph-level)
        global_feat = dgl.mean_nodes(g, 'h')
        time_pred = self.time_head(global_feat)
        
        return path_logits, g.edata['contention'], time_pred

class TorusGNNLayer(nn.Module):
    """Direction-aware GNN layer with attention"""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.direction_mlps = nn.ModuleDict({
            dir: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) for dir in ['N', 'S', 'E', 'W']
        })
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        
        self.update_fn = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, g):
        with g.local_scope():
            # Directional message passing
            for direction in ['N', 'S', 'E', 'W']:
                g.update_all(
                    message_func=fn.copy_u('h', 'm'),
                    reduce_func=fn.mean('m', f'{direction}_feat'),
                    etype=direction)
                
                g.ndata[direction] = self.direction_mlps[direction](
                    g.ndata[f'{direction}_feat'])
            
            # Cross-node attention
            node_feats = g.ndata['h'].unsqueeze(1)
            attn_out, _ = self.attention(node_feats, node_feats, node_feats)
            attn_out = attn_out.squeeze(1)
            
            # Residual connection
            combined = torch.cat([g.ndata['h'], attn_out], dim=1)
            g.ndata['h'] = self.update_fn(combined)
            
            return g

class LoadAdaptiveNorm(nn.Module):
    """Load-conditioned feature normalization"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gamma_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )
        self.beta_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh()
        )

    def forward(self, x, load_level):
        x = self.norm(x)
        gamma = 1 + self.gamma_net(load_level.view(-1, 1))
        beta = self.beta_net(load_level.view(-1, 1))
        return gamma * x + beta

class TemporalTorusNet(EnhancedTorusNet):
    """Extended model with temporal convolution"""
    def __init__(self, node_feat_dim=4, edge_feat_dim=6, hidden_dim=128, history_len=3):
        super().__init__(node_feat_dim, edge_feat_dim, hidden_dim)
        
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=history_len,
            padding=history_len//2)
        
        self.temporal_norm = nn.InstanceNorm1d(hidden_dim)

    def forward(self, g, load_level, history_states):
        # Temporal processing
        hist_feats = torch.stack([state.ndata['h'] for state in history_states], dim=-1)
        temporal_out = self.temporal_conv(hist_feats)
        temporal_out = self.temporal_norm(temporal_out)
        
        # Update node features
        g.ndata['h'] = g.ndata['h'] + temporal_out.mean(dim=-1)
        
        # Continue with base forward pass
        return super().forward(g, load_level)

class TorusNetLoss(nn.Module):
    """Multi-task loss function with dynamic weights"""
    def __init__(self):
        super().__init__()
        self.path_loss = nn.CrossEntropyLoss()
        self.contention_loss = nn.BCEWithLogitsLoss()
        self.time_loss = nn.HuberLoss()

    def forward(self, preds, targets, weights):
        path_loss = self.path_loss(preds['paths'], targets['paths'])
        contention_loss = self.contention_loss(
            preds['contention'], targets['contention'])
        time_loss = self.time_loss(preds['time'], targets['time'])
        
        total_loss = (weights['path'] * path_loss +
                      weights['contention'] * contention_loss +
                      weights['time'] * time_loss)
        
        return {
            'total': total_loss,
            'path': path_loss,
            'contention': contention_loss,
            'time': time_loss
        }