import dgl
import torch
import numpy as np

def create_dgl_graph(node_feats: np.ndarray, edge_feats: np.ndarray) -> dgl.DGLGraph:
    """Create DGL graph with proper 4x4 torus topology (bidirectional edges)"""
    g = dgl.DGLGraph()
    g.add_nodes(len(node_feats))
    
    # Generate all torus connections
    edges = []
    for node in range(16):
        x, y = node % 4, node // 4
        
        # East connection (x+1)
        east_neighbor = y * 4 + ((x + 1) % 4)
        edges.append((node, east_neighbor))
        
        # West connection (x-1)
        west_neighbor = y * 4 + ((x - 1) % 4)
        edges.append((node, west_neighbor))
        
        # South connection (y+1)
        south_neighbor = ((y + 1) % 4) * 4 + x
        edges.append((node, south_neighbor))
        
        # North connection (y-1)
        north_neighbor = ((y - 1) % 4) * 4 + x
        edges.append((node, north_neighbor))

    # Add bidirectional edges
    src, dst = zip(*edges)
    g.add_edges(src, dst)
    g.add_edges(dst, src)  # Reverse directions

    # Assign node features
    g.ndata['features'] = torch.FloatTensor(node_feats)

    # Process edge features for bidirectional links
    forward_feats = torch.FloatTensor(edge_feats)
    reverse_feats = forward_feats.clone()
    reverse_feats[:, 5] *= -1  # Flip x_dir for reverse edges
    reverse_feats[:, 6] *= -1  # Flip y_dir for reverse edges
    g.edata['features'] = torch.cat([forward_feats, reverse_feats])

    return g

def verify_dataset(data: pd.DataFrame):
    """Validate dataset contains required columns"""
    required = [
        'node_features', 'edge_features', 'path_directions',
        'contention', 'transmission_time', 'load_level'
    ]
    if not all(col in data.columns for col in required):
        missing = [col for col in required if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")

def save_checkpoint(model, path: str, epoch: int, metrics: Dict):
    """Save model state with metadata"""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'metrics': metrics
    }, path)

def load_checkpoint(path: str, model):
    """Load model state from checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    return model, checkpoint['epoch'], checkpoint['metrics']