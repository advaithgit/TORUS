import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .model import EnhancedTorusNet, TorusNetLoss
from .utils import create_dgl_graph
from typing import Dict

class TorusDataset(Dataset):
    def __init__(self, data):
        self.data = data.dropna()  # Remove invalid samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        return {
            'node_feats': torch.FloatTensor(sample['node_features']),
            'edge_feats': torch.FloatTensor(sample['edge_features']),
            'path_target': torch.LongTensor(sample['path_directions']),
            'contention_target': torch.FloatTensor([sample['contention']]),
            'time_target': torch.FloatTensor([sample['transmission_time']]),
            'load_level': torch.FloatTensor([sample['load_level']])
        }

class TorusTrainer:
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_loader = DataLoader(
            TorusDataset(train_data), 
            batch_size=32, 
            shuffle=True,
            collate_fn=self._collate  # Added batching support
        )
        self.val_loader = DataLoader(
            TorusDataset(val_data), 
            batch_size=64,
            collate_fn=self._collate
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.loss_fn = TorusNetLoss()
        self.curriculum = CurriculumScheduler()

    def _collate(self, batch):
        """Batch graphs using DGL's batching utility"""
        batched_graph = dgl.batch([
            create_dgl_graph(
                item['node_feats'].numpy(), 
                item['edge_feats'].numpy(),
                []  # Edges are auto-generated in create_dgl_graph
            ) for item in batch
        ])
        return {
            'graphs': batched_graph,
            'path_target': torch.cat([item['path_target'] for item in batch]),
            'contention_target': torch.cat([item['contention_target'] for item in batch]),
            'time_target': torch.cat([item['time_target'] for item in batch]),
            'load_level': torch.cat([item['load_level'] for item in batch])
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            graphs = batch['graphs']
            load_levels = batch['load_level']
            
            # Forward pass
            path_preds, contention_preds, time_preds = self.model(graphs, load_levels)
            
            # Loss calculation
            loss_dict = self.loss_fn(
                {
                    'paths': path_preds,
                    'contention': contention_preds,
                    'time': time_preds
                },
                {
                    'paths': batch['path_target'],
                    'contention': batch['contention_target'],
                    'time': batch['time_target']
                },
                self.model.get_loss_weights()
            )
            
            # Backward pass
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss_dict['total'].item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                graphs = batch['graphs']
                load_levels = batch['load_level']
                
                # Forward pass
                path_preds, contention_preds, time_preds = self.model(graphs, load_levels)
                
                # Loss calculation
                loss_dict = self.loss_fn(
                    {
                        'paths': path_preds,
                        'contention': contention_preds,
                        'time': time_preds
                    },
                    {
                        'paths': batch['path_target'],
                        'contention': batch['contention_target'],
                        'time': batch['time_target']
                    },
                    self.model.get_loss_weights()
                )
                val_loss += loss_dict['total'].item()
        return val_loss / len(self.val_loader)

    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Update curriculum based on validation loss
            self.curriculum.step(self.model, val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), f'best_model_epoch{epoch}.pt')
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')

class CurriculumScheduler:
    def __init__(self):
        self.stages = [0.1, 0.25, 0.5, 0.75, 1.0]
        self.current_stage = 0
        self.best_val_loss = float('inf')

    def step(self, model, val_loss):
        # Progress curriculum only if validation improves
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                weights = {
                    'path': max(0.3, 1.0 - self.stages[self.current_stage]),
                    'contention': min(0.7, 0.1 + self.stages[self.current_stage]),
                    'time': 0.2
                }
                model.set_loss_weights(**weights)