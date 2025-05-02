from enum import Enum

class TrafficPattern(Enum):
    UNIFORM = 1
    HOTSPOT = 2
    TRANSPOSE = 3
    BIT_COMPLEMENT = 4

class PriorityScheme(Enum):
    LEVEL = 1
    ID = 2
    AGE = 3
    HYBRID = 4

NETWORK_CONFIG = {
    'num_nodes': 16,
    'buffer_capacity': 64,
    'processing_rate': 8,
    'link_bandwidth': 1e9,
    'base_latency': 1
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'max_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5
}