import numpy as np
from collections import deque
import random
import heapq
from typing import List, Dict

class Packet:
    """Packet class with required attributes (previously a placeholder)"""
    def __init__(self, src: int, dst: int, size: int, injection_time: float, priority: float):
        self.src = src
        self.dst = dst
        self.size = size
        self.injection_time = injection_time
        self.priority = priority
        self.current_node = src
        self.route = [src]
        self.contention_events = 0

class TorusNode:
    """Node in a 4x4 torus network"""
    def __init__(self, node_id: int):
        self.id = node_id
        self.x = node_id % 4
        self.y = node_id // 4
        self.buffer_capacity = 64
        self.processing_rate = 8
        self.buffer_occupancy = 0
        self.contention_counter = 0
        self.throughput = 0.0
        self.packet_drop_count = 0
        self.processing_queue = deque()

    @property
    def buffer_utilization(self) -> float:
        return self.buffer_occupancy / self.buffer_capacity

    def update_throughput(self, processed_count: int):
        """Exponential moving average of throughput"""
        self.throughput = 0.8 * self.throughput + 0.2 * processed_count

class TorusLink:
    """Bidirectional link between two nodes in the torus"""
    def __init__(self, src: int, dst: int, direction: str):
        self.bandwidth = 1e9
        self.base_latency = 1
        self.current_utilization = 0.0
        self.congestion_factor = 1.0
        self.packet_queue = deque()
        self.contention_history = []
        self.status = 1  # 1=active, 0=failed
        self.transmission_end = 0.0
        self.priority_queue = []
        self.direction = direction

    def transmit_packet(self, packet: Packet, current_time: float) -> float:
        """Handle packet transmission with priority queuing (FIXED super() call)"""
        if self.priority_queue and current_time >= self.transmission_end:
            # Process next packet from priority queue
            _, _, next_packet = heapq.heappop(self.priority_queue)
            transmission_time = (next_packet.size * 8) / self.bandwidth
            self.transmission_end = current_time + transmission_time
            return self.transmission_end

        transmission_time = (packet.size * 8) / self.bandwidth
        if current_time >= self.transmission_end:
            self.transmission_end = current_time + transmission_time
            self.current_utilization = min(1.0, self.current_utilization + transmission_time)
            return self.transmission_end
        else:
            heapq.heappush(self.priority_queue, (packet.priority, current_time, packet))
            return -1  # Indicate contention

class TorusNetwork:
    """4x4 torus network with nodes and links"""
    def __init__(self):
        self.nodes = [TorusNode(i) for i in range(16)]
        self.links = {}
        self._initialize_topology()

    def _initialize_topology(self):
        """Create bidirectional torus connections with direction labels"""
        directions = {'N': (0, -1), 'S': (0, 1), 'E': (1, 1), 'W': (1, -1)}
        for node in self.nodes:
            x, y = node.x, node.y
            for dir_name, (axis, delta) in directions.items():
                if axis == 0:  # Vertical (N/S)
                    new_y = (y + delta) % 4
                    neighbor_id = new_y * 4 + x
                else:  # Horizontal (E/W)
                    new_x = (x + delta) % 4
                    neighbor_id = y * 4 + new_x
                
                # Create forward and reverse links
                self.links[(node.id, neighbor_id)] = TorusLink(node.id, neighbor_id, dir_name)
                self.links[(neighbor_id, node.id)] = TorusLink(neighbor_id, node.id, self._reverse_dir(dir_name))

    def _reverse_dir(self, direction: str) -> str:
        return {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}.get(direction, '')

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get directly connected node IDs"""
        return [dst for (src, dst) in self.links if src == node_id and self.links[(src, dst)].status == 1]

class FaultInjector:
    """Simulate link failures and recoveries"""
    def __init__(self, network: TorusNetwork):
        self.network = network
        self.fault_probability = 0.01
        self.recovery_probability = 0.1

    def inject_faults(self):
        """Randomly disable links"""
        for link in self.network.links.values():
            if random.random() < self.fault_probability:
                link.status = 0

    def recover_faults(self):
        """Randomly recover failed links"""
        for link in self.network.links.values():
            if link.status == 0 and random.random() < self.recovery_probability:
                link.status = 1