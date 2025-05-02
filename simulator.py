import heapq
import logging
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple
from collections import deque
from .network import TorusNetwork, Packet, FaultInjector
from .routing import XYRouter, AdaptiveRouter, QRouter, AttentionRouter, MinimalRouter

logging.basicConfig(level=logging.INFO)

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

class EnhancedNetworkSimulator:
    def __init__(self):
        self.network = TorusNetwork()
        self.fault_injector = FaultInjector(self.network)
        self.event_queue = []
        self.packet_log = []
        self.current_time = 0.0
        self.network_state_history = []
        self.active_packets = []
        self.load_level = 0.0
        self.routing_algorithm = "adaptive"
        self.priority_scheme = PriorityScheme.HYBRID
        self.packet_logger = EnhancedPacketLogger(self.network)  # Pass network here

    def run_simulation(self, load_level: float, duration: int,
                       pattern: TrafficPattern, priority_scheme: PriorityScheme,
                       routing_algorithm: str = "adaptive"):
        self.load_level = load_level
        self.routing_algorithm = routing_algorithm
        self.priority_scheme = priority_scheme
        self._schedule_state_capture(duration)
        self._generate_initial_traffic(load_level, duration, pattern, priority_scheme)

        while self.event_queue and self.current_time < duration:
            time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = time
            self.fault_injector.inject_faults()
            self.fault_injector.recover_faults()

            if event_type == 'inject':
                self._handle_packet_injection(data)
            elif event_type == 'process':
                self._handle_packet_processing(data)
            elif event_type == 'snapshot':
                self._capture_network_state()
            elif event_type == 'arrival':
                packet, next_node = data
                self._handle_packet_arrival(packet, next_node)
            elif event_type == 'retry':
                self._handle_retry_transmission(data)

    def _schedule_state_capture(self, duration: int):
        capture_interval = 5  # Time units between snapshots
        next_capture = 0
        while next_capture < duration:
            heapq.heappush(self.event_queue, (next_capture, 'snapshot', None))
            next_capture += capture_interval

    def _generate_initial_traffic(self, load_level, duration, pattern, priority_scheme):
        traffic_gen = TrafficGenerator(self.network)  # Pass network to generator
        traffic_gen.generate_traffic(load_level, duration, pattern, priority_scheme)
        self.event_queue = traffic_gen.event_queue

    def _handle_packet_injection(self, packet: Packet):
        node = self.network.nodes[packet.current_node]
        if node.buffer_occupancy < node.buffer_capacity:
            node.processing_queue.append(packet)
            node.buffer_occupancy += 1
            process_time = self.current_time + 1 / node.processing_rate
            heapq.heappush(self.event_queue, (process_time, 'process', packet))
        else:
            node.packet_drop_count += 1

    def _get_router(self):
        """Return router instance based on current algorithm"""
        return {
            'xy': XYRouter(self.network),
            'adaptive': AdaptiveRouter(self.network),
            'q': QRouter(self.network),
            'attention': AttentionRouter(self.network)
        }[self.routing_algorithm]

    def _handle_packet_processing(self, packet: Packet):
        current_node = packet.current_node
        router = self._get_router()
        next_node = router.route(current_node, packet.dst)  # Use router directly

        # Remove redundant routing logic (old _xy_routing/_adaptive_routing calls)
        packet.route.append(next_node)
        packet.current_node = next_node

        if next_node == packet.dst:
            self._log_packet(packet, success=True)
        else:
            self._schedule_transmission(packet, next_node)

    def _schedule_transmission(self, packet: Packet, next_node: int):
        link = self.network.links.get((packet.current_node, next_node), None)
        if not link or link.status == 0:
            self._handle_failed_link(packet)
            return

        transmission_time = (packet.size * 8) / link.bandwidth
        arrival_time = self.current_time + transmission_time + link.base_latency

        if self.current_time >= link.transmission_end:
            link.transmission_end = self.current_time + transmission_time
            link.current_utilization = min(1.0, link.current_utilization + transmission_time)
            heapq.heappush(self.event_queue, (arrival_time, 'arrival', (packet, next_node)))
        else:
            heapq.heappush(link.priority_queue, (packet.priority, self.current_time, packet))
            link.contention_history.append(self.current_time)
            packet.contention_events += 1
            retry_time = self.current_time + np.random.uniform(0.1, 0.5)
            heapq.heappush(self.event_queue, (retry_time, 'retry', packet))

    def _handle_failed_link(self, packet: Packet):
        packet.contention_events += 1
        retry_time = self.current_time + np.random.uniform(0.5, 1.0)
        heapq.heappush(self.event_queue, (retry_time, 'retry', packet))

    def _handle_packet_arrival(self, packet: Packet, next_node: int):
        if next_node == packet.dst:
            self._log_packet(packet, success=True)
        else:
            packet.current_node = next_node
            self._handle_packet_injection(packet)

    def _handle_retry_transmission(self, packet: Packet):
        current_node = packet.current_node
        neighbors = self.network.get_neighbors(current_node)
        if not neighbors:
            self._log_packet(packet, success=False)
            return
        next_node = random.choice(neighbors)
        self._schedule_transmission(packet, next_node)

    def _log_packet(self, packet: Packet, success: bool):
        entry = {
            'source_id': packet.src,
            'dest_id': packet.dst,
            'timestamp': packet.injection_time,
            'packet_size': packet.size,
            'route': packet.route.copy(),
            'transmission_time': self.current_time - packet.injection_time,
            'contention_count': packet.contention_events,
            'load_level': self.load_level,
            'success': success,
            'priority_scheme': self.priority_scheme.name,
            'routing_algorithm': self.routing_algorithm
        }
        self.packet_log.append(entry)
        self.packet_logger.calculate_rerouting(packet)  # Use updated packet logger

    def _capture_network_state(self):
        node_states = []
        for node in self.network.nodes:
            node_states.append({
                'id': node.id,
                'buffer_occ': node.buffer_occupancy,
                'buffer_util': node.buffer_utilization,
                'throughput': node.throughput,
                'drops': node.packet_drop_count,
                'contention': node.contention_counter
            })

        edge_states = []
        for (src, dst), link in self.network.links.items():
            edge_states.append({
                'src': src,
                'dst': dst,
                'util': link.current_utilization,
                'congestion': link.congestion_factor,
                'queue_len': len(link.priority_queue),
                'status': link.status,
                'contentions': len(link.contention_history)
            })

        self.network_state_history.append({
            'timestamp': self.current_time,
            'nodes': node_states,
            'edges': edge_states
        })

class TrafficGenerator:
    def __init__(self, network: TorusNetwork):
        self.network = network
        self.event_queue = []
        self.hotspots = [5, 6, 9, 10]

    def generate_traffic(self, load_level: float, duration: int,
                        pattern: TrafficPattern, priority_scheme: PriorityScheme):
        max_capacity = 16 * 8  # nodes * processing_rate
        target_rate = load_level * max_capacity
        avg_interval = 1.0 / target_rate if target_rate > 0 else float('inf')

        current_time = 0.0
        while current_time < duration:
            src, dst = self._select_endpoints(pattern)
            size = self._generate_packet_size()
            priority = self._calculate_priority(src, priority_scheme)
            packet = Packet(src, dst, size, current_time, priority)
            heapq.heappush(self.event_queue, (current_time, 'inject', packet))
            current_time += np.random.exponential(avg_interval)

    def _select_endpoints(self, pattern: TrafficPattern) -> Tuple[int, int]:
        if pattern == TrafficPattern.UNIFORM:
            src = np.random.randint(0, 16)
            dst = np.random.randint(0, 16)
            while dst == src:
                dst = np.random.randint(0, 16)
        elif pattern == TrafficPattern.HOTSPOT:
            src = np.random.choice(self.hotspots)
            dst = np.random.choice(self.hotspots)
            while dst == src:
                dst = np.random.choice(self.hotspots)
        elif pattern == TrafficPattern.TRANSPOSE:
            src = np.random.randint(0, 16)
            dst = (src + 8) % 16
        elif pattern == TrafficPattern.BIT_COMPLEMENT:
            src = np.random.randint(0, 16)
            dst = 15 - src
        return src, dst

    def _generate_packet_size(self) -> int:
        rand = np.random.random()
        if rand < 0.7:
            return int(2**10 * (1 + 9 * np.random.random()))  # 1KB-10KB
        elif rand < 0.95:
            return int(2**10 * (10 + 90 * np.random.random()))  # 10KB-100KB
        else:
            return int(2**20 * (0.1 + 9.9 * np.random.random()))  # 100KB-10MB

    def _calculate_priority(self, src: int, scheme: PriorityScheme) -> float:
        node = self.network.nodes[src]
        if scheme == PriorityScheme.LEVEL:
            return node.buffer_utilization
        elif scheme == PriorityScheme.ID:
            return src / 15.0
        elif scheme == PriorityScheme.AGE:
            return np.random.random()  # Simulate aging
        elif scheme == PriorityScheme.HYBRID:
            return (node.buffer_utilization + (src / 15.0)) / 2.0
        return 0.0

class EnhancedPacketLogger:
    def __init__(self, network: TorusNetwork):  # Now requires network reference
        self.minimal_path_cache = {}
        self.reroute_threshold = 64 * 0.8  # From NETWORK_CONFIG
        self.network = network  # Store network instance

    def calculate_rerouting(self, packet: Packet) -> Dict:
        actual_length = len(packet.route) - 1
        minimal_length = self.minimal_distance(packet.src, packet.dst)
        deviation = actual_length - minimal_length

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
        if (src, dst) not in self.minimal_path_cache:
            x1, y1 = src % 4, src // 4
            x2, y2 = dst % 4, dst // 4
            dx = min(abs(x1 - x2), 4 - abs(x1 - x2))
            dy = min(abs(y1 - y2), 4 - abs(y1 - y2))
            self.minimal_path_cache[(src, dst)] = dx + dy
        return self.minimal_path_cache[(src, dst)]