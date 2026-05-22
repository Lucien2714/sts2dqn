import logging
import random
import heapq

logger = logging.getLogger(__name__)

NODE_COSTS = {
    "elite": 10,
    "monster": 5,   
    "event": 2,     
    "rest": 0,      
    "shop": 0,
    "boss": 0       
}

class MapPolicy:
    def _get_node_cost(self, node_type: str) -> int:
        """Returns the assigned cost for a node type, defaulting to 5 for unknown types."""
        if not node_type:
            return 5
        return NODE_COSTS.get(node_type.lower(), 5)

    def _build_map_graph(self, map_state: dict) -> dict:
        """Parses the full map state from the API into a navigable graph."""
        graph = {}
        all_nodes = map_state.get("nodes", [])

        for node in all_nodes:
            node_id = f"{node.get('col')},{node.get('row')}"
            
            children_ids = []

            for child_coords in node.get("children", []):
                if isinstance(child_coords, list) and len(child_coords) >= 2:
                    children_ids.append(f"{child_coords[0]},{child_coords[1]}")

            graph[node_id] = {
                "type": node.get("type", "unknown"),
                "children": children_ids
            }
        return graph

    def _dijkstra_path_cost(self, start_node_id: str, graph: dict) -> tuple[float, list[str]]:
        """Calculates the minimum cost to reach the end of the map and returns the full path."""
        if start_node_id not in graph:
            return float('inf'), []

        pq = [(0, start_node_id, [start_node_id])]
        visited_costs = {start_node_id: 0}

        while pq:
            current_cost, node_id, path = heapq.heappop(pq)
            node_data = graph.get(node_id, {})
            children = node_data.get("children", [])
            node_type = node_data.get("type", "").lower()
            

            if node_type == "boss" or len(children) == 0:
                return current_cost, path
                
            if current_cost > visited_costs.get(node_id, float('inf')):
                continue
                
            for child_id in children:
                child_type = graph.get(child_id, {}).get("type", "")
                edge_cost = self._get_node_cost(child_type)
                
                new_cost = current_cost + edge_cost
                
                if new_cost < visited_costs.get(child_id, float('inf')):
                    visited_costs[child_id] = new_cost
                    new_path = path + [child_id]
                    heapq.heappush(pq, (new_cost, child_id, new_path))
                    
        return float('inf'), []
    def choose_action(self, state: dict) -> dict:
        map_state = state.get("map", {})
        nodes = map_state.get("next_options", state.get("next_options", []))
        logger.debug("MapPolicy: choosing action next_options=%d", len(nodes))

        if not nodes:
            action = {
                "type": "proceed"
            }
            logger.debug("MapPolicy: selected action=%s", action)
            return action

        full_graph = self._build_map_graph(map_state)

        best_node = None
        lowest_cost = float('inf')
        best_full_path = []

        for node in nodes:
            node_id = f"{node.get('col')},{node.get('row')}"
            
            immediate_cost = self._get_node_cost(node.get("type", ""))
            
            future_path_cost, future_path = self._dijkstra_path_cost(node_id, full_graph)
            total_expected_cost = immediate_cost + future_path_cost
            
            if total_expected_cost < lowest_cost:
                lowest_cost = total_expected_cost
                best_node = node
                best_full_path = future_path

        if best_full_path:
            path_string = " -> ".join(best_full_path)
            print(f"\nOptimal path to Boss (Cost: {lowest_cost})")
            print(f"Route: {path_string}\n")
            logger.info("Path selected: %s", path_string)

        if best_node is None:
            best_node = random.choice(nodes)
            logger.warning("MapPolicy: Dijkstra evaluation failed, defaulting to random choice.")

        try:
            correct_index = nodes.index(best_node)
        except ValueError:
            correct_index = 0
            
        action = {
            "type": "choose_map_node",
            "index": correct_index,
        }
        # ----------------------------------------------
        
        logger.debug("MapPolicy: selected action=%s", action)
        return action
