import heapq
import logging
import math
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


Position = tuple[int, int]


@dataclass
class MapNode:
    """Normalized node from the map DAG keyed by (col, row)."""

    position: Position
    node_type: str
    cost: float
    children: list[Position] = field(default_factory=list)


@dataclass
class MapGraph:
    """A compact graph representation used by Dijkstra."""

    nodes: dict[Position, MapNode] = field(default_factory=dict)
    targets: set[Position] = field(default_factory=set)


@dataclass
class MapRoutePlan:
    """The currently planned route and its total weighted cost."""

    path: list[Position] = field(default_factory=list)
    cost: float = math.inf


class MapPolicy:
    NODE_COSTS = {
        "Monster": 8.0,
        "Elite": 10.0,
        "Boss": 20.0,
        "Ancient": 0.0,
        "RestSite": 0.0,
        "Shop": 0.0,
        "Treasure": 0.0,
        "Unknown": 5.0,
    }
    DEFAULT_NODE_COST = 0.0

    def __init__(self):
        # These persist across map screens in one run so the policy follows one
        # planned route instead of recomputing a different path every floor.
        self._graph = MapGraph()
        self._shortest_path = MapRoutePlan()

    def choose_action(self, state: dict) -> dict:
        map_state = state.get("map", {})
        next_options = map_state.get("next_options", state.get("next_options", []))
        logger.debug("MapPolicy: choosing action next_options=%d", len(next_options))

        if not next_options:
            self._shortest_path = MapRoutePlan()
            action = {"type": "proceed"}
            logger.debug("MapPolicy: selected action=%s", action)
            return action

        if self._should_initialize_route(map_state):
            self._initialize_route(map_state)

        option = self._planned_next_option(next_options, map_state)
        if option is None:
            option, route, cost = self._best_route_from_options(next_options)
            self._shortest_path = MapRoutePlan(route, cost)
            logger.info(
                "MapPolicy: best route cost=%.2f route=%s",
                cost,
                self._format_route(route),
            )
            logger.debug(
                "MapPolicy: scheduled route cost=%.2f route=%s",
                cost,
                self._shortest_path.path,
            )

        self._consume_planned_option(option)
        action = {
            "type": "choose_map_node",
            "index": option.get("index", 0),
        }
        logger.debug(
            "MapPolicy: selected node index=%s type=%s col=%s row=%s",
            option.get("index"),
            option.get("type"),
            option.get("col"),
            option.get("row"),
        )
        logger.debug("MapPolicy: selected action=%s", action)
        return action

    def _should_initialize_route(self, map_state: dict) -> bool:
        """Start a new route plan at the beginning of a fresh map."""
        visited = map_state.get("visited", [])
        return not visited or not self._graph.nodes

    def _initialize_route(self, map_state: dict) -> None:
        """Build the normalized graph and store the full Dijkstra route."""
        self._graph = self._build_graph(map_state)
        start = self._route_start(map_state)

        if start is None:
            self._shortest_path = MapRoutePlan()
            logger.info("MapPolicy: could not initialize route; missing start position")
            return

        cost, path = self._dijkstra(start)
        self._shortest_path = MapRoutePlan(path, cost)
        logger.info(
            "MapPolicy: initialized best route cost=%.2f route=%s",
            cost,
            self._format_route(path),
        )

    def _build_graph(self, map_state: dict) -> MapGraph:
        """Convert raw API map nodes into a stable adjacency structure."""
        nodes: dict[Position, MapNode] = {}
        for raw_node in map_state.get("nodes", []):
            position = self._node_pos(raw_node)
            if position is None:
                continue
            node_type = self._normalize_node_type(raw_node.get("type", "Unknown"))
            nodes[position] = MapNode(
                position=position,
                node_type=node_type,
                cost=self._node_cost(raw_node),
                children=self._child_positions(raw_node),
            )

        return MapGraph(
            nodes=nodes,
            targets=self._target_positions(map_state, nodes),
        )

    def _route_start(self, map_state: dict) -> Position | None:
        """Prefer current_position, falling back to the last visited node."""
        current_position = self._node_pos(map_state.get("current_position", {}))
        if current_position in self._graph.nodes:
            return current_position

        visited = map_state.get("visited", [])
        if visited:
            visited_position = self._node_pos(visited[-1])
            if visited_position in self._graph.nodes:
                return visited_position

        return None

    def _best_route_from_options(
        self,
        next_options: list[dict],
    ) -> tuple[dict, list[Position], float]:
        """Re-plan from legal next choices when the stored route is unavailable."""
        best_option = next_options[0]
        best_route = self._single_node_route(best_option)
        best_cost = math.inf

        for option in next_options:
            start = self._node_pos(option)
            if start is None:
                continue
            cost, route = self._dijkstra(start)
            if not route:
                cost, route = self._lookahead_cost(option), [start]

            tie_breaker = (
                -self._parse_int(option.get("row")),
                self._parse_int(option.get("col")),
                self._parse_int(option.get("index")),
            )
            best_tie_breaker = (
                -self._parse_int(best_option.get("row")),
                self._parse_int(best_option.get("col")),
                self._parse_int(best_option.get("index")),
            )
            if cost < best_cost or (cost == best_cost and tie_breaker < best_tie_breaker):
                best_option = option
                best_route = route
                best_cost = cost

        return best_option, best_route, best_cost

    def _dijkstra(
        self,
        start: Position,
    ) -> tuple[float, list[Position]]:
        """Find the lowest-cost route from start to the boss/top-row target."""
        if start not in self._graph.nodes:
            return math.inf, []

        distances = {start: self._graph.nodes[start].cost}
        previous: dict[Position, Position | None] = {start: None}
        queue = [(distances[start], start)]

        while queue:
            current_cost, position = heapq.heappop(queue)
            if current_cost != distances[position]:
                continue
            if position in self._graph.targets:
                return current_cost, self._reconstruct_route(position, previous)

            node = self._graph.nodes[position]
            for child_position in node.children:
                child = self._graph.nodes.get(child_position)
                if child is None:
                    continue
                next_cost = current_cost + child.cost
                if next_cost < distances.get(child_position, math.inf):
                    distances[child_position] = next_cost
                    previous[child_position] = position
                    heapq.heappush(queue, (next_cost, child_position))

        return math.inf, []

    def _target_positions(
        self,
        map_state: dict,
        nodes: dict[Position, MapNode],
    ) -> set[Position]:
        """Use the boss node when present; otherwise use all top-row nodes."""
        boss = map_state.get("boss", {})
        boss_position = self._position_from_values(boss.get("col"), boss.get("row"))
        if boss_position in nodes:
            return {boss_position}

        if not nodes:
            return set()

        max_row = max(row for _, row in nodes)
        return {position for position in nodes if position[1] == max_row}

    def _planned_next_option(
        self,
        next_options: list[dict],
        map_state: dict,
    ) -> dict | None:
        """Return the next legal map option that matches the stored route."""
        self._drop_current_and_visited_positions(map_state)
        if not self._shortest_path.path:
            return None

        next_position = self._shortest_path.path[0]
        for option in next_options:
            if self._node_pos(option) == next_position:
                return option

        self._shortest_path = MapRoutePlan()
        return None

    def _drop_current_and_visited_positions(self, map_state: dict) -> None:
        """Remove nodes already reached so path[0] is the next node to click."""
        consumed = {
            position
            for position in [
                self._node_pos(map_state.get("current_position", {})),
                *(self._node_pos(node) for node in map_state.get("visited", [])),
            ]
            if position is not None
        }
        while self._shortest_path.path and self._shortest_path.path[0] in consumed:
            self._shortest_path.path.pop(0)

    def _consume_planned_option(self, option: dict) -> None:
        """Advance the stored route after choosing its next option."""
        position = self._node_pos(option)
        if (
            position is not None
            and self._shortest_path.path
            and self._shortest_path.path[0] == position
        ):
            self._shortest_path.path.pop(0)

    def _reconstruct_route(
        self,
        target: Position,
        previous: dict[Position, Position | None],
    ) -> list[Position]:
        route = []
        position: Position | None = target
        while position is not None:
            route.append(position)
            position = previous.get(position)
        route.reverse()
        return route

    def _child_positions(self, node: dict) -> list[Position]:
        positions = []
        for child in node.get("children", []):
            if isinstance(child, dict):
                position = self._position_from_values(child.get("col"), child.get("row"))
            elif isinstance(child, (list, tuple)) and len(child) >= 2:
                position = self._position_from_values(child[0], child[1])
            else:
                position = None
            if position is not None:
                positions.append(position)
        return positions

    def _node_cost(self, node: dict) -> float:
        node_type = self._normalize_node_type(node.get("type", "Unknown"))
        return self.NODE_COSTS.get(node_type, self.DEFAULT_NODE_COST)

    def _lookahead_cost(self, option: dict) -> float:
        cost = self._node_cost(option)
        leads_to = option.get("leads_to", [])
        if leads_to:
            cost += min(self._node_cost(child) for child in leads_to)
        return cost

    def _single_node_route(self, node: dict) -> list[Position]:
        position = self._node_pos(node)
        if position is None:
            return []
        return [position]

    def _format_route(
        self,
        route: list[Position],
    ) -> str:
        parts = []
        for col, row in route:
            node = self._graph.nodes.get((col, row))
            node_type = node.node_type if node is not None else "Unknown"
            parts.append(f"{node_type}({col},{row})")
        return " -> ".join(parts)

    def _node_pos(self, node: dict) -> Position | None:
        return self._position_from_values(node.get("col"), node.get("row"))

    def _position_from_values(self, col, row) -> Position | None:
        if col is None or row is None:
            return None
        return (self._parse_int(col), self._parse_int(row))

    def _parse_int(self, value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalize_node_type(self, node_type) -> str:
        if node_type is None:
            return "Unknown"
        return str(node_type).replace("_", " ").strip()
