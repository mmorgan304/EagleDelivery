from collections import deque
import heapq
from itertools import count


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


def reconstruct_path(parents, goal):
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def bfs(campus, start, goal):
    frontier = deque([start])
    parents = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            if successor not in parents:
                parents[successor] = state
                frontier.append(successor)
    return SearchResult(None, expanded)


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def astar(campus, start, goal):
    frontier = []
    heapq.heappush(frontier, (manhattan(start, goal), 0, start))
    parents = {start: None}
    g_costs = {start: 0} # g_costs[state] = cost to reach state from start
    expanded = 0
    while frontier: # while anything is in the frontier
        f_cost, g_cost, state = heapq.heappop(frontier) #pop the state with the lowest predicted cost
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded) # if we reached the goal, return the path and number of expanded nodes
        expanded += 1 # else, we expand the node and add its neighbors to the frontier
        for action, successor in campus.neighbors(state):
            new_g_cost = g_cost + 1
            insertion_number = next(count()) # tie-breaker for nodes with the same f_cost
            if successor not in g_costs or new_g_cost < g_costs[successor]: #only add the successor to the frontier if we haven't seen it before or if we found a cheaper path to it
                g_costs[successor] = new_g_cost
                parents[successor] = state
                f_cost = new_g_cost + manhattan(successor, goal)
                heapq.heappush(frontier, (f_cost, insertion_number, successor)) # add the successor to the frontier with its f_cost and insertion number
    return SearchResult(None, expanded) # if we exhaust the frontier without finding the goal, return None and the number of expanded nodes


def epsilon_greedy(q_values, state, actions, epsilon, rng):
    """TODO 3: Choose an action with epsilon-greedy exploration."""
    # q_values is a dictionary. Its keys are (state, action) pairs.
    # Example key: ((1, 1), "RIGHT"). Unseen entries have value 0.0.
    raise NotImplementedError("Complete TODO 3: epsilon_greedy")


def q_learning_update(q_values, state, action, reward, next_state,
                      actions, alpha, gamma, terminal):
    """TODO 4: Update exactly one Q-table entry in place; return nothing."""
    # Get Q(state, action), defaulting to 0.0.
    # If terminal is True, the target is reward with no continuation term.
    # Otherwise target = reward + gamma * max Q(next_state, next_action).
    # Consider every action, including unseen entries with value 0.0.
    # Apply Q <- Q + alpha * (target - Q).
    raise NotImplementedError("Complete TODO 4: q_learning_update")
