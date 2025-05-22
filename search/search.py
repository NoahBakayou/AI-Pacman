# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    from util import Stack  # Import Stack for DFS

    stack = Stack()  # Stack to keep track of positions and paths
    visited = set()  # Set to store visited positions

    # Get the starting position of Pac-Man
    start_position = problem.getStartState()
    start_path = []  # No moves have been made yet

    # Push the start position and path into the stack
    stack.push([start_position, start_path])

    while not stack.isEmpty():
        # Pop the most recent position and path from the stack
        current_position, current_path = stack.pop()

        # If Pac-Man has reached the goal, return the path taken
        if problem.isGoalState(current_position):
            return current_path

        # If this position has not been visited yet, process it
        if current_position not in visited:
            visited.add(current_position)  # Mark as visited

            # Get all possible next moves from the current position
            for next_position, action, _ in problem.getSuccessors(current_position):
                if next_position not in visited:
                    new_path = current_path + [action]  # Create an updated path
                    stack.push([next_position, new_path])  # Push new state into stack

    return []  # Return an empty list if no solution is found

def breadthFirstSearch(problem):
    """
    Implements Breadth-First Search (BFS) for Pac-Man in a grid-based environment.
    Returns a list of actions that lead to the goal.
    """
    from util import Queue  # Import Queue for BFS

    queue = Queue()  # Queue for BFS (FIFO)
    visited = set()  # Tracks visited positions

    # Get the starting position of Pac-Man
    start_position = problem.getStartState()
    start_path = []  # No moves yet

    # Push the start position and path into the queue
    queue.push([start_position, start_path])

    while not queue.isEmpty():
        # Remove the first item from the queue (FIFO)
        current_position, current_path = queue.pop()

        # If Pac-Man has reached the goal, return the path taken
        if problem.isGoalState(current_position):
            return current_path

        # If this position hasn't been visited, explore it
        if current_position not in visited:
            visited.add(current_position)  # Mark as visited

            # Get all possible next moves
            for next_position, action, _ in problem.getSuccessors(current_position):
                if next_position not in visited:
                    new_path = current_path + [action]  # Update path
                    queue.push([next_position, new_path])  # Add new state to queue

    return []  # Return an empty list if no solution is found

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """
    Implements Uniform-Cost Search (UCS) for Pac-Man.
    Returns a list of actions that lead to the goal.
    """
    from util import PriorityQueue  # Import Priority Queue for UCS

    priority_queue = PriorityQueue()  # Priority queue stores (state, path, cost)
    visited = {}  # Dictionary to store best known cost for each state

    # Get the starting position
    start_position = problem.getStartState()
    start_path = []  # No moves yet
    start_cost = 0  # Initial cost is zero

    # Push the start state into the priority queue with priority = 0
    priority_queue.push((start_position, start_path, start_cost), start_cost)

    while not priority_queue.isEmpty():
        # Pop the state with the lowest cost
        current_position, current_path, current_cost = priority_queue.pop()

        # If we reached the goal, return the path
        if problem.isGoalState(current_position):
            return current_path

        # Check if we have already visited this state with a lower cost
        if current_position not in visited or current_cost < visited[current_position]:
            visited[current_position] = current_cost  # Update best cost for this state

            # Expand current state and add successors to the queue
            for next_position, action, step_cost in problem.getSuccessors(current_position):
                new_cost = current_cost + step_cost  # Update cost
                new_path = current_path + [action]  # Update path
                priority_queue.push((next_position, new_path, new_cost), new_cost)

    return []  # Return empty list if no solution found

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """
    Implements A* Search for Pac-Man.
    Uses a heuristic function to guide the search.
    Returns a list of actions leading to the goal.
    """
    from util import PriorityQueue  # Import priority queue

    priority_queue = PriorityQueue()  # Priority queue for A* (stores state, path, cost)
    visited = {}  # Dictionary to store best known cost for each state

    # Get the starting position
    start_position = problem.getStartState()
    start_path = []  # No moves yet
    start_cost = 0  # Initial cost is zero

    # Compute f(n) = g(n) + h(n)
    start_priority = start_cost + heuristic(start_position, problem)

    # Push the start state into the priority queue
    priority_queue.push((start_position, start_path, start_cost), start_priority)

    while not priority_queue.isEmpty():
        # Pop the state with the lowest f(n)
        current_position, current_path, current_cost = priority_queue.pop()

        # If we reached the goal, return the path
        if problem.isGoalState(current_position):
            return current_path

        # Check if we have already visited this state with a lower cost
        if current_position not in visited or current_cost < visited[current_position]:
            visited[current_position] = current_cost  # Update best cost for this state

            # Expand current state and add successors to the queue
            for next_position, action, step_cost in problem.getSuccessors(current_position):
                new_cost = current_cost + step_cost  # Update path cost g(n)
                heuristic_cost = heuristic(next_position, problem)  # Compute h(n)
                new_priority = new_cost + heuristic_cost  # Compute f(n)

                new_path = current_path + [action]  # Update path
                priority_queue.push((next_position, new_path, new_cost), new_priority)

    return []  # Return empty list if no solution found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
