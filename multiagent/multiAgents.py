# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    from util import manhattanDistance

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the base score
        score = successorGameState.getScore()

        # Distance to the closest food
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, foodPos) for foodPos in foodList)
            score += 10.0 / minFoodDist  # Closer to food = better (reciprocal)
        
        # Avoid ghosts
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0:  # Ghost is dangerous
                if dist < 2:
                    score -= 1000  # Very bad to be close to a ghost

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:
                # Pacman's turn (maximize)
                bestValue = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = minimax(1, depth, successor)  # Next agent (first ghost)
                    bestValue = max(bestValue, value)
                return bestValue
            else:
                # Ghost's turn (minimize)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1  # After last ghost, go back to Pacman, increase depth

                bestValue = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = minimax(nextAgent, nextDepth, successor)
                    bestValue = min(bestValue, value)
                return bestValue

        # Now, choose the best action for Pacman
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            numAgents = gameState.getNumAgents()

            if agentIndex == 0:
                # Pacman (maximize)
                bestValue = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = expectimax(1, depth, successor)
                    bestValue = max(bestValue, value)
                return bestValue
            else:
                # Ghost (expected value)
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1

                actions = gameState.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(gameState)

                total = 0
                probability = 1.0 / len(actions)  # Uniform probability
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    total += expectimax(nextAgent, nextDepth, successor) * probability
                return total

        # Choose best action for Pacman
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    from util import manhattanDistance

    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Feature 1: Closest food distance
    if food:
        minFoodDist = min(manhattanDistance(pacmanPos, f) for f in food)
        score += 10.0 / minFoodDist  # Prioritize approaching food

    # Feature 2: Ghost distance (avoid if not scared)
    for ghost in ghostStates:
        dist = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            if dist < 2:
                score -= 200  # Penalize proximity to dangerous ghost
            else:
                score -= 2.0 / dist  # Slight penalty for being too close
        else:
            # If scared, chase ghost
            score += 10.0 / (dist + 1)

    # Feature 3: Remaining food count
    score -= 4 * len(food)

    # Feature 4: Remaining capsules (encourage using them)
    score -= 15 * len(capsules)

    return score


# Abbreviation
better = betterEvaluationFunction

