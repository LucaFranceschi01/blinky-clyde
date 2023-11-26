from myutils import PacmanQAgent, SimpleExtractor

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
import random
import util
import json

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ApproximateQAgent', second='ApproximateQAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

class ApproximateQAgent(PacmanQAgent):
    def __init__(self, index, time_for_computing=.1, **args):
        self.featExtractor = SimpleExtractor()
        PacmanQAgent.__init__(self, index, time_for_computing, **args)

        with open('src/contest/agents/blinky-clyde/weights.txt', 'r') as fin:
            raw_weights = fin.read()
        
        self.weights = util.Counter(json.loads(raw_weights))
        
        self.start = None
        # {'bias':0, '#_of_enemies_at_1_step':0,'dist_closest_food': 0.0, 'x_pos': 0, 'y_pos':0, 'dist_closest_enemy':0, 'dist_closest_vulnerable_enemy':0, 'food_carrying':0, }

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        PacmanQAgent.register_initial_state(self, game_state)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
     
        if len(legalActions) == 0:
            return None
        
        #Compute maximum Q-value for each possible action of a state, and return the value and the action taken
        q_max = [float('-inf'), legalActions[0]]
        for a in legalActions: 
            q_max = max(q_max, [self.getQValue(state, a), a], key=lambda x:x[0])
        return q_max[1]
    
    def getWeights(self): # gives some strange error
        return self.weights
    
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        return self.weights * self.weights # * self.featExtractor.getFeatures(state, action)
	# es temporal hasta que tengamos las features hechas 

    def update(self, state, action, nextState, reward, episode):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        delta = (float(reward) + self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for key in self.featExtractor.getFeatures(state, action):
            self.weights[key] += self.alpha * delta * self.featExtractor.getFeatures(state, action)[key]

        if episode % 2 == 0: # 2 es temporal
            with open('src/contest/agents/blinky-clyde/weights_out.txt', 'w') as fout: 
                fout.write(json.dumps(self.weights))

    def get_action(self, state):
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
          return action
        
        action = self.computeActionFromQValues(state) #Take best policy action
        
        return action
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}