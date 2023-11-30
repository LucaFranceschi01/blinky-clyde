from capture import GameState
from util import nearestPoint
import random
import util
import json
from game import Actions
from captureAgents import CaptureAgent

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

class ApproximateQAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # PacmanQAgent.__init__(self, index, time_for_computing, **args)
        # args['epsilon'] = epsilon
        # args['gamma'] = gamma
        # args['alpha'] = alpha
        # args['numTraining'] = numTraining
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        # self.numTraining = int(numTraining)
        self.epsilon = float(0.5)
        self.alpha = float(0.5)
        self.discount = float(1)
        self._distributions = None
        self.index = index
        self.red = None
        self.agentsOnTeam = None
        self.distancer = None
        self.observationHistory = []
        self.timeForComputing = time_for_computing
        self.display = None

        with open('agents/blinky-clyde/weights.txt', 'r') as fin: # habrá que cambiarlo por solo weights.txt
            raw_weights = fin.read()

        self.weights = util.Counter(json.loads(raw_weights))
        
        self.start = None
        self.action = None
        self.next_state = None
        self.episode = 0
        # {"bias":0, "#_of_enemies_at_1_step":0, "#_of_enemies_at_2_step":0, "dist_closest_food": 0.0, "x_pos": 0, "y_pos":0, "dist_closest_enemy":0, "dist_closest_vulnerable_enemy":0, "food_carrying":0}

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        self.next_state = game_state

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        legalActions = state.get_legal_actions(self.index)
     
        if len(legalActions) == 0:
            return 0.0

        #Compute maximum Q-value for each possible action of a state, and return the value
        q_max = float('-inf')
        for a in state.get_legal_actions(self.index): 
            q_max = max(q_max, self.getQValue(state, a))
        return q_max

    def computeActionFromQValues(self, game_state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = game_state.get_legal_actions(self.index)
     
        if len(legalActions) == 0:
            return None
        
        #Compute maximum Q-value for each possible action of a game_state, and return the value and the action taken
        q_max = [float('-inf'), legalActions[0]]
        for a in legalActions: 
            q_max = max(q_max, [self.getQValue(game_state, a), a], key=lambda x:x[0])
        self.action = q_max[1]
        return q_max[1]
    
    def getWeights(self): # gives some strange error
        return self.weights
    
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        self.features = self.get_features(state, action)
        return self.features * self.get_weights(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        delta = (float(reward) + self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for key in self.features:
            self.weights[key] += self.alpha * delta * self.get_features(state, action)[key]

        with open('agents/blinky-clyde/weights.txt', 'w') as fout: 
            fout.write(json.dumps(self.weights))
                
    def get_reward(self, state, action):
        '''A denser way of getting rewards that takes into account:
        More reward the closest to food
        If very near to an enemy, less reward'''
        reward = 0

        reward -= self.features['dist-closest-food'] * 100
        reward += self.features['eats-food']
        reward += self.features['eats-enemy']
        reward -= self.features['#-of-enemies-1-step-away'] * 10

        return reward
    
    def final(self, state):
        self.episode += 1
        reward = self.get_reward(state, self.action)

        self.update(state, self.action, self.next_state, reward)

        # if self.episode % 2 == 0: # 2 es temporal

    def choose_action(self, game_state):
        legalActions = game_state.get_legal_actions(self.index)
        action = None

        if len(legalActions) == 0:
          return action
        
        if random.random() < self.epsilon:
            action = self.computeActionFromQValues(game_state) # Take best policy action
            # self.update(game_state, action, self.get_successor(game_state, action), self.get_reward(game_state, action))
        else:
            action = random.choice(legalActions)
        
        return action

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
        
    def closestFood(self, pos, food):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        dist_food = float('inf')

        for i in range(food.width):
            for j in range(food.height):
                if food[i][j]:
                    dist_food = min(dist_food, self.get_maze_distance(pos, (i, j)))

    def get_features(self, game_state: GameState, action):
        """
        Returns a counter of features for the state
        """
        # features = util.Counter()
        # successor = self.get_successor(game_state, action)
        # features['successor_score'] = self.get_score(successor)
        
        # Initialize helpful variables
        features = util.Counter()
        if game_state.is_on_red_team(self.index):
            enemy_food = game_state.get_blue_food()
            team_food = game_state.get_red_food()
            enemies = game_state.get_blue_team_indices()
        else:
            enemy_food = game_state.get_red_food()
            team_food = game_state.get_blue_food()
            enemies = game_state.get_red_team_indices()

        walls = game_state.get_walls()
        score = game_state.get_score()

        ####################################  
        distances = game_state.get_agent_distances()
        if len(distances) == 0: # chapuza gorda
            distances = self.distances
        else:
            self.distances = distances
        enemy_states = [game_state.data.agent_states[enemy] for enemy in enemies]

        # Current and future locations of the agent
        x, y = game_state.get_agent_position(self.index)
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_x_2, next_y_2 = int(x + 2*dx), int(y + 2*dy) # NO ME CONVENCE, son 2 steps en linea recta, no en otras direcciones que también estén a 2 steps

        # FEATURE 1: BIAS
        features["bias"] = 1.0

        # FEATURE 2: NUMBER OF ENEMIES AT ONE STEP
        features["#-of-enemies-1-step-away"] = sum([distances[enemy] < 3 for enemy in enemies])

        # FEATURE 3: NUMBER OF ENEMIES TWO STEPS AWAY
        features["#-of-enemies-2-step-away"] = sum([distances[enemy] < 4 for enemy in enemies])

        # FEATURE 4: NUMBER OF SCARED ENEMIES ONE STEP AWAY
        features["#-of-scared-enemies-1-step-away"] = sum([distances[enemy] < 5 and 
                game_state.get_agent_state(enemy).scared_timer > 1 for enemy in enemies])
        
        if (x == 4 and y == 14):
            print('')

        # FEATURES 5 AND 6: EAT FOOD OR ENEMY
        features["eats-food"] = features["eats-enemy"] = 0.0
        if features["#-of-scared-enemies-1-step-away"] > 0:
            features["eats-enemy"] = 1.0
        elif features["#-of-enemies-1-step-away"] == 0 and (next_x, next_y) in enemy_food:
            features["eats-food"] = 1.0

        # FEATURE 7: DISTANCE TO CLOSEST FOOD
        current_dist_food = self.closestFood((x, y), enemy_food)
        next_dist_food = self.closestFood((next_x, next_y), enemy_food)
        if current_dist_food is not None and next_dist_food is not None:
            # make the distance a number less than one otherwise the update will diverge wildly
            # print(features["dist-closest-food"])
            features["dist-closest-food"] = float(current_dist_food-next_dist_food) / (walls.width * walls.height)
        # features.divideAll(10.0)

        # FEATURES 8 AND 9: POSITION IN THE BOARD
        # features["x-pos"] = next_x / walls.width - 0.5 # Normalized position in the board, with 0 in the center
        # features["y-pos"] = next_y / walls.height

        # FEATURE 10: FOOD_CARRYING
        features["food-carrying"] = game_state.get_agent_state(self.index).num_carrying

        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game game_state.  They can be either
        a counter or a dictionary.
        """
        return self.weights