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
        # self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.epsilon = float(0.8)
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

        with open('agents/blinky-clyde/weights.txt', 'r') as fin:
            raw_weights = fin.read()

        self.weights = util.Counter(json.loads(raw_weights))
        
        self.start = None
        self.action = None
        self.next_state = None
        self.episode = 0

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
        
        # Compute maximum Q-value for each possible action of a game_state, and return the value and the action taken
        q_max = [float('-inf'), legalActions[0]]
        for a in legalActions: 
            q_max = max(q_max, [self.getQValue(game_state, a), a], key=lambda x:x[0])
        self.action = q_max[1]
        return q_max[1]
    
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.get_features(state, action) * self.get_weights()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        delta = (float(reward) + self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for key in self.features:
            self.weights[key] += self.alpha * delta * self.get_features(state, action)[key]
        # self.weights.normalize()

        # with open('agents/blinky-clyde/weights.txt', 'w') as fout: 
        #     fout.write(json.dumps(self.weights))
                
    def get_reward(self, state, action):
        '''A denser way of getting rewards that takes into account:
        More reward the closest to food
        If very near to an enemy, less reward'''

        reward = 0
        if not state.is_over():
            my_state = state.get_agent_state(self.index)
            successor = self.get_successor(state, action)
            current_pos = state.get_agent_position(self.index)
            next_pos = successor.get_agent_position(self.index)
            
            distances = state.get_agent_distances()
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

            food = self.get_food(state)

            if len(distances) == 0: # chapuza gorda
                distances = self.distances
            else:
                self.distances = distances

            # Count food left to eat
            remaining_food = 0
            for i in range(food.width):
                for j in range(food.height):
                    if food[i][j]:
                        remaining_food += 1

            # WHILE THERE IS FOOD TO EAT, GO TOWARDS IT
            if remaining_food > 0:
                food_dist_curr = self.get_maze_distance(current_pos, self.closestFood(current_pos, food)[1])
                food_dist_new = self.get_maze_distance(next_pos, self.closestFood(next_pos, food)[1])
                reward += (food_dist_curr - food_dist_new) / remaining_food

            # IF THERE IS AN INVADER, GO TOWARDS IT
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            invader_dist_curr = invader_dist_new = float('inf')
            for inv in invaders:
                invader_dist_curr = min(invader_dist_curr, self.get_maze_distance(current_pos, inv.get_position()))
                invader_dist_new = min(invader_dist_new, self.get_maze_distance(next_pos, inv.get_position()))

            if len(invaders) > 0:
                if my_state.scared_timer > 0:
                    reward += invader_dist_new - invader_dist_curr
                else:
                    reward += invader_dist_curr - invader_dist_new

            # IF THERE IS A DEFENDER, RUN
            defenders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            defender_dist_curr = defender_dist_new = float('inf')
            for df in defenders:
                defender_dist_curr = min(defender_dist_curr, self.get_maze_distance(current_pos, df.get_position()))
                defender_dist_new = min(defender_dist_new, self.get_maze_distance(next_pos, df.get_position()))

            if len(defenders) > 0: # avoid +inf ?
                reward += defender_dist_new - defender_dist_curr

            self.reward = reward
        else:
            reward += state.get_score()
    
        return reward
    
    def final(self, state):
        self.episode += 1
        reward = self.get_reward(state, self.action)
        self.update(state, self.action, self.next_state, reward)

        with open('agents/blinky-clyde/weights.txt', 'w') as fout: 
            fout.write(json.dumps(self.weights))

    def choose_action(self, game_state):
        legalActions = game_state.get_legal_actions(self.index)
        action = None

        if len(legalActions) == 0:
          return action
        
        if random.random() < self.epsilon:
            action = self.computeActionFromQValues(game_state) # Take best policy action
            self.update(game_state, action, self.get_successor(game_state, action), self.get_reward(game_state, action))
        else:
            action = random.choice(legalActions)

        return action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        if not game_state.is_over():
            successor = game_state.generate_successor(self.index, action)
            pos = successor.get_agent_state(self.index).get_position()
            if pos != nearestPoint(pos):
                # Only half a grid position was covered
                return successor.generate_successor(self.index, action)
            else:
                return successor
        else: return None
        
    def closestFood(self, pos, food):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        dist_food = [float('inf'), (0, 0)]
        for i in range(food.width):
            for j in range(food.height):
                if food[i][j]:
                    dist_food = min(dist_food, [self.get_maze_distance(pos, (i, j)), (i, j)], key=lambda x:x[0])
        return dist_food
      

    def get_features(self, game_state: GameState, action):
        """
        Returns a counter of features for the state
        """
        # Initialize helpful variables
        features = util.Counter()
        if not game_state.is_over():
            my_agent_state = game_state.get_agent_state(self.index)
            successor = self.get_successor(game_state, action)
            enemy_food = self.get_food(game_state)
            # enemy_states = [game_state.data.agent_states[enemy] for enemy in enemies_id]
            walls = game_state.get_walls()
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

            distances = game_state.get_agent_distances()
            if len(distances) == 0: # chapuza gorda
                distances = self.distances
            else:
                self.distances = distances

            # Current and future locations of the agent
            x, y = game_state.get_agent_position(self.index)
            dx, dy = Actions.direction_to_vector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            # FEATURE 1: DISTANCE TO FOOD
            remaining_food = 0
            for i in range(enemy_food.width):
                for j in range(enemy_food.height):
                    if enemy_food[i][j]:
                        remaining_food += 1

            if remaining_food > 0:
                current_distance_to_food = self.get_maze_distance((x, y), self.closestFood((x, y), enemy_food)[1])
                new_distance_to_food = self.get_maze_distance((next_x, next_y), self.closestFood((next_x, next_y), enemy_food)[1])
                features['dist-closest-food'] = float(current_distance_to_food-new_distance_to_food) / (walls.width * walls.height)

            # FEATURE 2: INVADERS CLOSE
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            features['invaders-close'] = len(invaders) / len(enemies)

            # FEATURE 3: DEFENDERS CLOSE
            defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            features['defenders-close'] = len(defenders) / len(enemies)
                
            self.features = features
        else:
            features = self.features

        #FEATURE 3: RETURN FOOD TO HOME
        #distance_to_home = self.get_maze_distance((x, y), self.start)
        #features['return-food'] = float(distance_to_home) / (walls.width * walls.height)

        return features

    def get_weights(self):
        """
        Normally, weights do not depend on the game game_state.  They can be either
        a counter or a dictionary.
        """
        return self.weights