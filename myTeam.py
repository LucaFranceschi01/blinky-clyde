from capture import GameState
from util import nearestPoint
import random
import util
import json
from game import Actions
from captureAgents import CaptureAgent
from myutils import closest_food, count_food

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ApproximateQAgent', second='ApproximateQAgent', num_training=0):
    '''
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ('first' and 'second' are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    '''
    return [eval(first)(first_index), eval(second)(second_index)]

class ApproximateQAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # READ WEIGHTS FROM FILE AND LOAD THEM INTO self.weights
        with open('agents/blinky-clyde/weights.txt', 'r') as fin:
            raw_weights = fin.read()
        self.weights = util.Counter(json.loads(raw_weights))

        # Create helpful variables for Q-values computation
        self.epsilon = float(0.8)
        self.alpha = float(0.5)
        self.discount = float(1)

        self.start = None # Position where the agent is born
        self.action = None # Action that is taken in the current turn (ironically only used in final function)
        self.next_state = None # State resulting from taking self.action in current state (same irony)
        self.distances = None # Distances from the agent to the other agents

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)

        food = self.get_food_you_are_defending(game_state)
        remaining_food = count_food(food)
        self.initial_food = remaining_food

    def computeValueFromQValues(self, game_state):
        '''
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        '''
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return 0.0

        # Compute maximum Q-value for each possible action of a state
        q_max = float('-inf')
        for a in legalActions: 
            q_max = max(q_max, self.getQValue(game_state, a))
        return q_max

    def computeActionFromQValues(self, game_state):
        '''
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        '''
        legalActions = game_state.get_legal_actions(self.index)
        if len(legalActions) == 0:
            return None
        
        # Compute maximum Q-value for each possible action of a game_state, and return the value and the action taken
        q_max = [float('-inf'), legalActions[0]]
        for a in legalActions: 
            q_max = max(q_max, [self.getQValue(game_state, a), a], key=lambda x:x[0])

        self.action = q_max[1]
        self.next_state = self.get_successor(game_state, self.action)
        return self.action
    
    def get_features(self, game_state: GameState, action):
        '''
        Returns a counter of features for the state
        '''
        features = util.Counter()
        if not game_state.is_over():
            # Initialize helpful variables
            state_current = game_state.get_agent_state(self.index)
            state_successor = self.get_successor(game_state, action)
            states_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            states_enemies_successor = [state_successor.get_agent_state(i) for i in self.get_opponents(state_successor)]

            # Count food left to eat
            food_enemy = self.get_food(game_state)
            remaining_food = count_food(food_enemy)

            # Count food remaining food to defend
            food_team = self.get_food_you_are_defending(game_state)
            remaining_food_to_defend = count_food(food_team)
            
            walls = game_state.get_walls()

            # Not used by now
            distances = game_state.get_agent_distances()
            if len(distances) == 0:
                distances = self.distances
            else:
                self.distances = distances

            # Current and future locations of the agent
            position_current = game_state.get_agent_position(self.index)
            position_successor = state_successor.get_agent_position(self.index)

            # FEATURE 1: DISTANCE TO FOOD
            if remaining_food > 0:
                current_distance_to_food = self.get_maze_distance(position_current, closest_food(self, position_current, food_enemy)[1])
                new_distance_to_food = self.get_maze_distance(position_successor, closest_food(self, position_successor, food_enemy)[1])
                features['dist-closest-food'] = float(current_distance_to_food-new_distance_to_food) / (walls.width * walls.height)
            else:
                features['dist-closest-food'] = 0

            # FEATURE 2: INVADERS CLOSE
            invaders = [a for a in states_enemies if a.is_pacman and a.get_position() is not None]
            invader_dist_curr = invader_dist_new = float('inf')
            for inv in invaders:
                invader_dist_curr = min(invader_dist_curr, self.get_maze_distance(position_current, inv.get_position()))
                invader_dist_new = min(invader_dist_new, self.get_maze_distance(position_successor, inv.get_position()))
            if len(invaders) > 0:
                features['invaders-close'] = float(invader_dist_curr-invader_dist_new) / (walls.width * walls.height)
            else:
                features['invaders-close'] = 0

            # FEATURE 3: DEFENDERS CLOSE
            defenders = [a for a in states_enemies if not a.is_pacman and a.get_position() is not None]
            defender_dist_curr = defender_dist_new = float('inf')
            for df in defenders:
                defender_dist_curr = min(defender_dist_curr, self.get_maze_distance(position_current, df.get_position()))
                defender_dist_new = min(defender_dist_new, self.get_maze_distance(position_successor, df.get_position()))
            if len(defenders) > 0:
                features['defenders-close'] = float(defender_dist_curr-defender_dist_new) / (walls.width * walls.height)
            else:
                features['defenders-close'] = 0

            # FEATURE 4: RETURN FOOD HOME
            if game_state.get_agent_state(self.index).num_carrying > 0:
                home_dist_curr = self.get_maze_distance(position_current, self.start)
                home_dist_new = self.get_maze_distance(position_successor, self.start)
                features['return-food'] = float(home_dist_curr - home_dist_new) / walls.width
            else: features['return-food'] = 0

            self.features = features
        else:

            features = self.features

        #FEATURE 3: RETURN FOOD TO HOME
        #distance_to_home = self.get_maze_distance(position_current, self.start)
        #features['return-food'] = float(distance_to_home) / (walls.width * walls.height)

        # FEATURE ---
        # if remaining_food_to_defend > 0:
        #     food_def_dist_curr = self.get_maze_distance(position_current, closest_food(self, position_current, food_team)[1])
        #     food_def_dist_new = self.get_maze_distance(position_successor, closest_food(self, position_successor, food_team)[1])
        #     features['dist-closest-defending-food'] = float(food_def_dist_curr-food_def_dist_new) / (walls.width * walls.height)

        return features

    def get_reward(self, game_state, action):
        '''A denser way of getting rewards that takes into account:
        More reward the closest to food
        If very near to an enemy, less reward'''

        reward = 0
        if not game_state.is_over():
            # Initialize helpful variables 
            state_current = game_state.get_agent_state(self.index)
            state_successor = self.get_successor(game_state, action)
            states_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

            position_current = game_state.get_agent_position(self.index)
            position_successor = state_successor.get_agent_position(self.index)

            food_enemy = self.get_food(game_state)
            food_team = self.get_food_you_are_defending(game_state)
            walls = game_state.get_walls()

            # Not used by now
            distances = game_state.get_agent_distances()
            if len(distances) == 0: # Sometimes it may output no distances, we can replace them and they will be probably similar
                distances = self.distances
            else:
                self.distances = distances

            # Count food left to eat
            remaining_food_to_eat = count_food(food_enemy)

            # Count remaining food to defend
            remaining_food_to_defend = count_food(food_team)

            # Distances to initial position for current and next state
            home_dist_curr = self.get_maze_distance(position_current, self.start)
            home_dist_new = self.get_maze_distance(position_successor, self.start)

            # Distances to closest food to eat
            if remaining_food_to_eat > 0:
                food_dist_curr = self.get_maze_distance(position_current, closest_food(self, position_current, food_enemy)[1])
                food_dist_new = self.get_maze_distance(position_successor, closest_food(self, position_successor, food_enemy)[1])

            # Distances to closest food to defend
            if remaining_food_to_defend > 0:
                food_def_dist_curr = self.get_maze_distance(position_current, closest_food(self, position_current, food_team)[1])
                food_def_dist_new = self.get_maze_distance(position_successor, closest_food(self, position_successor, food_team)[1])

            # Distances to nearest defender (agent is attacking)
            defenders = [a for a in states_enemies if not a.is_pacman and a.get_position() is not None]
            defender_dist_curr = defender_dist_new = float('inf')
            for df in defenders:
                defender_dist_curr = min(defender_dist_curr, self.get_maze_distance(position_current, df.get_position()))
                defender_dist_new = min(defender_dist_new, self.get_maze_distance(position_successor, df.get_position()))

            # Distances to nearest attacker (agent is defending)
            invaders = [a for a in states_enemies if a.is_pacman and a.get_position() is not None]
            invader_dist_curr = invader_dist_new = float('inf')
            for inv in invaders:
                invader_dist_curr = min(invader_dist_curr, self.get_maze_distance(position_current, inv.get_position()))
                invader_dist_new = min(invader_dist_new, self.get_maze_distance(position_successor, inv.get_position()))

            # From variables defined above, which are the rewards ?
            if len(invaders) > 0:
                if state_current.scared_timer > 0:
                    reward += invader_dist_new - invader_dist_curr
                else:
                    reward += invader_dist_curr - invader_dist_new
                    # if invader_dist_new < 1: # SI SE LO COME
                    #     reward += 10
            # elif remaining_food_to_defend < self.initial_food * 0.2:
            #     reward += (food_def_dist_curr - food_def_dist_new) / remaining_food_to_defend
            else:
                if not state_current.is_pacman:
                    reward += (food_dist_curr - food_dist_new) / remaining_food_to_eat
                else:
                    if len(defenders) > 0:
                        if defenders[0].scared_timer > 0:
                            reward += defender_dist_curr - defender_dist_new
                        else:
                            reward += defender_dist_new - defender_dist_curr
                    else:
                        if state_current.num_carrying > 5 or remaining_food_to_eat == 0:
                            reward += (home_dist_curr - home_dist_new) / walls.width
                        else:
                            reward += (food_dist_curr - food_dist_new) / remaining_food_to_eat

            self.reward = reward

        return reward

    def get_weights(self):
        '''
        Normally, weights do not depend on the game game_state.  They can be either
        a counter or a dictionary.
        '''
        return self.weights

    def getQValue(self, game_state, action):
        '''
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        '''
        return self.get_features(game_state, action) * self.get_weights()

    def update(self, game_state, action, nextState, reward):
        '''
           Should update your weights based on transition
        '''
        delta = (float(reward) + self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(game_state, action)
        for key in self.features:
            self.weights[key] += self.alpha * delta * self.get_features(game_state, action)[key]
        # self.weights.normalize()

        # with open('agents/blinky-clyde/weights.txt', 'w') as fout: 
        #     fout.write(json.dumps(self.weights))

    def final(self, game_state):
        '''
        Is called when the game is finished, last opportunity for updating weights
        '''
        reward = self.get_reward(game_state, self.action)
        self.update(game_state, self.action, self.next_state, reward)

        with open('agents/blinky-clyde/weights.txt', 'w') as fout: 
            fout.write(json.dumps(self.weights))

    def choose_action(self, game_state):
        '''
        Returns the action chosen for this game_state according to the Approximate Q-value Model
        self.epsilon of the times, otherwise it chooses a random legal action
        '''
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
        '''
        Finds the next successor which is a grid position (location tuple).
        '''
        if not game_state.is_over():
            successor = game_state.generate_successor(self.index, action)
            pos = successor.get_agent_state(self.index).get_position()
            if pos != nearestPoint(pos):
                # Only half a grid position was covered
                return successor.generate_successor(self.index, action)
            else:
                return successor
        else: return None