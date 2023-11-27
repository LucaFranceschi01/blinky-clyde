import contest.distanceCalculator as distanceCalculator
from game import Directions, Actions, Agent
import util
import time
import json

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

#TO DO
def closestGhost(pos, blue_ghost, walls):
    """
    Return closest ghost to our position
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a ghost at this location then exit
        if blue_ghost[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no ghost found
    return None


#TO DO
def closestScaredGhost(pos, blue_scared_ghost, walls):
    """
    Return closest ghost to our position
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a scared ghost at this location then exit
        if blue_scared_ghost[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no scared ghost found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.get_red_food()
        walls = state.get_walls()

        ####################################       
        ghosts = state.get_agent_position(state.blue_team[0]) # uno cualquiera, pero tiene que ser del equipo contrario o algo asi
        scared_ghosts = state.get_agent_position(state.blue_team[0]) ##HOW TO DIFFERENTIATE BOTH GHOSTS
        ####################################
        
        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.get_agent_position(state.red_team[0])
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_x_2, next_y_2 = int(x + 2*dx), int(y + 2*dy)

        
        # FEATURE 1: NUMBER OF ACTIVE GHOST ONE STEP AWAY
        #count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        #FEATURE 2: NUMBER OF ACTIVE GHOST TWO STEPS AWAY
        # count the number of ghosts 1-step away
        features["#-of-ghosts-2-step-away"] = sum((next_x_2, next_y_2) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        #FEATURE 3: EAT FOOD IF NO ACTIVE GHOSTS NEAR
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        #FEATURE 4: DISTANCE TO CLOSEST FOOD
        dist_food = closestFood((next_x, next_y), food, walls)
        if dist_food is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["dist_closest_food"] = float(dist_food) / (walls.width * walls.height)
        features.divideAll(10.0)

        #FEATURE 5: POSITION IN THE BOARD
        #"x_pos": 0, "y_pos":0
        features["x_pos"] = next_x
        features["y_pos"] = next_y

        #FEATURE 6: DISTANCE TO CLOSEST GHOST
        #Implementar closestGhost
        dist_ghost = closestGhost((next_x, next_y), ghosts, walls)
        if dist_ghost is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["dist_closest_enemy"] = float(dist_ghost) / (walls.width * walls.height)
        features.divideAll(10.0)
        
        
        #FEATURE 7: DISTANCE TO CLOSEST SCARED GHOST
        #Implementar closestScaredGhost
        dist_scared_ghost = closestScaredGhost((next_x, next_y), scared_ghosts, walls)
        if dist_scared_ghost is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["dist_closest_vulnerable_enemy"] = float(dist_scared_ghost) / (walls.width * walls.height)
        features.divideAll(10.0)

        #FEATURE 8: FOOD_CARRYING
        #features["food_carrying"] = 


        # FEATURE 9: NUMBER OF SCARED GHOST ONE STEP AWAY
        #count the number of ghosts 1-step away
        features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in scared_ghosts)

        #FEATURE 10: NUMBER OF SCARED GHOST TWO STEPS AWAY
        # count the number of ghosts 1-step away
        features["#-of-scared-ghosts-2-step-away"] = sum((next_x_2, next_y_2) in Actions.get_legal_neighbors(g, walls) for g in scared_ghosts)


        #FEATURE 11: DISTANCE TO CAPSULES??


        return features
        
class ValueEstimationAgent(Agent):
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """
    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raiseNotDefined()

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return state.get_legal_actions()

    def observeTransition(self, state, action, nextState, deltaReward, episode):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward, episode)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward, self.episodesSoFar)
        return state

    def registerInitialState(self, state): # WE HAVE TO CALL FROM TRAIN.PY
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, state): # aqui podemos guardar los weights para cada action
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.get_score() - self.lastState.get_score()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward, self.episodesSoFar)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))
    
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.q_values = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        if self.q_values[(state, action)] == 0:
            return 0.0
        return self.q_values[(state, action)]
    

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
     
        if len(legalActions) == 0:
            return 0.0

        #Compute maximum Q-value for each possible action of a state, and return the value
        q_max = float('-inf')
        for a in self.getLegalActions(state): 
            q_max = max(q_max, self.getQValue(state, a))
        return q_max


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


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
          return action
        
        action = self.computeActionFromQValues(state) #Take best policy action
        
        return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        a = self.alpha
        g = self.discount

        #Update action-value function estimate
        self.q_values[(state, action)] = (1-a)*self.getQValue(state, action) + a*(reward + g*self.computeValueFromQValues(nextState))


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, index=0, time_for_computing=.1, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self._distributions = None
        self.index = index
        self.red = None
        self.agentsOnTeam = None
        self.distancer = None
        self.observationHistory = []
        self.timeForComputing = time_for_computing
        self.display = None
        QLearningAgent.__init__(self, **args)
        
    def register_initial_state(self, game_state):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        """
        self.red = game_state.is_on_red_team(self.index)
        self.register_team(self.get_team(game_state))
        self.distancer = distanceCalculator.Distancer(game_state.data.layout)

        # comment this out to forgo maze distance computation and use manhattan distances
        self.distancer.getMazeDistances()

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def register_team(self, agents_on_team):
        """
        Fills the self.agentsOnTeam field with a list of the
        indices of the agents on your team.
        """
        self.agentsOnTeam = agents_on_team

    def get_team(self, game_state):
        """
        Returns agent indices of your team. This is the list of the numbers
        of the agents (e.g., red might be the list of 1,3,5)
        """
        if self.red:
            return game_state.get_red_team_indices()
        else:
            return game_state.get_blue_team_indices()

    # def getAction(self, state):
        # """
        # Simply calls the getAction method of QLearningAgent and then
        # informs parent of action for Pacman.  Do not change or remove this
        # method.
        # """
        # self.observationHistory.append(state)
        # my_state = state.get_agent_state(self.index)
        # my_pos = my_state.get_position()
# 
        # if my_pos != util.nearestPoint(my_pos):
            # We're halfway from one position to the next
            # return QLearningAgent.getAction(self,state)
        # else:
            # return self.choose_action(state)
        
    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        self.observationHistory.append(state)
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
        
    def choose_action(self, game_state):
        """
        Override this method to make a good agent. It should return a legal action within
        the time limit (otherwise a random legal action will be chosen for you).
        """
        util.raiseNotDefined()