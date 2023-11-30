import contest.distanceCalculator as distanceCalculator
from game import Directions, Actions, Agent

def closestEnemy(pos, enemies, game_state):
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
        # if we find an enemy at this location then exit
        for enemy in enemies:
            if game_state.get_agent_position(enemy) == (pos_x, pos_y):
                return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.get_legal_neighbors((pos_x, pos_y), game_state.get_walls())
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no ghost found
    return None

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