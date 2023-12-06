from game import Actions

def closest_food(approxQagent, pos, food):
    '''
    Returns distance and position of the closest food found from the position of the agent and
    the matrix of food where if a position is true, means that there is a food in that position.
    '''
    dist_food = [float('inf'), (0, 0)]
    for i in range(food.width):
        for j in range(food.height):
            if food[i][j]:
                dist_food = min(dist_food, [approxQagent.get_maze_distance(pos, (i, j)), (i, j)], key=lambda x:x[0])
    return dist_food

def closest_capsule(approxQagent, pos, capsules):
    dist_capsule = float('inf')
    for c in capsules:
        dist_capsule = min(dist_capsule, approxQagent.get_maze_distance(pos, c))
    return dist_capsule

def count_food(food):
    food_count = 0
    for i in range(food.width):
        for j in range(food.height):
            if food[i][j]:
                food_count += 1
    return food_count

# UNUSED BY NOW
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