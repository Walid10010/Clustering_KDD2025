delete_set = set([])


def expandAroundGobalHotSpot(sigma, ind_to_name, name_to_dPoint):
    global delete_set
    delete_set = set([])
    new_neighbours = set([])
    sigma.neighbours = set(sigma.symmetric_knn)
    for neighbour in sigma.neighbours:
        dPoint = name_to_dPoint[ind_to_name[neighbour]]
        new_neighbours.update(disjunkt(sigma, dPoint, neighbour))

    expand_(sigma, ind_to_name, name_to_dPoint, new_neighbours)
    return sigma


def expand_(sigma, ind_to_name, name_to_dPoint, new_neighbours):
    global delete_set
    sigma.neighbours.update(new_neighbours)
    sigma.neighbours = sigma.neighbours.difference(delete_set)
    delete_set = set([])
    tmp_neighbours = set([])
    if len(new_neighbours) > 0:
        for neighbour in new_neighbours:
            dPoint = name_to_dPoint[ind_to_name[neighbour]]
            tmp_neighbours.update(disjunkt(sigma, dPoint, neighbour))
        expand_(sigma, ind_to_name, name_to_dPoint, tmp_neighbours)


def disjunkt(sigma, neighour, ind):
    disjunctSet = [_ for _ in neighour.neighbours if _ not in sigma.neighbours]
    disjunctSet_2 = [_ for _ in neighour.symmetric_knn if _ not in sigma.neighbours]


    rho_con = len(disjunctSet)/(len(neighour.neighbours) - len(disjunctSet)) > 1


    if rho_con  :
        delete_set.add(ind)
        return set([])
    else:
         return set(disjunctSet_2)
