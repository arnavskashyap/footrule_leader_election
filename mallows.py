import random


#Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps in a vote sampled from Mallows model
def calc_exp_swap_distance(m, phi):
    res = phi * m / (1 - phi)
    for j in range(1, m + 1):
        res = res + (j * (phi**j)) / ((phi**j) - 1)
    return res


#Given the number m of candidates and a absolute number of expected swaps exp_abs, this function returns a value of phi such that in a vote sampled from Mallows model with this parameter the expected number of swaps is exp_abs
def calc_phi(m, exp_abs):
    if exp_abs == m * (m - 1) / 2:
        return 1
    if exp_abs == 0:
        return 0
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = calc_exp_swap_distance(m, mid)
        #print('mid',mid)
        if abs(cur - exp_abs) < 1e-5:
            return mid
        if mid > 0.999999:
            return 1
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1


def calc_insertion_prob(i, phi):
    probas = (i + 1) * [0]
    for j in range(i + 1):
        probas[j] = pow(phi, (i + 1) - (j + 1))
    return probas


def weighted_choice(choices):
    total = 0
    for w in choices:
        total = total + w
    r = random.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(choices):
        if upto + w >= r:
            return i
        upto = upto + w
    assert False, "Shouldn't get here"


def mallows_vote(m, insertion_probas):
    vote = [0]
    for i in range(1, m):
        we = insertion_probas[:i + 1]
        #rounding issue
        if we[-1] == 0:
            we[-1] = 1
        index = weighted_choice(we)
        vote.insert(index, i)
    return vote


#Number n of voters, number m of candidadtes, number num_elections of elections to be returned
#relphi=True means that we use normalized phi, relphi=False means we use classical Mallows
#lphi\in [0,1] that gets normalized depending on the value of relphi
def mallows_election(n, m, num_elections, lphi, reverse=0):
    elections = []
    phi = calc_phi(m, lphi * (m * (m - 1)) / 4)
    #print(phi)
    insertion_probas = calc_insertion_prob(m, phi)
    #print(insertion_probas)
    base_order = list(range(m))
    for _ in range(num_elections):
        election = []
        for _ in range(n):
            current_order = base_order[:]  # Copy of the base order
            if reverse > 0:
                probability = random.random()
                if probability <= reverse:
                    current_order.reverse()

            vote = mallows_vote(m, insertion_probas)
            mapped_vote = [current_order[v] for v in vote]
            election.append(mapped_vote)
        elections.append(election)
    return elections
