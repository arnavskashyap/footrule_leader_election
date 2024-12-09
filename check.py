import itertools
from ballot import Ballot
from typing import List
from scipy.optimize import linear_sum_assignment
import math
from collections import Counter


# Generate all permutations of the rankings for n=5 (ideal ranking is [0,1,2,3,4])
ideal_ranking = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
permutations = list(itertools.permutations(range(10)))

def kendall_tau_distance(r1, r2):
        dist = 0
        for i, item1 in enumerate(r1):
            for j, item2 in enumerate(r1):
                if i < j:
                    # Compare pairwise order in r1 and r2
                    dist += (r1.index(item1)
                             < r1.index(item2)) != (r2.index(item1)
                                                    < r2.index(item2))
        return dist

def spearman_footrule_old(num_candidates, rankings) -> List[int]:
        # Calculate the cost matrix based on Spearman's Footrule
        cost_matrix = [([0] * num_candidates)
                       for i in range(num_candidates)]
        for i in range(num_candidates):
            for j in range(num_candidates):
                cost = 0
                for ranking in rankings:
                    # Find the positions of items i and j in ranking r
                    pi = ranking.index(i)
                    cost += abs(pi - j)
                cost_matrix[i][j] = cost

        # c = self.num_candidates % 2
        # scaling = 2.0/(self.num_candidates**2 - c)
        # cost_matrix = cost_matrix * scaling

        # Apply the Hungarian algorithm to find the optimal matching
        _, col_ind = linear_sum_assignment(cost_matrix)

        # The result is the optimal aggregated ranking
        best_ranking = [item.item() for item in col_ind]

        return best_ranking
    
def spearman_footrule_distance_pair(reference_ranking: List[int],
                                         voter_ranking: List[int]):
        # Create a dictionary to map candidates to their positions in ranking2
        voter_position = {
            candidate: i
            for i, candidate in enumerate(voter_ranking)
        }

        # Calculate the Spearman Footrule distance
        distance = 0
        for i, candidate in enumerate(reference_ranking):
            # Find the position of the candidate in ranking1 and ranking2
            position1 = i
            position2 = voter_position[candidate]

            # Add the absolute difference of the positions
            distance += abs(position1 - position2)

        return distance
    
def f_most_distant_rankings(f: int, reference_ranking: List[int], rankings):
    # Calculate Kemeny distance from r for each ranking
    distances = []
    for voter_ranking in rankings:
        distance = spearman_footrule_distance_pair(
            reference_ranking, voter_ranking)
        distances.append((distance, voter_ranking))

    # Sort the distances in descending order to find the k most distant rankings
    distances.sort(reverse=True, key=lambda x: x[0])

    # Extract the k most distant rankings
    most_distant_rankings = [ranking for _, ranking in distances[:f]]

    return most_distant_rankings
    
def spearman_footrule_pruned(num_candidates, rankings) -> List[int]:
        # Calculate the cost matrix based on Spearman's Footrule
        # Find the f-most-distant rankings and remove from ballot
        best_dist = math.inf
        best_ranking = None
        for ranking in rankings:
            curr_ranking, curr_dist = spearman_footrule_prune(num_candidates, ranking, rankings)
            next_ranking, next_dist = spearman_footrule_prune(num_candidates, curr_ranking, rankings)
            better_ranking = curr_ranking
            better_dist = curr_dist
            if next_dist < curr_dist:
                better_ranking = curr_ranking
                better_dist = next_dist
            if better_dist < best_dist:
                best_dist = better_dist
                best_ranking = better_ranking
                
        return best_ranking

def spearman_footrule_prune(num_candidates, ranking, rankings):
    most_distant_rankings = f_most_distant_rankings(
                33, ranking, rankings)
    remove_counts = Counter(tuple(r) for r in most_distant_rankings)
    pruned_rankings = []
    removed = Counter()
    for original_ranking in rankings:
        original_ranking_tuple = tuple(original_ranking)
        if original_ranking_tuple in remove_counts and removed[original_ranking_tuple] < remove_counts[original_ranking_tuple]:
            removed[original_ranking_tuple] += 1
        else:
            pruned_rankings.append(original_ranking)
    
    cost_matrix = [([0] * num_candidates)
                for i in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(num_candidates):
            cost = 0
            for ranking in pruned_rankings:
                # Find the positions of items i and j in ranking r
                pi = ranking.index(i)
                cost += abs(pi - j)
            cost_matrix[i][j] = cost

    # c = self.num_candidates % 2
    # scaling = 2.0/(self.num_candidates**2 - c)
    # cost_matrix = cost_matrix * scaling

    # Apply the Hungarian algorithm to find the optimal matching
    _, col_ind = linear_sum_assignment(cost_matrix)

    # The result is the optimal aggregated ranking
    curr_ranking = [item.item() for item in col_ind]
    dist = sum(
        spearman_footrule_distance_pair(curr_ranking, ranking)
        for ranking in pruned_rankings)
    return curr_ranking, dist
        
    
TOTAL_BALLOTS = 50
TOTAL_VOTERS = 100
NUM_CANDIDATES = list(range(3, 9))
GOOD_PROBS = list(range(55, 95, 5))
# BAD_PROBS = [x / 100.0 for x in range(75, 95, 5)]
# GOOD_PROBS = [0.40]
BAD_PROBS = [0.90]

for num_candidates in NUM_CANDIDATES:
    for good_prob in GOOD_PROBS:
        print(f"Running iteration with {num_candidates} candidates and good_prob={good_prob}")
        ballot = Ballot(num_candidates, 67, 33, good_prob, 0.90)
        ideal_ranking = list(range(num_candidates))
        sf_result = spearman_footrule_old(num_candidates, ballot.rankings)
        print(f"SF={kendall_tau_distance(ideal_ranking, sf_result)}")
        psf_result = spearman_footrule_pruned(num_candidates, ballot.rankings)
        print(f"PSF={kendall_tau_distance(ideal_ranking, psf_result)}")
        
        
    