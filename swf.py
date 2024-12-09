from collections import Counter, defaultdict
from itertools import permutations
import math
from typing import Dict, List, Literal, Tuple
from scipy.optimize import linear_sum_assignment

from ballot import Ballot


class Evaluator:

    def __init__(self, ballot: Ballot):
        self.rankings = ballot.rankings
        self.num_candidates = len(self.rankings[0])
        self.num_rankings = len(self.rankings)
        self.num_bad = len(ballot.bad_rankings)
        self.results = defaultdict(list)

    def apply_swf(
        self,
        method: Literal["PlacePlurality", "PairwiseComparison", "BordaCount",
                        "Kemeny", "PrunedKemeny", "SpearmanFootrule",
                        "IdealPrunedSpearmanFootrule",
                        "PracticalPrunedSpearmanFootrule", "all"] = "all"
    ) -> Dict[str, List[int]]:
        if method == "PlacePlurality":
            rank = self._place_plurality()
        elif method == "PairwiseComparison":
            rank = self._condorcet_pairwise_comparison()
        elif method == "BordaCount":
            rank = self._borda_count()
        elif method == "Kemeny":
            # rank = self._kemeny()
            rank = self._kemeny_young()
        elif method == "PrunedKemeny":
            rank = self._pruned_kemeny()
        elif method == "SpearmanFootrule":
            # rank = self._spearman_footrule()
            rank = self._spearman_footrule()
        elif method == "IdealPrunedSpearmanFootrule":
            rank = self._ideal_pruned_spearman_footrule()
        elif method == "PracticalPrunedSpearmanFootrule":
            rank = self._practical_spearman_footrule_pruned()

        if method == "all":
            self.results[method] = {
                "PlacePlurality":
                self.apply_swf("PlacePlurality"),
                "PairwiseComparison":
                self.apply_swf("PairwiseComparison"),
                "BordaCount":
                self.apply_swf("BordaCount"),
                "Kemeny":
                self.apply_swf("Kemeny"),
                "PrunedKemeny":
                self.apply_swf("PrunedKemeny"),
                # "SpearmanFootrule":
                # self.apply_swf("SpearmanFootrule"),
                # "IdealPrunedSpearmanFootrule":
                # self.apply_swf("IdealPrunedSpearmanFootrule"),
                "PracticalPrunedSpearmanFootrule":
                self.apply_swf("PracticalPrunedSpearmanFootrule")
            }
        else:
            self.results[method] = rank
        return self.results[method]

    def _pairwise_comparisons(self, ranking) -> List[Tuple[int, int]]:
        """
            Generate a set of pairwise comparisons for a ranking.
            Each comparison is a tuple (a, b) where a is ranked higher than b.
            """
        comparisons = set()
        for i, _ in enumerate(ranking):
            for j in range(i + 1, len(ranking)):
                comparisons.add((ranking[i], ranking[j]))
        return comparisons

    def eval_swf(
        self, method: Literal["PlacePlurality", "PairwiseComparison", "BordaCount",
                        "Kemeny", "PrunedKemeny", "SpearmanFootrule",
                        "IdealPrunedSpearmanFootrule",
                        "PracticalPrunedSpearmanFootrule"]
    ) -> int:
        ideal_ranking = list(range(self.num_candidates))
        return self._kendall_tau_distance(ideal_ranking, self.results[method])

        # Evaluate each ranking
        ranking_comparisons = self._pairwise_comparisons(self.results[method])
        matches = len(ideal_comparisons & ranking_comparisons)
        return len(ideal_comparisons) - matches
        # result = self._kemeny_distance_pair(ideal_ranking, self.results[method])
        # return result

    def _create_pairwise_matrix(self, rankings) -> List[List[int]]:
        # Initialize a pairwise comparison matrix
        pairwise_matrix = [[0] * self.num_candidates
                           for _ in range(self.num_candidates)]

        # Populate the pairwise matrix with pairwise comparisons
        for ranking in rankings:
            for i in range(self.num_candidates):
                for j in range(i + 1, self.num_candidates):
                    # If candidate i is ranked higher than candidate j
                    if ranking.index(i) < ranking.index(j):
                        pairwise_matrix[i][j] += 1
                    elif ranking.index(i) > ranking.index(j):
                        pairwise_matrix[j][i] += 1

        return pairwise_matrix

    def _place_plurality(self) -> List[int]:
        candidates = set(range(self.num_candidates))
        ranked_candidates = []

        # Loop over all rankings to count votes at each position for each candidate
        for position in range(self.num_candidates):
            position_votes = Counter()
            for ranking in self.rankings:
                voted_candidate = ranking[position]
                position_votes[voted_candidate] += 1

            # Determine the candidate that best fits each position
            unplaced = candidates - set(ranked_candidates)
            filtered_votes = {
                candidate: position_votes[candidate]
                for candidate in unplaced
            }
            winner = max(filtered_votes, key=filtered_votes.get)
            ranked_candidates.append(winner)

        return ranked_candidates

    def _condorcet_pairwise_comparison(self) -> List[int]:
        full_pairwise_matrix = self._create_pairwise_matrix(self.rankings)
        victories = [0] * self.num_candidates
        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                if full_pairwise_matrix[i][j] > full_pairwise_matrix[j][i]:
                    victories[i] += 1

        # Rank candidates based on their pairwise victories
        best_ranking = sorted(range(self.num_candidates),
                              key=lambda x: victories[x],
                              reverse=True)

        return best_ranking

    def _borda_count(self) -> List[int]:
        borda_scores = [0] * self.num_candidates

        # Step 1: Calculate the Borda score for each candidate based on the voters' rankings
        for ranking in self.rankings:
            for i, candidate in enumerate(ranking):
                # Assign the points based on position in the ranking
                # (highest position gets self.num_candidates-1 points, next gets self.num_candidates-2, etc.)
                borda_scores[candidate] += (self.num_candidates - 1 - i)

        # Step 2: Rank candidates based on the total Borda scores in descending order
        best_ranking = sorted(range(self.num_candidates),
                              key=lambda x: borda_scores[x],
                              reverse=True)

        return best_ranking

    def _get_all_rankings(self) -> List[List[int]]:
        # Generate all possible rankings (permutations of candidates)
        return list(permutations(range(self.num_candidates)))

    def _kemeny_distance(self, ranking: List[int],
                         rankings: List[List[int]]) -> int:
        pairwise_matrix = self._create_pairwise_matrix(rankings)

        distance = 0
        for i in range(self.num_candidates):
            for j in range(i + 1, self.num_candidates):
                if ranking.index(i) < ranking.index(j):
                    distance += pairwise_matrix[i][j]
                else:
                    distance += pairwise_matrix[j][i]
        return distance

    def _kemeny_distance_pair(self, reference_ranking: List[int],
                              voter_ranking: List[int]) -> int:
        distance = 0
        # Count pairwise disagreements
        for i in range(self.num_candidates):
            for j in range(i + 1, self.num_candidates):
                # If ranking1[i] ranks higher than ranking1[j], and ranking2 does the opposite, it's a disagreement
                if (reference_ranking.index(i)
                        < reference_ranking.index(j)) != (
                            voter_ranking.index(i) < voter_ranking.index(j)):
                    distance += 1

        return distance

    def _kemeny(self):
        all_rankings = self._get_all_rankings()

        # Find the ranking with the minimum Kemeny distance
        best_ranking = None
        min_distance = math.inf

        for ranking in all_rankings:
            distance = self._kemeny_distance(ranking, self.rankings)
            if distance < min_distance:
                min_distance = distance
                best_ranking = ranking

        return best_ranking

    def _spearman_footrule_distance_pair(self, reference_ranking: List[int],
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

    def _spearman_footrule_old(self) -> List[int]:
        # Calculate the cost matrix based on Spearman's Footrule
        cost_matrix = [([0] * self.num_candidates)
                       for i in range(self.num_candidates)]
        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                cost = 0
                for ranking in self.rankings:
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

    def _f_most_distant_rankings(
            self, f: int, reference_ranking: List[int],
            distance_method: Literal["kemeny", "spearman-footrule"]):
        # Calculate Kemeny distance from r for each ranking
        distances = []
        for voter_ranking in self.rankings:
            distance = self._spearman_footrule_distance_pair(
                reference_ranking, voter_ranking)
            if distance_method == "kemeny":
                distance = self._kendall_tau_distance(reference_ranking,
                                                      voter_ranking)
            distances.append((distance, voter_ranking))

        # Sort the distances in descending order to find the k most distant rankings
        distances.sort(reverse=True, key=lambda x: x[0])

        # Extract the k most distant rankings
        most_distant_rankings = [ranking for _, ranking in distances[:f]]

        return most_distant_rankings

    def _pruned_kemeny(self):
        # Generate all possible rankings (permutations of candidates)
        all_perms = self._get_all_rankings()

        # Find the ranking with the minimum Kemeny distance
        best_ranking = None
        min_distance = math.inf

        for perm in all_perms:
            # Find the f-most-distant rankings and remove from ballot
            most_distant_rankings = self._f_most_distant_rankings(
                self.num_bad, perm, "kemeny")
            remove_counts = Counter(tuple(r) for r in most_distant_rankings)
            pruned_rankings = []
            removed = Counter()
            for original_ranking in self.rankings:
                original_ranking_tuple = tuple(original_ranking)
                if original_ranking_tuple in remove_counts and removed[
                        original_ranking_tuple] < remove_counts[
                            original_ranking_tuple]:
                    removed[original_ranking_tuple] += 1
                else:
                    pruned_rankings.append(original_ranking)

            total_distance = sum(
                self._kendall_tau_distance(perm, ranking)
                for ranking in pruned_rankings)
            if total_distance < min_distance:
                min_distance = total_distance
                best_ranking = perm

        return best_ranking

    # Function to calculate Kendall-Tau distance for two rankings
    def _kendall_tau_distance(self, r1, r2):
        dist = 0
        for i, item1 in enumerate(r1):
            for j, item2 in enumerate(r1):
                if i < j:
                    # Compare pairwise order in r1 and r2
                    dist += (r1.index(item1)
                             < r1.index(item2)) != (r2.index(item1)
                                                    < r2.index(item2))
        return dist

    def _kemeny_young(self):
        """
        Implements the Kemeny-Young method for rank aggregation.

        Parameters:
            self._rankings (list of lists): A list of self._rankings, where each ranking is a list of integers or items.

        Returns:
            tuple: The consensus ranking and the associated total Kendall-Tau distance.
        """
        # Get the set of all unique items across all self._rankings
        items = set(item for ranking in self.rankings for item in ranking)
        n = len(items)

        # Convert the set of items into a sorted list
        items = sorted(items)

        # Generate all possible permutations of the items (potential consensus self.rankings)
        all_perms = list(permutations(range(self.num_candidates)))

        # Generate all possible rankings (permutations of candidates)
        all_perms = self._get_all_rankings()

        # Find the consensus ranking with minimum total Kendall-Tau distance
        min_distance = float('inf')
        best_ranking = None

        for perm in all_perms:
            total_distance = sum(
                self._kendall_tau_distance(perm, ranking)
                for ranking in self.rankings)
            if total_distance < min_distance:
                min_distance = total_distance
                best_ranking = perm

        return best_ranking

    def _spearman_footrule_distance(self, ranking1, ranking2):
        """
        Calculate Spearman's Footrule distance between two rankings.

        Parameters:
        - ranking1, ranking2: Lists of rankings, where each is a list of candidate indices ordered from most to least preferred.

        Returns:
        - The Spearman's Footrule distance between the two rankings.
        """
        m = len(ranking1)  # Number of candidates
        # Create a dictionary for the position of each candidate in ranking1 and ranking2
        position1 = {candidate: i for i, candidate in enumerate(ranking1)}
        position2 = {candidate: i for i, candidate in enumerate(ranking2)}

        # Calculate the sum of absolute differences in ranks for each candidate
        distance = 0
        for candidate in position1:
            distance += abs(position1[candidate] - position2[candidate])

        return distance

    def _spearman_footrule(self):
        n = len(self.rankings)  # Number of voters

        # # Initialize a list to store total Spearman distances for each candidate ranking
        # total_distances = [0] * n

        # # Calculate the Spearman Footrule distance of each ranking to all others
        # for i in range(n):
        #     total_distances[i] = sum(self._spearman_footrule_distance(self.rankings[i], self.rankings[j]) for j in range(n) if i != j)

        # Select the ranking with the minimal total Spearman Footrule distance
        # best_ranking_index = total_distances.index(min(total_distances))
        # return self.rankings[best_ranking_index]

        # Changed
        all_perms = self._get_all_rankings()
        min_distance = float('inf')
        best_ranking = None

        for perm in all_perms:
            total_distance = sum(
                self._spearman_footrule_distance(perm, ranking)
                for ranking in self.rankings)
            if total_distance < min_distance:
                min_distance = total_distance
                best_ranking = perm

        return best_ranking

    def _ideal_pruned_spearman_footrule(self):
        # Generate all possible rankings (permutations of candidates)
        all_perms = self._get_all_rankings()

        # Find the ranking with the minimum Kemeny distance
        best_ranking = None
        min_distance = math.inf

        for perm in all_perms:
            # Find the f-most-distant rankings and remove from ballot
            most_distant_rankings = self._f_most_distant_rankings(
                self.num_bad, perm, "spearman-footrule")
            remove_counts = Counter(tuple(r) for r in most_distant_rankings)
            pruned_rankings = []
            removed = Counter()
            for original_ranking in self.rankings:
                original_ranking_tuple = tuple(original_ranking)
                if original_ranking_tuple in remove_counts and removed[
                        original_ranking_tuple] < remove_counts[
                            original_ranking_tuple]:
                    removed[original_ranking_tuple] += 1
                else:
                    pruned_rankings.append(original_ranking)

            total_distance = sum(
                self._kendall_tau_distance(perm, ranking)
                for ranking in pruned_rankings)
            if total_distance < min_distance:
                min_distance = total_distance
                best_ranking = perm

        return best_ranking

    def _practical_spearman_footrule_pruned(self) -> List[int]:
        # Calculate the cost matrix based on Spearman's Footrule
        # Find the f-most-distant rankings and remove from ballot
        best_dist = math.inf
        best_ranking = None
        for ranking in self.rankings:
            curr_ranking, curr_dist = self._spearman_footrule_prune(ranking)
            next_ranking, next_dist = self._spearman_footrule_prune(
                curr_ranking)
            better_ranking = curr_ranking
            better_dist = curr_dist
            if next_dist < curr_dist:
                better_ranking = next_ranking
                better_dist = next_dist
            if better_dist < best_dist:
                best_dist = better_dist
                best_ranking = better_ranking
        return best_ranking

    def _spearman_footrule_prune(self, ranking):
        most_distant_rankings = self._f_most_distant_rankings(
            self.num_bad, ranking, "spearman-footrule")
        remove_counts = Counter(tuple(r) for r in most_distant_rankings)
        pruned_rankings = []
        removed = Counter()
        for original_ranking in self.rankings:
            original_ranking_tuple = tuple(original_ranking)
            if original_ranking_tuple in remove_counts and removed[
                    original_ranking_tuple] < remove_counts[
                        original_ranking_tuple]:
                removed[original_ranking_tuple] += 1
            else:
                pruned_rankings.append(original_ranking)

        cost_matrix = [([0] * self.num_candidates)
                       for i in range(self.num_candidates)]
        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
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
            self._spearman_footrule_distance_pair(curr_ranking, ranking)
            for ranking in pruned_rankings)
        return curr_ranking, dist
