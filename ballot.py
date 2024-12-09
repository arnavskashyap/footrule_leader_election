import random
from typing import List, Literal, Tuple
from enum import Enum
from collections import defaultdict
from itertools import combinations
from mallows import *


class Ballot:

    def __init__(self, num_candidates: int, num_good: int, num_bad: int,
                 good_prob: float, bad_prob: float):
        self.good_rankings = []
        self.bad_rankings = []
        self.num_candidates = num_candidates

        mallows_lphi = (1 - (good_prob / 100.0))
        # mallows_lphi = (200 - (2 * good_prob)) / 200.0
        # print(f"mallows lphi={mallows_lphi}")
        self.good_rankings = mallowsElection(num_good, self.num_candidates, 1,
                                             mallows_lphi * 2)[0]
        self.bad_rankings = mallowsElection(num_bad, self.num_candidates, 1,
                                            0.20, 1.0)[0]

        # for _ in range(num_good):
        #     ranking = self._generate_voter_ranking(good_prob / 100.0)
        #     self.good_rankings.append(ranking)

        # for _ in range(num_bad):
        #     ranking = self._generate_voter_ranking(bad_prob, False)
        #     self.bad_rankings.append(ranking)

        self.rankings = self.good_rankings + self.bad_rankings

    def _generate_voter_ranking(self,
                                prob_threshold: float,
                                good=True) -> List[int]:

        def flip_order(ranking, i, j):
            """Flip the order of elements i and j in the ranking."""
            idx_i, idx_j = ranking.index(i), ranking.index(j)
            ranking[idx_i], ranking[idx_j] = ranking[idx_j], ranking[idx_i]

        ideal_ranking = list(range(self.num_candidates))
        ranking = ideal_ranking[:]
        if not good:
            ranking.reverse()

        # For each pair (i, j) in the ranking
        for i in range(self.num_candidates):
            for j in range(i + 1, self.num_candidates):
                # Determine if the pairwise comparison should be correct
                if random.random() > prob_threshold:
                    # flip_order(ranking, ideal_ranking[i], ideal_ranking[j])
                    if (ranking.index(i)
                            < ranking.index(j)) == (ideal_ranking.index(i)
                                                    < ideal_ranking.index(j)):
                        flip_order(ranking, ideal_ranking[i], ideal_ranking[j])

        return ranking

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

    def eval_ballot_ideality(self,
                             split: Literal["good", "bad",
                                            "all"] = "all") -> int:
        # Pick the correct split
        rankings = self.rankings
        if split == "good":
            rankings = self.good_rankings
        if split == "bad":
            rankings = self.bad_rankings

        # Generate the set of pairwise comparisons for the ideal ranking
        ideal_ranking = list(range(self.num_candidates))
        ideal_comparisons = self._pairwise_comparisons(ideal_ranking)

        # # Evaluate each ranking
        # match_percentages = []
        # for ranking in rankings:
        #     ranking_comparisons = self._pairwise_comparisons(ranking)
        #     # Count matches
        #     matches = len(ideal_comparisons & ranking_comparisons)
        #     total_comparisons = len(ideal_comparisons)
        #     match_percentage = (matches / total_comparisons) * 100
        #     match_percentages.append(match_percentage)

        # # print(f"{split} match percentages={match_percentages}")
        # # return match_percentages
        # match_mean = sum(match_percentages) / len(match_percentages)
        # return match_mean

        ideal_positions = {item: i for i, item in enumerate(ideal_ranking)}
        matches_pct = []
        for ranking in rankings:

            # Total number of pairs
            same_pairs = 0
            # Iterate over all possible pairs of items
            total_pairs = 0
            for item1, item2 in combinations(ranking, 2):
                # Get their relative order in the given ranking
                rank_order = (ranking.index(item1) < ranking.index(item2))
                # Get their relative order in the ideal ranking
                ideal_order = (ideal_positions[item1] < ideal_positions[item2])

                # Compare the pairwise orders
                if rank_order == ideal_order:
                    same_pairs += 1
                total_pairs += 1

            # Calculate the percentage of agreement
            match_pct = (same_pairs /
                         total_pairs) * 100 if total_pairs > 0 else 0.0
            matches_pct.append(match_pct)
        return sum(matches_pct) / len(matches_pct)
