import random
from typing import List, Literal
from itertools import combinations

from mallows import mallows_election


class Ballot:

    def __init__(self, num_candidates: int, num_good: int, num_bad: int,
                 good_prob: float, bad_prob: float):
        self.good_rankings = []
        self.bad_rankings = []
        self.num_candidates = num_candidates

        mallows_lphi = 1 - (good_prob / 100.0)
        self.good_rankings = mallows_election(num_good, self.num_candidates, 1,
                                              mallows_lphi * 2)[0]
        self.bad_rankings = mallows_election(num_bad, self.num_candidates, 1,
                                             (1 - bad_prob) * 2, 1.0)[0]
        self.rankings = self.good_rankings + self.bad_rankings

    def _generate_voter_ranking(self,
                                prob_threshold: float,
                                good=True) -> List[int]:
        ideal_ranking = list(range(self.num_candidates))
        ranking = ideal_ranking[:]
        if not good:
            ranking.reverse()

        # For each pair (i, j) in the ranking
        for i in range(self.num_candidates):
            for j in range(i + 1, self.num_candidates):
                # Determine if the pairwise comparison should be correct
                if random.random() > prob_threshold:
                    if (ranking.index(i)
                            < ranking.index(j)) == (ideal_ranking.index(i)
                                                    < ideal_ranking.index(j)):
                        idx_i, idx_j = ranking.index(
                            ideal_ranking[i]), ranking.index(ideal_ranking[j])
                        ranking[idx_i], ranking[idx_j] = ranking[
                            idx_j], ranking[idx_i]
        return ranking

    def evaluate(self, split: Literal["good", "bad", "all"] = "all") -> float:
        # Pick the correct split
        rankings = self.rankings
        if split == "good":
            rankings = self.good_rankings
        elif split == "bad":
            rankings = self.bad_rankings

        ideal_ranking = list(range(self.num_candidates))
        ideal_positions = {item: i for i, item in enumerate(ideal_ranking)}
        matches_pct = []

        for ranking in rankings:
            same_pairs = 0
            total_pairs = 0
            for item1, item2 in combinations(ranking, 2):
                rank_order = ranking.index(item1) < ranking.index(item2)
                ideal_order = ideal_positions[item1] < ideal_positions[item2]

                if rank_order == ideal_order:
                    same_pairs += 1
                total_pairs += 1

            # Calculate the percentage of agreement
            match_pct = (same_pairs /
                         total_pairs) * 100 if total_pairs > 0 else 0.0
            matches_pct.append(match_pct)
        return sum(matches_pct) / len(matches_pct)
