from collections import defaultdict
import csv

from ballot import Ballot
from swf import SWF, Ranker

TOTAL_BALLOTS = 50
TOTAL_VOTERS = 100
NUM_CANDIDATES = list(range(3, 9))
GOOD_PROBS = list(range(55, 95, 5))
# BAD_PROBS = [x / 100.0 for x in range(75, 95, 5)]
BAD_PROBS = [0.90]
RANKINGS_SPLITS = ["good", "bad", "all"]


def main():
    total_bad_voters = TOTAL_VOTERS // 3
    total_good_voters = TOTAL_VOTERS - total_bad_voters

    for k in NUM_CANDIDATES:
        for bad_prob in BAD_PROBS:
            for good_prob in GOOD_PROBS:
                print(
                    f"Running iteration with good_prob={good_prob}, bad_prob={bad_prob}, and k={k}"
                )
                ballots = []
                match_pcts = [[] for _ in RANKINGS_SPLITS]
                epoch_results = {"good_prob": good_prob, "k": k}

                for _ in range(TOTAL_BALLOTS):
                    ballot = Ballot(k, total_good_voters, total_bad_voters,
                                    good_prob, bad_prob)
                    ballots.append(ballot)

                    for split_idx, split in enumerate(RANKINGS_SPLITS):
                        match_mean = ballot.evaluate(split)
                        match_pcts[split_idx].append(match_mean)

                for split_idx, split in enumerate(RANKINGS_SPLITS):
                    test_match_mean = sum(match_pcts[split_idx]) / len(
                        match_pcts[split_idx])
                    epoch_results.update({split: test_match_mean})
                    print(f"{split} match mean={test_match_mean}")

                local_results = defaultdict(list)
                for ballot in ballots:
                    ranker = Ranker(ballot)

                    for swf in SWF:
                        ranker.apply_swf(swf)
                        result = ranker.evaluate_swf(swf)
                        local_results[swf].append(result)

                for swf in SWF:
                    swf_mean = sum(local_results[swf]) / len(ballots)
                    print(f"{swf} mean={swf_mean}")
                    epoch_results.update({swf: swf_mean})

                with open('results_fixed.csv', mode='a',
                          encoding='utf-8') as results_file:
                    writer = csv.writer(results_file)
                    writer.writerow(epoch_results.values())


if __name__ == "__main__":
    main()
