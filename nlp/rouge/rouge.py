"""ROUGE Implementation
See paper here: https://www.aclweb.org/anthology/W04-1013/
"""

import pprint
from typing import List, Dict
from collections import defaultdict


class RougeN:
    """Implementation of ROUGE-N
    """

    def __init__(self, n_gram: int, refs: List[List[str]], verbose=False):
        self.logger_fn = pprint.pprint if verbose else lambda _: None
        self.n_gram = n_gram

        self.reference_sets = self._jackknife_resample(
            [
                [self._get_n_gram_count(s) for s in r]
                for r in refs
            ]
        ) if len(refs) >= 2 else [refs]

    @staticmethod
    def _jackknife_resample(original_set: List[List]) -> List[List[List]]:
        resampled_sets = []
        for i, _ in enumerate(original_set):
            new_set = original_set.copy()
            new_set.pop(i)
            resampled_sets.append(new_set)
        return resampled_sets

    def _get_n_gram_count(self, doc: str) -> Dict:
        n_gram_count: Dict = defaultdict(int)
        token_list = doc.lower().split(" ")
        for n in range(1, self.n_gram + 1):
            end = len(token_list) - n + 1
            for ind, _ in enumerate(token_list[:end]):
                n_gram_count[" ".join(token_list[ind: ind + n])] += 1

        return n_gram_count

    def score(self, candidate_text: str) -> float:
        """Get GOUGE-N Score

        Args:
            candidate_text (str): Candidate text to be evaluated

        Returns:
            float: ROUGE-N score
        """
        candidate_n_gram_count = self._get_n_gram_count(candidate_text)
        self.logger_fn(candidate_n_gram_count)

        score_over_sets = []
        for set_ in self.reference_sets:

            score_over_refs = []
            for ref in set_:

                n_gram_match_count, n_gram_count = 0, 0
                for doc in ref:
                    for token in doc:
                        if token in candidate_n_gram_count:
                            n_gram_match_count += max(
                                candidate_n_gram_count[token],
                                doc[token]
                            )
                        n_gram_count += doc[token]

                score_over_refs.append(n_gram_match_count / n_gram_count)
            # maximum over references
            score_over_sets.append(max(score_over_refs))
        # average over sets
        return sum(score_over_sets) / len(score_over_sets)
