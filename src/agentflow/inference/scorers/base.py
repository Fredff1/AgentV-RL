from typing import List, Tuple, Dict, Any, Sequence, Callable, Optional

from tqdm import tqdm

from agentflow.core.interfaces import CanGenerate, CanRMScores, CanChoiceProbs, SupportChatTemplate
from agentflow.utils.tag_util import find_tags
from agentflow.utils.vllm import SupportVllm, free_vllm_mem


class SupportLogitsScore(CanGenerate,CanChoiceProbs):
    ...

class BoolLogitsScorer(CanRMScores):
    

    def __init__(
        self,  
        prob_calculator: CanChoiceProbs,
        *,
        prob_bs: int = 1,
        choice_labels: Sequence[str] = ("true", "false"), 
        eps: float = 1e-15,   
    ):
        super().__init__()
        self.prob_calculator = prob_calculator
        self.prob_bs = max(1, int(prob_bs))
        self.choice_labels = tuple(choice_labels)
        self.eps = eps
        
        
    def _chunk(self, xs: Sequence[Any], n: int):
        for i in range(0, len(xs), n):
            yield i, xs[i:i+n]

    def _batched_choice_probs(
        self,
        prefixes: Sequence[str],
        labels: Sequence[str],
        **kw
    ) -> List[List[float]]:
        """分批调用 prob_calculator.choice_probs，保持顺序不变。"""
        all_probs: List[List[float]] = []
        for _, pref_chunk in self._chunk(prefixes, self.prob_bs):
            choices_chunk = [list(labels) for _ in range(len(pref_chunk))]
            try:
                probs_chunk = self.prob_calculator.choice_probs(pref_chunk, choices_chunk, **kw)
            except Exception:
                probs_chunk = [[0.0 for _ in labels] for _ in range(len(pref_chunk))]
            probs_chunk = [list(map(float, p)) for p in probs_chunk]
            all_probs.extend(probs_chunk)
        return all_probs
    
    def score(self, sequences: Sequence[str], extra: List[Dict] = None, **kwargs) -> Tuple[List[float],List[Dict]]: 
       
        prefixes = []
        invalid_idxs = []
        for idx, seq in enumerate(sequences):
            answer_tags = find_tags(seq,["answer"])
            if answer_tags:
                target = answer_tags[-1]
                prefix_text = seq[:target.start]+"<answer>"
            else:
                prefix_text = "Mock"
                invalid_idxs.append(idx)
            prefixes.append(prefix_text)
        probs = self._batched_choice_probs(prefixes,self.choice_labels)
        results: List[float] = []
        for idx, prob in enumerate(probs):
            if idx in invalid_idxs:
                results.append(-1)
            else:
                prob_true = prob[0]
                prob_false = prob[1]
                results.append((prob_true)/max((prob_true+prob_false),1e-15))
                
        final_metas = [{} for _ in range(len(sequences))]

        return results, final_metas
        
        
        