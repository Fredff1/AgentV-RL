from torch.utils.data import Sampler
from typing import Iterator, List

from  verl.utils.common.agent_rl_utils import MultiStagePlan

class FlatBatchSampler(Sampler[List[int]]):
    
    def __init__(self, plan: MultiStagePlan):
        self.plan = plan
        
    def __iter__(self) -> Iterator[List[int]]:
        for b in self.plan.batches:
            yield b
            
    def __len__(self) -> int:
        return len(self.plan.batches)