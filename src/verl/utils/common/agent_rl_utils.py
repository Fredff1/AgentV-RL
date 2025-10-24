from typing import List, Tuple

Stage = str 

class MultiStagePlan:
    """
    扁平规划器：
    - 构建一批“虚拟样本”的查表数组（parallel arrays）
    - 按组 + 按阶段顺序生成 batches（每个 batch 是一串虚拟索引）
    """

    def __init__(
        self,
        base_len: int,
        base_batch_size: int,   
        schedule: List[Tuple[Stage,int]], 
        permuted_base_indices: List[int]   
    ):
        """Initialize a multistage plan.

        Args:
            base_len (int): original sample nums.
            base_batch_size (int): train batch size.
            schedule: stage schedule info, format [("plan",1),("subtask",1),("review",1)].
            permuted_base_indices: shuffled rl dataset indicies
        """
        self.base_len = base_len
        self.bs = base_batch_size
        self.schedule = schedule

        self.groups: List[List[int]] = [
            permuted_base_indices[i:i+self.bs]
            for i in range(0, base_len, self.bs)
            if len(permuted_base_indices[i:i+self.bs]) == self.bs
        ]

        self.base_idx_arr:   List[int]   = []
        self.stage_arr:      List[Stage] = []
        self.repeat_arr:     List[int]   = []
        self.group_arr:      List[int]   = []
        self.is_final_arr:   List[bool]  = []
        self.subslot_arr:    List[int]   = []

        self.batches: List[List[int]] = []

        vid = 0
        for g, group in enumerate(self.groups):
            for si, (stage, cnt) in enumerate(schedule):
                for r in range(cnt):
                    is_final = (si == len(schedule) - 1) and (r == cnt - 1)
                    subslot = r if stage == "subtask" else -1

                    this_batch: List[int] = []
                    for base_idx in group:
                        self.base_idx_arr.append(base_idx)
                        self.stage_arr.append(stage)
                        self.repeat_arr.append(r)
                        self.group_arr.append(g)
                        self.is_final_arr.append(is_final)
                        self.subslot_arr.append(subslot)

                        this_batch.append(vid)
                        vid += 1

                    self.batches.append(this_batch)

    def decode(self, virtual_idx: int):
        return {
            "base_idx":     self.base_idx_arr[virtual_idx],
            "stage":        self.stage_arr[virtual_idx],
            "repeat_id":    self.repeat_arr[virtual_idx],
            "group_id":     self.group_arr[virtual_idx],
            "is_group_final": self.is_final_arr[virtual_idx],
            "subtask_slot": self.subslot_arr[virtual_idx],
        }
