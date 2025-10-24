from __future__ import annotations
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.common.agent_rl_utils import MultiStagePlan

def multistage_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert samples, "empty batch"
    batch_meta = samples[0]["__batch_meta__"]

    data: Dict[str, Union[torch.Tensor, np.ndarray]] = {}

    tensor_keys = [k for k,v in samples[0].items() if isinstance(v, torch.Tensor)]
    for k in tensor_keys:
        data[k] = torch.stack([s[k] for s in samples], dim=0)

    array_keys  = [k for k,v in samples[0].items() if isinstance(v, np.ndarray)]
    for k in array_keys:
        data[k] = np.array([s[k] for s in samples])   # shape (B,)

    return {"data": data, "meta_info": batch_meta}

class StageExpandedRLHFDataset(Dataset):
    """Provide a virtual multistage view of the original RLHF dataset

    """
    
    def __init__(self, base_dataset: RLHFDataset, plan: MultiStagePlan):
        self.base = base_dataset
        self.plan = plan
        self._len = len(plan.base_idx_arr)  

    def __len__(self): 
        return self._len

    def __getitem__(self, virtual_idx: int) -> Dict[str, Any]:
        meta = self.plan.decode(virtual_idx)           
        row  = self.base[meta["base_idx"]]            

        out: Dict[str, Any] = {
            "input_ids":      row["input_ids"],
            "attention_mask": row["attention_mask"],
            "position_ids":   row["position_ids"],
        }

        out["index"] = np.array(row.get("index", meta["base_idx"]), dtype=np.int64)
        if row.get("raw_prompt_ids", None):
            out["raw_prompt_ids"] = np.array(row.get("raw_prompt_ids", None),dtype=object)
        if row.get("tools_kwargs", None):
            out["tools_kwargs"] = np.array(row.get("tools_kwargs", None), dtype=object)
        if row.get("interaction_kwargs", None):
            out["interaction_kwargs"] = np.array(row.get("interaction_kwargs", None), dtype=object)
        out["extra_info"] = np.array(row.get("extra_info", None), dtype=object)

        out["__batch_meta__"] = {
            "stage": meta["stage"],
            "group_id": meta["group_id"],
            "repeat_id": meta["repeat_id"],
            "is_group_final": meta["is_group_final"],
            "subtask_slot": meta["subtask_slot"],
        }
        return out