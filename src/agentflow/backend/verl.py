import os
import re
import copy
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Protocol
from collections import defaultdict
from logging import Logger

from vllm import LLM, SamplingParams
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

from agentflow.core.interfaces import CanGenerate,SupportChatTemplate
from agentflow.utils.log_util import get_logger
from agentflow.utils.chat_template import is_chat_messages, safe_apply_chat_template, ChatTemplateDefaultsMixin, left_truncate_text_by_token, resolve_context_window_len

class VerlWg(Protocol):
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        ...
        
    
class VerlWgBackend(ChatTemplateDefaultsMixin, CanGenerate, SupportChatTemplate):
    """A backend using verl working group for RL rollout generation with verl
    """
    
    def __init__(
        self, 
        config: Dict[str,Any],
        wg: VerlWg,
        tokenizer: PreTrainedTokenizer,
        logger: Logger = None, 
        max_prompt_length: int = 8192,
        truncation: bool = True,
        **kwargs,
    ):
        super().__init__()
        ChatTemplateDefaultsMixin.__init__(self)
        self.config = config # 占位符，目前没有用到
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self.wg = wg
        self.tokenizer = tokenizer

        self.truncation = truncation
        self.max_prompt_length = max_prompt_length
    
    
    def apply_chat_template(self, messages: List[List[Dict[str,str]]], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            **additional_params) -> Union[str,Any]:
        merged = self._merge_for_call(additional_params)
        result, _ = safe_apply_chat_template(
            self.tokenizer,
            messages=messages,
            tokenize = tokenize,
            add_generation_prompt = add_generation_prompt,
            explicit_max_model_len=self.max_prompt_length,
            **merged
        )
        return result
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If chat messages are provided, it will automatically apply chat template
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw_output":Dataproto}
        """
        
        if is_chat_messages(prompts):
            raw_prompts = self.apply_chat_template(prompts)
        else:
            for i in range(len(prompts)):
                prompts[i]=left_truncate_text_by_token(self.tokenizer, str(prompts[i]), self.max_prompt_length)
                raw_prompts = prompts
                
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        model_inputs = self.tokenizer(
            raw_prompts,
            add_special_tokens=False,
            padding=True,                
            truncation=True,              
            max_length=self.max_prompt_length,
            return_attention_mask=True,
            return_tensors="pt"           
        )
        
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="left",
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        tensor_dict = {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "position_ids":position_ids,
        }
        input_proto = DataProto.from_single_dict(tensor_dict,meta_info={"eos_token_id":self.tokenizer.eos_token_id,"pad_token_id":self.tokenizer.pad_token_id})
        # TODO 参考verl的方法构造wg需要的输入
        # 1 verl/utils/dataset/rl_dataset.py 将一个message变成ids,计算mask position ids
        # 2 verl/trainer/ppo/ray_trainer.py 构造一个data proto
        # wg需要的tensor包括 input_ids attention_mask position_ids 
        # metainfo包括 meta_info.eos_token_id
        # non tensor batch可以没有
        
        output_proto = self.wg.generate_sequences(
            input_proto
        )
        
        responses_ids = output_proto.batch["responses"]
        response_texts = self.tokenizer.batch_decode(responses_ids)
        
        
        return response_texts, [{"raw_output": output_proto, "prompt":prompt} for prompt in prompts]
    
    def generate_sequences(self, prompts: DataProto, **kwargs):
        return self.wg.generate_sequences(prompts, **kwargs)
    


