# src/core/interfaces.py
from typing import List, Tuple, Dict, Any, Protocol, Sequence, Union, runtime_checkable, ContextManager, Iterable, overload, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field




@runtime_checkable
class SupportChatTemplate(Protocol):
    """Protocol for components that can render chat-style messages into a model-ready
    prompt using a chat template (and optionally tokenize the result).
    """
    def apply_chat_template(self, messages: Union[List[Dict[str,str]],List[List[Dict[str,str]]]], 
                            tokenize=False, add_generation_prompt=True, **additional_params
        ) -> Union[str,Any]:
        """
        Args:
            messages (Union[List[Dict[str,str]]): Standard chat messages or list of chat messages
            tokenize (bool, optional): Whether return tokenized output. Defaults to False.
            add_generation_prompt (bool, optional): Whether add generation prompts. Defaults to True.

        Returns:
            Union[str,Any]: The result with chat template applied
        """
        ...
        
    
    def set_chat_template_defaults(self, **defaults: Any) -> None: 
        """Set addistional parameters for auto chat template application
        For example, a llm backend may automatically use apply_chat_templates with the additional 
        parameters set here when the given input is chat message.
        """
        ...
    def get_chat_template_defaults(self) -> Dict[str, Any]: 
        """Get default chat template parameters

        Returns:
            Dict[str, Any]: Default params
        """
        ...
    def reset_chat_template_defaults(self) -> None: 
        """Clear default chat template parameters
        """
        ...
        
    def using_chat_template_defaults(self, **overrides: Any) -> ContextManager[None]: 
        """A simple context manager for using chat_template with temp parameters
        """
        ...

class CanGenerate(Protocol):
    """Protocol for a component that can generate text completions for the given prompts.
    Implementations may support both raw strings and chat-rendered strings as input.

    """
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str.
            extra (List[Dict], optional): Extra info dicts for generation. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated string sequences and metainfo of the generation.
        """
        ...

@runtime_checkable
class CanRMScores(Protocol):
    """Protocol for a component that can assign scalar scores to given sequences
    (e.g., reward model / preference model outputs). 
    """
    def score(self, sequences: Sequence[str], extra: List[Dict] = None, **kwargs) -> Tuple[List[float],List[Dict]]: 
        """Score the given sequences

        Args:
            sequences (Sequence[str]): The sequence list for scoring
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[float],List[Dict]]: Scores and any metainfo
        """
        ...

@runtime_checkable
class CanChoiceProbs(Protocol):
    """
    Protocol for a component that can compute **choice probabilities** for one or more textual
    prefixes. For each prefix `p_i` with a set of candidate continuations `choices[i] = [c_{i,0}, …]`,
    the implementation must estimate the conditional distribution
    `P(c_{i,j} | p_i)` over those candidates and return it as a probability vector.

    Requirements:
    - One probability vector is returned per prefix.
    - Within each vector, probabilities are ordered to correspond exactly to `choices[i]`.
    - Each vector should form a valid distribution (non-negative and summing to ~1 within numerical
      tolerance). If `choices[i]` is empty, return an empty list for that position.
    - Implementations may use any backend (e.g., LM log-likelihood of the continuation), but should
      treat each prefix independently and must not mix candidates across groups.

    Example:
        >>> backend.choice_probs(
        ...     prefixes=["Q1: 2+2=", "Q2: prime? 9 ->"],
        ...     choices=[["4", "5"], ["yes", "no"]],
        ... )
        [[0.95, 0.05], [0.02, 0.98]]
    """
    def choice_probs(self, prefixes: Sequence[str], choices: Sequence[Sequence[str]]) -> List[List[float]]:
        """Compute choice probabilities for each prefix.

        Args:
            prefixes: Sequence of textual prefixes `p_i`. Length N.
            choices:  Sequence of candidate lists, where `choices[i]` contains the textual options
                      to evaluate **conditioned on** `prefixes[i]`. Must have the same length as
                      `prefixes`. Each inner list may be empty (then return an empty vector).

        Returns:
            List[List[float]]:
            A list of N probability vectors. For each i:
            - `result[i]` has length `len(choices[i])`.
            - `result[i][j]` approximates `P(choices[i][j] | prefixes[i])`.
            - The entries in `result[i]` should sum to 1 (within numerical tolerance).
        """
        ...

@runtime_checkable
class CanContLogprobs(Protocol):
    def continuation_logprobs(self, prefixes: Sequence[str], continuations: Sequence[str]) -> List[float]: ...
    
    
