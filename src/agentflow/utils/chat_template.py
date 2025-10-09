from __future__ import annotations
from typing import Any, List, Dict, Optional, Tuple, Union

MessageDict = Dict[str, Any] 

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

def is_chat_messages(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return True
    head = obj[0]
    if isinstance(head,list) and len(head)>0:
        head = head[0]
    if not isinstance(head, dict):
        return False
    return "role" in head and "content" in head

def default_trans_messages(messages: Union[List[MessageDict],List[List[MessageDict]]]) -> Union[str,List[str]]:
    if len(messages)==0:
        return ""
    def _process_one(messages:List[MessageDict]):
        parts: List[str] = []
        for m in messages:
            name = f" ({m.get('name')})" if m.get("name") else ""
            parts.append(f"[{m.get('role')}{name}] {str(m.get('content',''))}")
        return "\n".join(parts)
    first = messages[0]
    if isinstance(first,dict):
        return _process_one(messages)
    else:
        results = [_process_one(msg) for msg in messages]
        return results
    
    
def _encode_len(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text)
    
def left_truncate_text_by_token(
    tokenizer: Any, text: str, keep_tokens: int, prefix: str = "[…]\n"
) -> str:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= keep_tokens:
            return text
        kept = ids[-keep_tokens:]
        return prefix + tokenizer.decode(kept, skip_special_tokens=False)
    except Exception:
        if len(text) <= keep_tokens:
            return text
        return prefix + text[-keep_tokens:]
    
def _flat_text(messages: List[MessageDict]) -> str:
    return "\n".join(f"[{m.get('role')}] {m.get('content','')}" for m in messages)

def _apply_to_ids(
    tokenizer: Any,
    messages: List[MessageDict],
    *,
    add_generation_prompt: bool,
    template_kwargs: Dict[str, Any],
) -> Tuple[List[int], bool]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs
            )
            return ids, True
        except Exception as e:
            print(e)
    txt = _flat_text(messages)
    try:
        return tokenizer.encode(txt, add_special_tokens=False), False
    except Exception:
        return [], False

def _truncate_ids_tail(ids: List[int], keep: int) -> List[int]:
    if keep is None or keep <= 0 or len(ids) <= keep:
        return ids
    return ids[-keep:]

def safe_apply_chat_template(
    tokenizer: Any,
    messages: List[Any],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    explicit_max_model_len: Optional[int] = None,
    generation_max_new_tokens: Optional[int] = None,
    safety_margin_tokens: int = 32,
    minimum_keep_tokens: int = 128,
    **kwargs,
) -> Tuple[Union[str, List[int], List[str], List[List[int]]], bool]:
    """Safely apply chat template. Automatically truncate tokens if explicit_max_model_len and generation_max_new_tokens are provided
    

    Args:
        tokenizer (Any): Tokenizer with chat template
        messages (List[Any]): A single or a list of chat message lists
        explicit_max_model_len (Optional[int], optional): max model length. Defaults to None.
        generation_max_new_tokens (Optional[int], optional): max genewation length. Defaults to None.
        safety_margin_tokens (int, optional): margin token for truncation. Defaults to 32.
        minimum_keep_tokens (int, optional): minium tokens to keep. Defaults to 128.

    Returns:
        Tuple[Union[str, List[int], List[str], List[List[int]]], bool]: result
    """
    template_kwargs = dict(kwargs) if kwargs else {}

    keep_tokens: Optional[int] = None
    if explicit_max_model_len is not None and generation_max_new_tokens is not None:
        keep_tokens = max(
            minimum_keep_tokens,
            int(explicit_max_model_len) - int(generation_max_new_tokens) - int(safety_margin_tokens)
        )

    is_batch = (
        isinstance(messages, list) and messages and isinstance(messages[0], list)
        and (not messages[0] or isinstance(messages[0][0], dict))
    )

    if not isinstance(messages, list) or len(messages) == 0:
        if is_batch:
            return ([] if tokenize else []), False
        return ([] if tokenize else ""), False

    def _process_one(msgs: List[MessageDict]) -> Tuple[Union[str, List[int]], bool]:
        ids, ok = _apply_to_ids(
            tokenizer, msgs,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=template_kwargs,
        )
        ids = _truncate_ids_tail(ids, keep_tokens) if keep_tokens is not None else ids
        if tokenize:
            return ids, ok
        try:
            text = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            text = _flat_text(msgs)
        return text, ok

    if is_batch:
        payloads: List[Union[str, List[int]]] = []
        all_ok = True
        for one in messages:
            payload, ok = _process_one(one)
            payloads.append(payload)
            all_ok = all_ok and ok
        return payloads, all_ok

    if isinstance(messages[0], dict):
        return _process_one(messages)
    else:
        raw = str(messages)
        try:
            ids = tokenizer.encode(raw, add_special_tokens=False)
        except Exception:
            ids = []
        ids = _truncate_ids_tail(ids, keep_tokens) if keep_tokens is not None else ids
        if tokenize:
            return ids, False
        try:
            return tokenizer.decode(ids, skip_special_tokens=False), False
        except Exception:
            return raw, False

def resolve_context_window_len(
    engine: Any,
    tokenizer: Any,
    explicit_max_model_len: Optional[int] = None,
    fallback: int = 8192,
) -> int:
    if isinstance(explicit_max_model_len, int) and explicit_max_model_len > 0:
        return explicit_max_model_len
    try:
        llm_engine = getattr(engine, "llm_engine", None)
        if llm_engine is not None:
            mc = getattr(llm_engine, "model_config", None)
            mlen = getattr(mc, "max_model_len", None)
            if isinstance(mlen, int) and mlen > 0:
                return mlen
    except Exception:
        pass

    for attr in ("model_config", "_model_config"):
        try:
            mc = getattr(engine, attr, None)
            mlen = getattr(mc, "max_model_len", None)
            if isinstance(mlen, int) and mlen > 0:
                return mlen
        except Exception:
            pass
    try:
        tlen = getattr(tokenizer, "model_max_length", None)
        if isinstance(tlen, int) and 0 < tlen < 10**7:
            return tlen
    except Exception:
        pass

    return fallback

def __safe_apply_chat_template(
    tokenizer: Any,
    messages: List[Any],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    **kwargs
) -> Tuple[Union[str,List[str]]:, bool]:
    """Wrapper function for apply chat template to avoid exceptions for tokenizers without chat template
    """
    has_method = hasattr(tokenizer, "apply_chat_template")
    if has_method:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            return text, True
        except Exception as e:  
            print(e)
            return default_trans_messages(messages), False
    return default_trans_messages(messages), False


class ChatTemplateDefaultsMixin:
    """为 backend 提供统一的 chat template 默认参数管理。"""

    def __init__(self, *args, **kwargs):
        self._chat_template_defaults = {}
        super().__init__(*args, **kwargs)

    def set_chat_template_defaults(self, **defaults: Any) -> None:
        self._chat_template_defaults.update(defaults)

    def get_chat_template_defaults(self) -> Dict[str, Any]:
        return dict(self._chat_template_defaults)

    def reset_chat_template_defaults(self) -> None:
        self._chat_template_defaults={}