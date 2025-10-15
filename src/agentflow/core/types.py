from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Generic
from collections.abc import Sequence as SequenceABC
import json

TMeta = TypeVar("TMeta", bound=Dict[str, Any])


@dataclass(frozen=True, slots=True)
class SampleResult(Generic[TMeta]):
    text: str
    meta: TMeta


class _SampleView(SequenceABC, Generic[TMeta]):
    __slots__ = ("_texts", "_metas")
    def __init__(self, texts: Sequence[str], metas: Sequence[TMeta]):
        self._texts = texts
        self._metas = metas

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, i: int) -> SampleResult[TMeta]:
        return SampleResult(self._texts[i], self._metas[i])

    def __iter__(self) -> Iterator[SampleResult[TMeta]]:
        for t, m in zip(self._texts, self._metas):
            yield SampleResult(t, m)


@dataclass(frozen=True)
class GenerationResult(Generic[TMeta]):
    texts: Tuple[str, ...]
    metas: Tuple[TMeta, ...]
    batch_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.texts) != len(self.metas):
            raise ValueError(f"length mismatch: texts={len(self.texts)} metas={len(self.metas)}")

    # ---- 基础属性 ----
    @property
    def batch_size(self) -> int:
        return len(self.texts)

    @property
    def samples(self) -> _SampleView[TMeta]:
        return _SampleView(self.texts, self.metas)

    def update_batch_info(self, **updates: Any) -> GenerationResult[TMeta]:
        new_info = dict(self.batch_info)
        new_info.update(updates)
        return replace(self, batch_info=new_info)

    def select(self, indices: Sequence[int]) -> GenerationResult[TMeta]:
        texts = tuple(self.texts[i] for i in indices)
        metas = tuple(self.metas[i] for i in indices)
        return replace(self, texts=texts, metas=metas)

    def filter(self, pred: Callable[[SampleResult[TMeta], int], bool]) -> GenerationResult[TMeta]:
        kept_t, kept_m = [], []
        for i, (t, m) in enumerate(zip(self.texts, self.metas)):
            if pred(SampleResult(t, m), i):
                kept_t.append(t)
                kept_m.append(m)
        return replace(self, texts=tuple(kept_t), metas=tuple(kept_m))

    def to_legacy(self) -> Tuple[List[str], List[TMeta]]:
        return list(self.texts), list(self.metas)

    def to_dict(self) -> Dict[str, Any]:
        return {"texts": list(self.texts), "metas": list(self.metas), "batch_info": dict(self.batch_info)}

    @staticmethod
    def from_legacy(texts: Sequence[str], metas: Sequence[TMeta], batch_info: Optional[Dict[str, Any]] = None) -> GenerationResult[TMeta]:
        return GenerationResult(tuple(texts), tuple(metas), dict(batch_info or {}))

    def __repr__(self) -> str:
        n = self.batch_size
        t_preview = (self.texts[0][:60] + "…") if n else ""
        return f"GenerationResult(batch_size={n}, first_text={t_preview!r}, batch_info_keys={list(self.batch_info.keys())})"
