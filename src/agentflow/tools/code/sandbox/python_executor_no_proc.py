from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import partial
from tqdm import tqdm

from .python_sandbox import (
    SandboxConfig, SandboxRuntime,
    ExecutionResult, _run_in_sandbox
)

from .execution_plan import ExecPlan

class PythonExecutorNoProc:
    def __init__(self,
                 config: Optional[SandboxConfig] = None,
                 max_workers: Optional[int] = None):
        self.config = config or SandboxConfig()
        self.max_workers = max_workers or 2
        self._headers: List[str] = []
        self._context: Dict[str, Any] = {}
        self._helpers: Dict[str, Any] = {}
        self._helper_code: List[str] = []

    def set_headers(self, headers: List[str]) -> None:
        self._headers = list(headers)

    def register_header(self, code: str) -> None:
        self._headers.append(code)

    def set_context(self, ctx: Dict[str, Any]) -> None:
        self._context.update(ctx)

    def inject_from_module(self, module: str,
                           names: Optional[List[str]] = None,
                           alias: Optional[Dict[str,str]] = None) -> None:
        helpers = SandboxRuntime.load_helpers_from_module(module, names=names, alias=alias)
        self._helpers.update(helpers)

    def inject_from_code(self, code: str,
                         export: Optional[List[str]] = None,
                         alias: Optional[Dict[str,str]] = None) -> None:
        self._helper_code.append(code)

    def inject_helpers(self, helpers: Dict[str, Any]) -> None:
        self._helpers.update(helpers)

    def _bound_run(self, plan: ExecPlan) -> ExecutionResult:
        return _run_in_sandbox(
            code=plan.code,
            capture_mode=plan.capture_mode,
            answer_symbol=plan.answer_symbol,
            answer_expr=plan.answer_expr,
            config=self.config,
            headers=self._headers + self._helper_code,
            context=self._context,
            helpers=self._helpers,
        )

    def run(self, plan: ExecPlan) -> ExecutionResult:
        try:
            with ThreadPoolExecutor(max_workers=1) as tp:
                fut = tp.submit(self._bound_run, plan)
                return fut.result(timeout=self.config.time_limit_s)
        except FuturesTimeout:
            return ExecutionResult(ok=False, result="", stdout="", error="Timeout", duration_s=self.config.time_limit_s)
        except Exception as e:
            return ExecutionResult(ok=False, result="", stdout="", error=f"{type(e).__name__}", duration_s=self.config.time_limit_s)

    def run_many(self, plans: List[ExecPlan], show_progress: bool=False) -> List[ExecutionResult]:
        if not plans:
            return []
        outs: List[Optional[ExecutionResult]] = [None] * len(plans)
        use_pbar = show_progress
        pbar = tqdm(total=len(plans), desc="Execute") if use_pbar else None
        try:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(plans))) as tp:
                futs: List[Tuple[int, Any]] = []
                for i, p in enumerate(plans):
                    futs.append((i, tp.submit(self._bound_run, p)))
                for i, fut in futs:
                    try:
                        outs[i] = fut.result(timeout=self.config.time_limit_s)
                    except FuturesTimeout:
                        outs[i] = ExecutionResult(ok=False, result="", stdout="", error="Timeout", duration_s=self.config.time_limit_s)
                    except Exception as e:
                        outs[i] = ExecutionResult(ok=False, result="", stdout="", error=f"Exception: {type(e).__name__}: {e}", duration_s=None)
                    if pbar: pbar.update(1)
        finally:
            if pbar: pbar.close()
        return [x if x is not None else ExecutionResult(ok=False, result="", stdout="", error="UnknownError", duration_s=None) for x in outs]