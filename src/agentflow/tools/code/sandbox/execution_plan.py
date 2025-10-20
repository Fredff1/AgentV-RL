from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ExecPlan:
    code: str
    capture_mode: str = "stdout"
    answer_symbol: Optional[str] = None
    answer_expr: Optional[str] = None
