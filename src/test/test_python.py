# tests/test_python_execution_tool.py

import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from agentflow.tools.code.python_execution import PythonExecutionTool
from agentflow.tools.base import ToolCallRequest

def test_python_tool_print_basic():
    tool = PythonExecutionTool(timeout_length=5,use_proc=False)
    request = ToolCallRequest(
        index=0,
        name="search",
        content="print(1+2+3)"
    )
    result = tool.run_one(request)
    print(result)


def test_python_tool_forbidden_calls():
    tool = PythonExecutionTool(timeout_length=3)
    request = ToolCallRequest(
        index=0,
        name="python",
        content="input('x')"
    )
    result = tool.run_one(request)
    print(result)
    

def test_python_tool_timeout():
    tool = PythonExecutionTool(timeout_length=3)
    request = ToolCallRequest(
        index=0,
        name="python",
        content="import time\nprint('start')\ntime.sleep(20)\nprint('end')"
    )
    result = tool.run_one(request)
    print(result)

def test_python_tool_batch():
    tool = PythonExecutionTool(timeout_length=5)
    tool.register_helpers_from_code(
    """
import math
import numpy as np
def calculate(a,b):
    return a+b
def foo():
    return np.sum([1,2,3])
"""
    )
    requests = [
        ToolCallRequest(
        index=0,
        name="python",
        content="print(calculate(10,20))"
    ),
        ToolCallRequest(
        index=0,
        name="python",
        content="print(foo())"
    )
    ]
    results = tool.run_batch(requests)
    print(results)
    
if __name__ == "__main__":
    # test_python_tool_print_basic()
    test_python_tool_batch()
    # test_python_tool_forbidden_calls()
    # test_python_tool_timeout()
