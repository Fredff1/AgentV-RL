import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)   

from typing import Dict, List

from agentflow.backend.openai import OpenaiBackend
from agentflow.agent.planner.llm_planner import LLMPlanner, JsonPlanParser
from agentflow.config import load_config
from agentflow.utils.json_util import JsonUtil


def generate_data(
    config: Dict,
    input_path: str,
    output_path: str,
):
    backend = OpenaiBackend(config)
    planner = LLMPlanner(backend)
    data: List[Dict] = JsonUtil.read_jsonlines(input_path)
    for block in data:
        pass


if __name__ == "__main__":
    CONFIG_PATH=""
    config = load_config(CONFIG_PATH)
    generate_data(config)