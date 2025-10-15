from typing import Any, Dict, List, Union
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "..",".."))  
sys.path.insert(0, ROOT_DIR)     

from agentflow.utils.chat_template import ChatTemplateDefaultsMixin


class MockBackend(ChatTemplateDefaultsMixin):
    """一个最简单的后端，只打印最终传给模板的参数。"""

    def __init__(self, name: str):
        self.name = name
        super().__init__()

    def apply_chat_template(
        self,
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **additional_params: Any,
    ):
        merged = self._merge_for_call(additional_params)
        # —— 这里打印出“真正生效”的参数 —— #
        print(f"[{self.name}] defaults = {self.get_chat_template_defaults()}")
        print(f"[{self.name}] call_params = {additional_params}")
        print(f"[{self.name}] MERGED >>> {merged}")
        # 模拟渲染结果
        return {"rendered": True, "merged_params": merged, "messages": messages}


# ---------- 演示 ----------
if __name__ == "__main__":
    a = MockBackend("A")
    b = MockBackend("B")

    msgs = [{"role": "user", "content": "hello"}]

    print("\n-- 初始：两个实例默认都为空 --")
    a.apply_chat_template(msgs)  # MERGED {}
    b.apply_chat_template(msgs)  # MERGED {}

    print("\n-- 只给 A 设置默认 enable_thinking=True --")
    a.set_chat_template_defaults(enable_thinking=True, system_style="think")
    a.apply_chat_template(msgs)  # MERGED {'enable_thinking': True, 'system_style': 'think'}
    b.apply_chat_template(msgs)  # 仍然 MERGED {}

    print("\n-- A 本次调用覆盖默认值（按需关闭 think） --")
    a.apply_chat_template(msgs, enable_thinking=False)  # 本次参数优先 -> False
    # 验证默认没有被永久改写
    a.apply_chat_template(msgs)  # 又回到 True

    print("\n-- reset 之后 A 恢复为空 --")
    a.reset_chat_template_defaults()
    a.apply_chat_template(msgs)  # MERGED {}

    print("\n-- B 设置与调用混合测试 --")
    b.set_chat_template_defaults(template="direct", max_tokens=64)
    b.apply_chat_template(msgs, max_tokens=128)  # max_tokens 以本次为准：128
    
    a.reset_chat_template_defaults()
    a.apply_chat_template(msgs)
    
    with a.using_chat_template_defaults(tmp = "Test"):
        a.apply_chat_template(msgs)
    a.apply_chat_template(msgs)