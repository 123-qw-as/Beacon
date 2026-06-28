"""集中配置。所有可调参数都从这里读，避免节点里散落硬编码。"""
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("MATH_AGENT_DEFAULT_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.getenv("MATH_AGENT_STRONG_MODEL", "gpt-4o")

# 节点 -> 模型 的路由表。便于 Critic 用强模型，常规节点用便宜模型。
MODEL_ROUTING = {
    "analyst": STRONG_MODEL,
    "modeler": STRONG_MODEL,
    "model_critic": STRONG_MODEL,
    "coder": DEFAULT_MODEL,
    "writer": STRONG_MODEL,
}

# 循环 / 重试上限
MAX_MODEL_ITERATIONS = 3      # basic -> improved -> final 之外的修正轮次
MAX_LLM_RETRIES = 2           # 单次 LLM 调用的结构化解析重试
