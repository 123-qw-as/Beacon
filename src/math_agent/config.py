"""集中配置。所有可调参数都从这里读，避免节点里散落硬编码。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL = os.getenv("MATH_AGENT_DEFAULT_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.getenv("MATH_AGENT_STRONG_MODEL", "gpt-4o")

# 节点 -> 模型 的路由表。便于 Critic 用强模型，常规节点用便宜模型。
MODEL_ROUTING = {
    "analyst": STRONG_MODEL,
    "modeler": STRONG_MODEL,
    "model_critic": STRONG_MODEL,
    "coder": DEFAULT_MODEL,
    "writer": STRONG_MODEL,
    "figure_critic": STRONG_MODEL,   # 多模态
    "paper_critic": STRONG_MODEL,
    "evaluation": STRONG_MODEL,
}

# 循环 / 重试上限
MAX_MODEL_ITERATIONS = 3       # basic -> improved -> final 之外的修正轮次
MAX_WRITER_ITERATIONS = 2      # paper_critic 未通过时 writer 最多重写次数
MAX_LLM_RETRIES = 2            # 单次 LLM 调用的结构化解析重试
MAX_CODE_RETRIES = 1           # coder / sensitivity 沙箱失败后再给一次机会，避免成本失控

# RAG（默认关闭；ingest 后通过环境变量启用）
RAG_ENABLED = os.getenv("MATH_AGENT_RAG_ENABLED", "0") == "1"
RAG_DB_PATH = os.getenv("MATH_AGENT_RAG_DB", str(PROJECT_ROOT / "runs" / "rag.sqlite"))
RAG_EMBEDDING_MODEL = os.getenv("MATH_AGENT_RAG_EMBED", "text-embedding-3-small")
RAG_EMBEDDING_DIM = int(os.getenv("MATH_AGENT_RAG_DIM", "1536"))
RAG_TOPK = int(os.getenv("MATH_AGENT_RAG_TOPK", "4"))
# 注入 prompt 的最大字符数（writer prompt 已接近 8k token 上限，控紧一点）
RAG_CTX_MAX_CHARS_ANALYST = 1500
RAG_CTX_MAX_CHARS_MODELER = 1500
RAG_CTX_MAX_CHARS_WRITER = 800
