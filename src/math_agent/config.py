"""集中配置。所有可调参数都从这里读，避免节点里散落硬编码。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL = os.getenv("MATH_AGENT_DEFAULT_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.getenv("MATH_AGENT_STRONG_MODEL", "gpt-4o")
# Figure Pipeline 是唯一需要多模态（视觉）能力的节点，单独配置以便选用支持图像的模型。
# 默认回退到 STRONG_MODEL，保持向后兼容。
FIGURE_MODEL = os.getenv("MATH_AGENT_FIGURE_MODEL", STRONG_MODEL)

# 节点 -> 模型 的路由表。便于 Critic 用强模型，常规节点用便宜模型。
MODEL_ROUTING = {
    "analyst": STRONG_MODEL,
    "modeler": STRONG_MODEL,
    "model_critic": STRONG_MODEL,
    "coder": DEFAULT_MODEL,
    "writer": STRONG_MODEL,
    "figure_critic": FIGURE_MODEL,    # 多模态：图像质量评审
    "figure_analyst": FIGURE_MODEL,   # 多模态：图说生成
    "paper_critic": STRONG_MODEL,
    "evaluation": STRONG_MODEL,
}

# 循环 / 重试上限
MAX_MODEL_ITERATIONS = int(os.getenv("MATH_AGENT_MAX_MODEL_ITERATIONS", "3"))
MAX_WRITER_ITERATIONS = 2      # paper_critic 未通过时 writer 最多重写次数
MAX_LLM_RETRIES = 2            # 单次 LLM 调用的结构化解析重试
MAX_CODE_RETRIES = 1           # coder / sensitivity 沙箱失败后再给一次机会，避免成本失控
MAX_BLUEPRINT_ITERATIONS = 2   # blueprint critic 允许的评估次数（首次 + 一次 retry）
MAX_CODE_VERIFY_ITERATIONS = 2  # model_code_consistency 允许的评估次数（首次 + 一次 retry）

# LLM / embedding 调用的单次 HTTP 超时（秒）。
# 防止本地 router 半挂连接导致 httpx 无限阻塞：超时被 classify_exception
# 归为 LLMTransportError，自动纳入 llm_retry 现有 tenacity 指数退避重试
# （5 次 2/4/8/16/32s + jitter），最坏 ~5min 后抛 LLMError 干净退出，
# 而不是无限僵死。env 可调以适配更大的生成预算。
LLM_TIMEOUT = float(os.getenv("MATH_AGENT_LLM_TIMEOUT", "300"))
EMBED_TIMEOUT = float(os.getenv("MATH_AGENT_EMBED_TIMEOUT", "60"))

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
