"""集中配置。所有可调参数都从这里读，避免节点里散落硬编码。"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL = os.getenv("MATH_AGENT_DEFAULT_MODEL", "gpt-4o-mini")
STRONG_MODEL = os.getenv("MATH_AGENT_STRONG_MODEL", "gpt-4o")
CODER_MODEL = os.getenv("MATH_AGENT_CODER_MODEL", DEFAULT_MODEL)


def _parse_model_list(value: str) -> tuple[str, ...]:
    """解析逗号分隔的备用模型列表，并保持声明顺序去重。"""
    result: list[str] = []
    for item in value.split(","):
        model = item.strip()
        if model and model not in result:
            result.append(model)
    return tuple(result)


# 9router 的连接回退只覆盖“同一 provider/model 下的多个连接”。当整个上游模型
# 返回 5xx、连接失败或超时时，Beacon 还需要像 Codex/Claude CLI 一样切换模型，
# 同时保留节点 checkpoint。留空即关闭跨模型故障转移。
LLM_FALLBACK_MODELS = _parse_model_list(
    os.getenv("MATH_AGENT_LLM_FALLBACK_MODELS", "")
)
# Figure Pipeline 是唯一需要多模态（视觉）能力的节点，单独配置以便选用支持图像的模型。
# 默认回退到 STRONG_MODEL，保持向后兼容。
FIGURE_MODEL = os.getenv("MATH_AGENT_FIGURE_MODEL", STRONG_MODEL)

# 节点 -> 模型 的路由表。便于 Critic 用强模型，常规节点用便宜模型。
MODEL_ROUTING = {
    "analyst": STRONG_MODEL,
    "modeler": STRONG_MODEL,
    "model_critic": STRONG_MODEL,
    "coder": CODER_MODEL,
    "writer": STRONG_MODEL,
    "figure_critic": FIGURE_MODEL,    # 多模态：图像质量评审
    "figure_analyst": FIGURE_MODEL,   # 多模态：图说生成
    "paper_critic": STRONG_MODEL,
    "evaluation": STRONG_MODEL,
}

# 循环 / 重试上限
MAX_MODEL_ITERATIONS = int(os.getenv("MATH_AGENT_MAX_MODEL_ITERATIONS", "3"))
MAX_WRITER_ITERATIONS = int(os.getenv("MATH_AGENT_MAX_WRITER_ITERATIONS", "3"))
MAX_LLM_RETRIES = 2            # 单次 LLM 调用的结构化解析重试
MAX_CODE_RETRIES = int(os.getenv("MATH_AGENT_MAX_CODE_RETRIES", "2"))
MAX_BLUEPRINT_ITERATIONS = 2   # blueprint critic 允许的评估次数（首次 + 一次 retry）
MAX_CODE_VERIFY_ITERATIONS = int(os.getenv("MATH_AGENT_MAX_CODE_VERIFY_ITERATIONS", "3"))

# LLM / embedding 调用的单次 HTTP 超时（秒）。
# 防止本地 router 半挂连接导致 httpx 无限阻塞：超时被 classify_exception
# 归为 LLMTransportError，自动纳入 llm_retry 现有 tenacity 指数退避重试
# （5 次 2/4/8/16/32s + jitter），最坏 ~5min 后抛 LLMError 干净退出，
# 而不是无限僵死。env 可调以适配更大的生成预算。
LLM_TIMEOUT = float(os.getenv("MATH_AGENT_LLM_TIMEOUT", "300"))
EMBED_TIMEOUT = float(os.getenv("MATH_AGENT_EMBED_TIMEOUT", "60"))

# ---- LLM 超时/重试重构（§8 配置兼容层）----
# 新变量优先；若仅存在旧 MATH_AGENT_LLM_TIMEOUT，按 §8.2 推导并输出弃用提示。
_NEW_ATTEMPT_TIMEOUT = os.getenv("MATH_AGENT_LLM_ATTEMPT_TIMEOUT")
_NEW_TOTAL_TIMEOUT = os.getenv("MATH_AGENT_LLM_TOTAL_TIMEOUT")
_LEGACY_LLM_TIMEOUT = os.getenv("MATH_AGENT_LLM_TIMEOUT")

if _NEW_ATTEMPT_TIMEOUT is None and _LEGACY_LLM_TIMEOUT is not None:
    # 兼容模式：旧变量解释为 standard.attempt_timeout
    _old = float(_LEGACY_LLM_TIMEOUT)
    _effective = min(_old, 600.0)
    print(
        f"[config] 弃用提示：MATH_AGENT_LLM_TIMEOUT={_old} 将在下一主版本删除。\n"
        f"  请改用 MATH_AGENT_LLM_ATTEMPT_TIMEOUT 和 MATH_AGENT_LLM_TOTAL_TIMEOUT。\n"
        f"  当前按兼容规则推导：standard.attempt={_effective}s（旧值上限 600s）。",
        flush=True,
    )

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
