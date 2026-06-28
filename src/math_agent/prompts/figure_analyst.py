"""FigureAnalyst：基于图像 + 数据上下文写一段专业图说。"""

SYSTEM = (
    "你是国赛论文图说撰写者。给定一张图与它对应的数据/参数信息，"
    "写一段 100-200 字的中文专业解读，覆盖：趋势、关键拐点、对模型结论的支撑。"
    "不要复述坐标轴标签。"
)


def build_prompt(purpose: str, context: str) -> str:
    return (
        f"# 图的目的\n{purpose}\n\n# 数据上下文\n{context}\n\n"
        f"请输出 JSON：{{\"analysis\": str}}。"
    )
