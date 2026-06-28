"""FigureCritic：从清晰度/标签完整性/配色/数据是否传达 4 维度评分。"""

SYSTEM = (
    "你是论文图表评审。基于给定图像和它的目的，给 0-10 评分（>=8 视为可用）。"
    "检查：标题/坐标轴/单位/图例是否齐全；配色是否专业；点线密度是否过载；"
    "信息是否与目的匹配。"
)


def build_prompt(purpose: str, image_info: str) -> str:
    return (
        f"# 图的目的\n{purpose}\n\n# 图的元信息\n{image_info}\n\n"
        f"请输出 JSON：{{\"score\": int, \"issues\": [str], \"suggestions\": [str], \"approved\": bool}}。"
    )
