"""Coder：把 final 模型转成可执行 Python 代码。"""

SYSTEM = (
    "你是建模队的工程师。把给定的最终模型实现为一段**独立可运行**的 Python 脚本。"
    "约束：只用 numpy / scipy / matplotlib；不联网；不读取本地未声明的文件；"
    "中文字体：开头加 `matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus']=False`；"
    "需 print 关键结果（含具体数字），并把图保存到当前目录的 *.png。\n"
    # 图表质量（参考 nature-figure 准则，只抓核心几条）：
    "绘图质量要求（每张图都必须满足）："
    "(1) 先定核心结论：每张图只论证一句话，多余面板不画；"
    "(2) 必备元素：title、坐标轴标签 + 单位、legend（除非只有一条曲线）、网格线适度（alpha≤0.3）；"
    "(3) 发表级 rcParams：`figure.dpi`=150（保存时 savefig dpi≥300）、`font.size`≥10、`axes.linewidth`=0.8、`axes.spines.right/top`=False、`legend.frameon`=False；"
    "(4) 配色克制：一张图最多 1 个 neutral + 1 个 signal + 1 个 accent 系列，避免彩虹色；"
    "(5) 多面板：优先 hero panel + 从属面板的非对称布局，少用等大小 2×2 网格。"
)


def build_prompt(model, prev_failure=None):
    eqs = "\n".join(f"- {e}" for e in model.equations)
    vars_ = "\n".join(f"- {k}: {v}" for k, v in model.variables.items())
    fb = ""
    if prev_failure:
        fb = f"\n# 上次运行失败\nstderr 节选：\n{prev_failure[:1000]}\n请修正后重试。"
    return (
        f"# 模型描述\n{model.description}\n\n# 方程\n{eqs}\n\n# 变量\n{vars_}\n{fb}\n\n"
        f"请输出 JSON：{{\"purpose\": str, \"code\": str}}，code 字段是完整的 Python 源码。"
    )
