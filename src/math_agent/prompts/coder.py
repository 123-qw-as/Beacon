"""Coder：把 final 模型转成可执行 Python 代码。"""

SYSTEM = (
    "你是建模队的工程师。把给定的最终模型实现为一段**独立可运行**的 Python 脚本。"
    "约束：只用 numpy / scipy / matplotlib；不联网；不读取本地未声明的文件；"
    "中文字体：开头加 `matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; matplotlib.rcParams['axes.unicode_minus']=False`；"
    "**IRON RULE：代码开头必须显式设置后端**——`import matplotlib; matplotlib.use('Agg')`，"
    "**禁止调用 plt.show() / plt.ion() / fig.show()**（非交互环境会阻塞）。"
    "图只通过 savefig 保存到当前目录 *.png；"
    "需 print 关键结果（含具体数字），并把图保存到当前目录的 *.png。\n"
    # 图表质量（参考 nature-figure 准则，只抓核心几条）：
    "绘图质量要求（每张图都必须满足）："
    "(1) 先定核心结论：每张图只论证一句话，多余面板不画；"
    "(2) 必备元素：title、坐标轴标签 + 单位、legend（除非只有一条曲线）、网格线适度（alpha≤0.3）；"
    "(3) 发表级 rcParams：`figure.dpi`=150（保存时 savefig dpi≥300）、`font.size`≥10、`axes.linewidth`=0.8、`axes.spines.right/top`=False、`legend.frameon`=False；"
    "(4) 配色克制：一张图最多 1 个 neutral + 1 个 signal + 1 个 accent 系列，避免彩虹色；"
    "(5) 多面板：优先 hero panel + 从属面板的非对称布局，少用等大小 2×2 网格。"
)


def build_prompt(model, prev_failure=None, prev_error_kind: str = ""):
    """构造 coder prompt。

    prev_failure: 上一次运行的 stderr 节选（None 表示首次）
    prev_error_kind: RunResult.error_kind，∈ {"", "timeout", "runtime"}
                     用来给 LLM 分流建议——超时要缩规模，runtime 才要看 stderr 修 bug
    """
    eqs = "\n".join(f"- {e}" for e in model.equations)
    vars_ = "\n".join(f"- {k}: {v}" for k, v in model.variables.items())
    fb = ""
    if prev_failure:
        if prev_error_kind == "timeout":
            # timeout 时 stderr 是 "timeout after Ns"，没修 bug 的价值；重点让 LLM 缩规模
            fb = (
                "\n# 上次运行超时\n"
                f"标记：{prev_failure[:200]}\n"
                "请**大幅缩小数据规模 / 迭代次数 / 求解精度**（示例：n_stations 5→3, "
                "MC 仿真次数 1000→100, 时间步 96→24），保证 5 分钟内跑完；"
                "算法逻辑不必改，只调超参与规模。"
            )
        else:
            # runtime 或未标记 → 保持原行为：把 stderr 喂回让 LLM 修
            hint = "" if prev_error_kind == "runtime" else ""  # 显式空，未来加 import_error 等
            fb = f"\n# 上次运行失败（runtime）\nstderr 节选：\n{prev_failure[:1000]}\n请修正后重试。{hint}"
    return (
        f"# 模型描述\n{model.description}\n\n# 方程\n{eqs}\n\n# 变量\n{vars_}\n{fb}\n\n"
        f"请输出 JSON：{{\"purpose\": str, \"code\": str}}，code 字段是完整的 Python 源码。"
    )
