"""Coder：把 final 模型转成可执行 Python 代码。"""

SYSTEM = (
    "你是建模队的工程师。把给定的最终模型实现为一段**独立可运行**的 Python 脚本。"
    "约束：只用 numpy / scipy / matplotlib；不联网；不读取本地未声明的文件；"
    "需 print 关键结果，并把图保存到当前目录的 *.png。"
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
