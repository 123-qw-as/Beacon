"""MathModelingState：整张图共享的状态，**同时充当 LangGraph 的 state schema**。

设计要点：
- 直接把 Pydantic 模型作为 LangGraph 的 state（langgraph>=0.2 支持）。
  list 字段用 `Annotated[..., operator.add]` 标记追加语义，节点返回增量字典时由
  reducer 自动合并；标量字段用"最新覆盖"语义。
- 节点签名统一为 `(state: MathModelingState) -> dict`，返回 **增量**。
- 不在 state 里放大文件，code/figure 路径只存路径字符串。
- 不再维护一份单独的 GraphState TypedDict，避免双重维护。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field

ModelStage = Literal["basic", "improved", "final"]


class Assumption(BaseModel):
    statement: str
    rationale: str = ""
    sensitivity_relevant: bool = False  # Plan B: sensitivity 节点消费该字段


class DerivationStep(BaseModel):
    """模型推导链中的一步（动机 → 数学陈述 → 结果）。"""
    title: str                       # "参数估计" / "约束推导" / "等价变换"
    motivation: str                  # 为何做这步
    statement: str                   # 数学陈述（含 inline LaTeX）
    result: str = ""                 # 推导结论


class ModelVersion(BaseModel):
    stage: ModelStage
    description: str
    equations: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    figure_purposes: list[str] = Field(default_factory=list)  # Plan D：modeler 建议要画的图
    derivation_steps: list[DerivationStep] = Field(default_factory=list)
    derivation_notes: str = ""  # Plan D：self-consistency gate 产出的问题标注


class Reference(BaseModel):
    """真实文献条目。来源：Semantic Scholar API 或静态库。"""
    id: str                        # Semantic Scholar paperId 或静态库自定义 id
    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str = ""
    year: int = 0
    doi: str = ""
    domains: list[str] = Field(default_factory=list)  # problem_domains 交集标记


class CodeArtifact(BaseModel):
    purpose: str
    code: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_paths: list[str] = Field(default_factory=list)  # 生成的图、数据等
    # ponytail: 不新建 BaselineResult 模型，复用 CodeArtifact + category 区分
    # "figure" = 主方案绘图, "baseline:no_schedule" / "baseline:simple_pred" / "baseline:greedy" = 对照方案
    category: str = "figure"


class CriticIssue(BaseModel):
    """结构化评审意见，section 字段限定到固定 enum。"""
    section: Literal[
        "abstract", "problem_restatement", "assumptions", "notation",
        "model_section", "solution", "sensitivity", "conclusion", "references", "general",
    ] = "general"
    problem: str


class CriticReport(BaseModel):
    target: Literal["analyst", "modeler", "coder", "writer", "paper"]
    score: int  # 0-10
    issues: list[CriticIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    approved: bool = False
    # stage 标记 critic 是针对哪个建模阶段产生的；analyst/coder/writer/paper 类型可为 None
    stage: Optional[ModelStage] = None


class SensitivityRun(BaseModel):
    parameter: str
    values: list[float]
    metric: str
    results: list[float]
    interpretation: str = ""
    figure_path: str | None = None


class FigureArtifact(BaseModel):
    path: str
    purpose: str
    caption: str = ""
    quality_score: int = 0           # 0-10，FigureCritic 打分
    quality_issues: list[str] = Field(default_factory=list)
    analysis: str = ""               # FigureAnalyst 产出的段落


class EvaluationReport(BaseModel):
    """对齐国赛四大标准 + 国一加分项。每项 0-10。"""
    assumption_reasonableness: int
    modeling_creativity: int
    result_correctness: int
    writing_clarity: int
    extra_depth: int                 # 加分项：敏感性/创新/分析深度
    overall: float                   # 加权总评
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class HumanDecision(BaseModel):
    approved: bool
    notes: str = ""


class PaperSections(BaseModel):
    abstract: str = ""
    problem_restatement: str = ""
    assumptions: str = ""
    notation: str = ""
    model_section: str = ""
    solution: str = ""
    sensitivity: str = ""            # Plan B 引入
    conclusion: str = ""
    references: str = ""
    keywords: str = ""               # 摘要末尾的关键词，逗号分隔（gmcm 模板用）


class MathModelingState(BaseModel):
    # 输入
    problem: str = ""  # 防御性默认值：checkpoint 重建容错（S2 bug，见 P2 评级）
    background: str = ""
    questions: list[str] = Field(default_factory=list)

    # 中间产物（list 字段都是 append 语义）
    assumptions: Annotated[list[Assumption], add] = Field(default_factory=list)
    model_versions: Annotated[list[ModelVersion], add] = Field(default_factory=list)
    code_artifacts: Annotated[list[CodeArtifact], add] = Field(default_factory=list)
    critic_reports: Annotated[list[CriticReport], add] = Field(default_factory=list)
    sensitivity_runs: Annotated[list[SensitivityRun], add] = Field(default_factory=list)
    figures: Annotated[list[FigureArtifact], add] = Field(default_factory=list)

    # 论文（覆盖语义）
    paper: PaperSections = Field(default_factory=PaperSections)

    # 评估（覆盖语义）
    evaluation: EvaluationReport | None = None
    human_decision: HumanDecision | None = None

    # 流程控制（覆盖语义）
    iteration: int = 0
    writer_iteration: int = 0           # 写作阶段的重试计数（paper_critic 闭环用）
    stage_target: ModelStage = "basic"  # 当前要产出的阶段
    problem_domains: list[str] = Field(default_factory=list)  # Plan D: analyst 输出，writer references 用
    errors: Annotated[list[str], add] = Field(default_factory=list)

    # writer 子流程状态（覆盖语义）。队列空 = 本轮写完。
    # ponytail: 队列即进度，不需要 completed_groups/current_group/pending_rewrite。
    writer_section_queue: list[str] = Field(default_factory=list)
    writer_outline_dump: dict = Field(default_factory=dict)   # WriterOutline.model_dump()
    writer_retrieved_context: str = ""                        # RAG 检索结果，prep 查一次 section 复用
    # table_assembler 产出的清洗/注入警告（覆盖语义；每次 table_assembler 运行整体替换）
    table_warnings: list[str] = Field(default_factory=list)

    # 输出
    output_dir: Optional[str] = None

    # LaTeX 模板选择 + 队伍信息（仅 gmcm 模板用到）
    latex_template: str = "default"   # "default" | "gmcm"
    school: Optional[str] = None
    team_id: Optional[str] = None
    members: Optional[str] = None     # "张三,李四,王五"

    # ---- 便利方法 ----
    def latest_model(self) -> ModelVersion | None:
        return self.model_versions[-1] if self.model_versions else None

    def latest_critic(self, target: str) -> CriticReport | None:
        for r in reversed(self.critic_reports):
            if r.target == target:
                return r
        return None

    def latest_critic_for_stage(self, target: str, stage: ModelStage) -> CriticReport | None:
        """按 (target, stage) 过滤，避免上一阶段未通过的反馈污染下一阶段。"""
        for r in reversed(self.critic_reports):
            if r.target == target and r.stage == stage:
                return r
        return None