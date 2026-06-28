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


class ModelVersion(BaseModel):
    stage: ModelStage
    description: str
    equations: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class CodeArtifact(BaseModel):
    purpose: str
    code: str
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    artifact_paths: list[str] = Field(default_factory=list)  # 生成的图、数据等


class CriticReport(BaseModel):
    target: Literal["analyst", "modeler", "coder", "writer", "paper"]
    score: int  # 0-10
    issues: list[str] = Field(default_factory=list)
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


class MathModelingState(BaseModel):
    # 输入
    problem: str
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
    stage_target: ModelStage = "basic"  # 当前要产出的阶段
    errors: Annotated[list[str], add] = Field(default_factory=list)

    # 输出
    output_dir: Optional[str] = None

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