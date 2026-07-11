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


# ---------------------------------------------------------------------------
# Problem Blueprint：题目理解的结构化资产（analyst 产出，后续节点消费）
# ---------------------------------------------------------------------------

class SubQuestionBlueprint(BaseModel):
    id: str
    original_text: str
    task_type: Literal[
        "evaluation", "prediction", "optimization", "simulation",
        "classification", "explanation", "strategy", "generic",
    ]
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""
    success_criteria: list[str] = Field(default_factory=list)


class DecisionVariable(BaseModel):
    name: str
    meaning: str
    domain: str = ""
    unit: str = ""


class ObjectiveSpec(BaseModel):
    direction: Literal["minimize", "maximize", "evaluate", "predict", "explain"]
    description: str
    math_hint: str = ""


class ConstraintSpec(BaseModel):
    description: str
    math_hint: str = ""
    source: Literal["given", "derived", "assumed"] = "assumed"


class MetricSpec(BaseModel):
    name: str
    meaning: str
    direction: Literal["higher_better", "lower_better", "target", "neutral"] = "neutral"
    unit: str = ""


class DataRequirement(BaseModel):
    field: str
    meaning: str
    needed_for: str
    given_or_missing: Literal["given", "missing", "derived", "unknown"] = "unknown"
    handling_strategy: str = ""


class DataFileSheet(BaseModel):
    """xlsx/csv 的单个 sheet/表摘要。"""
    name: str
    rows: int = 0
    cols: int = 0
    columns: list[str] = Field(default_factory=list)
    preview: list[list[str]] = Field(default_factory=list)


class DataFileInfo(BaseModel):
    """上传附件的元信息，供 analyst/coder 理解数据。"""
    filename: str
    file_type: str                           # "xlsx"|"csv"|"pdf"|"docx"|"txt"|"md"
    path: str                                # 相对 data_dir 的文件名
    summary: dict = Field(default_factory=dict)  # 灵活 dict，不同类型结构不同


class ModelingCandidate(BaseModel):
    name: str
    route: str
    suitable_for: list[str] = Field(default_factory=list)
    data_required: list[str] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    risk: str = ""
    recommendation_score: int = Field(default=0, ge=0, le=10)


class RecommendedRoute(BaseModel):
    route: str
    reason: str
    baseline: str = ""
    fallback: str = ""


class ValidationPlanItem(BaseModel):
    target: str
    method: str
    pass_criteria: str = ""


class ProblemBlueprint(BaseModel):
    core_task: str
    subquestions: list[SubQuestionBlueprint] = Field(default_factory=list)
    decision_variables: list[DecisionVariable] = Field(default_factory=list)
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    data_requirements: list[DataRequirement] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    problem_domains: list[str] = Field(default_factory=list)
    modeling_candidates: list[ModelingCandidate] = Field(default_factory=list)
    recommended_route: RecommendedRoute | None = None
    validation_plan: list[ValidationPlanItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class DerivationStep(BaseModel):
    """模型推导链中的一步（动机 -> 数学陈述 -> 结果）。"""
    title: str                       # "参数估计" / "约束推导" / "等价变换"
    motivation: str                  # 为何做这步
    statement: str                   # 数学陈述（含 inline LaTeX）
    result: str = ""                 # 推导结论


class ModelQuestionCoverage(BaseModel):
    """模型对小问的覆盖说明（modeler 产出，model_critic 交叉验证）。"""
    question_id: str
    how_answered: str  # 必须引用具体 equation/variable 名称，不允许纯自然语言
    related_equations: list[str] = Field(default_factory=list)
    related_metrics: list[str] = Field(default_factory=list)


class ModelVersion(BaseModel):
    stage: ModelStage
    description: str
    equations: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    figure_purposes: list[str] = Field(default_factory=list)  # Plan D：modeler 建议要画的图
    derivation_steps: list[DerivationStep] = Field(default_factory=list)
    derivation_notes: str = ""  # Plan D：self-consistency gate 产出的问题标注
    # Problem Blueprint 对齐（modeler final 阶段填充）
    question_coverage: list[ModelQuestionCoverage] = Field(default_factory=list)
    objective_mapping: list[str] = Field(default_factory=list)
    constraint_mapping: list[str] = Field(default_factory=list)
    validation_mapping: list[str] = Field(default_factory=list)


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
    # 标识属于第几轮代码生成（coder 每次 retry 递增；一致性审查只看最新 batch）
    batch: int = 0


class CriticIssue(BaseModel):
    """结构化评审意见，section 字段限定到固定 enum。"""
    section: Literal[
        "abstract", "problem_restatement", "assumptions", "notation",
        "model_section", "solution", "sensitivity", "conclusion", "references", "general",
    ] = "general"
    problem: str


class CriticReport(BaseModel):
    target: Literal["analyst", "modeler", "coder", "writer", "paper"]
    score: int = Field(ge=0, le=10)
    issues: list[CriticIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    approved: bool = False
    # stage 标记 critic 是针对哪个建模阶段产生的；analyst/coder/writer/paper 类型可为 None
    stage: Optional[ModelStage] = None
    # critic_type 区分同 target 的不同 reviewer（如 target="analyst" 的 blueprint_critic
    # 和将来可能的 data_auditor）。默认空字符串兼容旧数据。
    critic_type: str = ""


class ModelCodeConsistencyReport(BaseModel):
    """模型-代码一致性审查报告。

    不复用 CriticReport：一致性审查需要细粒度的变量/目标/约束对齐信息
   （implemented_variables、missing_constraints 等），这些结构化字段无法无损嵌入
    CriticReport 的通用 issues/suggestions 列表。
    """
    score: int = Field(ge=0, le=10)
    approved: bool
    implemented_variables: list[str] = Field(default_factory=list)
    missing_variables: list[str] = Field(default_factory=list)
    implemented_objectives: list[str] = Field(default_factory=list)
    missing_objectives: list[str] = Field(default_factory=list)
    implemented_constraints: list[str] = Field(default_factory=list)
    missing_constraints: list[str] = Field(default_factory=list)
    output_metric_alignment: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


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
    quality_score: int = Field(default=0, ge=0, le=10)
    quality_issues: list[str] = Field(default_factory=list)
    analysis: str = ""               # FigureAnalyst 产出的段落


class EvaluationReport(BaseModel):
    """对齐国赛四大标准 + 国一加分项。每项 0-10。"""
    assumption_reasonableness: int = Field(ge=0, le=10)
    modeling_creativity: int = Field(ge=0, le=10)
    result_correctness: int = Field(ge=0, le=10)
    writing_clarity: int = Field(ge=0, le=10)
    extra_depth: int = Field(ge=0, le=10)  # 加分项：敏感性/创新/分析深度
    overall: float = Field(ge=0, le=10)  # 加权总评
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

    # 当前 Blueprint 的假设（覆盖语义）。analyst 重试后必须替换旧假设，
    # 否则 modeler/sensitivity 会继续消费被 critic 否决的首轮内容。
    assumptions: list[Assumption] = Field(default_factory=list)
    model_versions: Annotated[list[ModelVersion], add] = Field(default_factory=list)
    code_artifacts: Annotated[list[CodeArtifact], add] = Field(default_factory=list)
    critic_reports: Annotated[list[CriticReport], add] = Field(default_factory=list)
    sensitivity_runs: Annotated[list[SensitivityRun], add] = Field(default_factory=list)
    figures: Annotated[list[FigureArtifact], add] = Field(default_factory=list)
    # 模型-代码一致性报告（追加语义；一致性节点每次 retry 追加一份）
    model_code_reports: Annotated[list[ModelCodeConsistencyReport], add] = Field(default_factory=list)

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

    # Problem Blueprint（覆盖语义；analyst 每次 retry 整体替换）
    problem_blueprint: ProblemBlueprint | None = None
    blueprint_iteration: int = 0    # blueprint_critic 评估次数（首次 + 最多一次 retry）
    code_verify_iteration: int = 0  # model_code_consistency 评估次数

    # writer 子流程状态（覆盖语义）。队列空 = 本轮写完。
    # ponytail: 队列即进度，不需要 completed_groups/current_group/pending_rewrite。
    writer_section_queue: list[str] = Field(default_factory=list)
    writer_outline_dump: dict = Field(default_factory=dict)   # WriterOutline.model_dump()
    writer_retrieved_context: str = ""                        # RAG 检索结果，prep 查一次 section 复用
    # table_assembler 产出的清洗/注入警告（覆盖语义；每次 table_assembler 运行整体替换）
    table_warnings: list[str] = Field(default_factory=list)

    # 输出
    output_dir: Optional[str] = None

    # 附件数据（覆盖语义；run 启动时确定，中途不变）
    data_dir: Optional[str] = None
    data_files: list[DataFileInfo] = Field(default_factory=list)

    # LaTeX 模板选择 + 队伍信息（仅 gmcm 模板用到）
    latex_template: str = "default"   # "default" | "gmcm"
    school: Optional[str] = None
    team_id: Optional[str] = None
    members: Optional[str] = None     # "张三,李四,王五"

    # ---- 便利方法 ----
    def latest_model(self) -> ModelVersion | None:
        return self.model_versions[-1] if self.model_versions else None

    def latest_code_artifacts(self) -> list[CodeArtifact]:
        """返回最新 coder 批次；旧批次只用于追踪，不应进入论文或图表。"""
        if not self.code_artifacts:
            return []
        latest_batch = max(artifact.batch for artifact in self.code_artifacts)
        return [artifact for artifact in self.code_artifacts if artifact.batch == latest_batch]

    def latest_critic(self, target: str, critic_type: str = "") -> CriticReport | None:
        """返回最新的 target 匹配的 CriticReport。

        critic_type 非空时额外按 critic_type 筛选，防止同 target 的不同 reviewer 碰撞。
        """
        for r in reversed(self.critic_reports):
            if r.target == target:
                if not critic_type or r.critic_type == critic_type:
                    return r
        return None

    def latest_critic_for_stage(self, target: str, stage: ModelStage) -> CriticReport | None:
        """按 (target, stage) 过滤，避免上一阶段未通过的反馈污染下一阶段。"""
        for r in reversed(self.critic_reports):
            if r.target == target and r.stage == stage:
                return r
        return None
