# 分析服务接口

资料服务在 8765 端口提供 `/releases`、`/releases/term-a` 和版本内的表分页。表页携带 revision；客户端必须读取同一清单版本并验证摘要，不能合并不同版本的页。

应用服务在 8766 端口提供 `GET /catalog`、`POST /analyses`、`GET /analyses`、`GET /analyses/{id}`。当前登记 overview 课程资料汇总与 topic_evolution 跨学期主题演化。分析实现登记在 `learning.domain.ANALYSES`，函数接收完整 tables 和 parameters，返回可序列化对象。

创建请求字段为 request_key、release、analysis 和可选的 parameters 对象。request_key 是调用方的幂等编号：相同完整请求返回同一分析，相同编号不同内容返回 409。分析返回 id、request、revision 和 result。不同 request_key 是不同留存记录。缺少字段、未知分析类型或无效资料返回 422；查询不存在的记录返回 404。JSON 不允许 NaN 或 Infinity，无法估计的量使用 null 并在结果中说明原因。

保存后的分析内容不因资料更新而重算。新分析可在 result 内保存统计对象、参数、诊断及所需的版本化结构；不可覆盖其他分析或跳过资料完整性校验。数据库只保存本地分析，不是资料服务的业务真源。

## topic_evolution 参数与结果

`"analysis": "topic_evolution"` 比较 parameters.terms（缺省为资料中全部学期，按 term 升序）内各期的主题结构。每个学期独立拟合模型，主题身份 `topic_id` 由该期主题词项分布摘要派生；`fit_index` 只是本次拟合的内部编号，不跨期继承，不能把不同学期相同编号的簇视为同一主题。第 t 期拟合只读取第 t 期留言，衔接诊断只使用相邻两期词表的并集，不读取更晚学期；术语意见仅在 `applies_from <= term` 时参与建模，永不回溯改写更早学期。

parameters 可选项（全部带缺省值，未知字段返回 422）：

- `model`：`type` 为 `lda`（缺省）或 `nmf`；`n_topics` 为 2..12 的整数或 `"auto"`（缺省，按 UMass 一致性在 `auto_min`..`auto_max` 间选择，候选分数随结果返回）；另含 `max_iter`、`random_state`，固定随机种子保证可复现。
- `weighting`：`adjusted`（缺省）按同一人同学期留言数、完全重复文本数下调权重，并对词数少于 `min_tokens` 的短文本乘 `short_weight`；设为 `none` 表示等权，用于比较权重假设本身的影响。`person_cap`、`duplicate_text_cap` 为封顶计数（不小于 1）。
- `use_terminology`（缺省 true）：是否启用人工术语意见。术语是带适用起点的证据而非永久字典；关闭后原始词项直接参与建模。
- 衔接阈值：`align_min`（强衔接）、`split_merge_min`（连通分量内保留为强边的下限）、`align_weak`（低于该值判定无衔接），需满足 `align_weak <= split_merge_min <= align_min`；`ambiguity_margin` 为文档归属前两主题概率差的模糊判定线。
- `terms`：学期编号的严格递增子集；`top_words`、`representatives`、`max_evidence` 控制回查信息规模。
- `alternatives`：最多 5 个备选建模规格（不可再嵌套），每个对象可覆盖上述任意参数并用 `name` 命名。基线与备选分别拟合，结果 `alternative_comparisons` 给出逐期文档归属调整兰德指数、事件状态一致率和双方事件计数，用于比较建模选择；基线结果不会被备选覆盖。

result 主要结构：

- `term_models[]`：每期 `summary`（文档数、去重后有效样本量、渠道占比、当期生效的术语替换及证据文档、术语冲突、拟合质量与收敛标志）、`topics[]`（主题词及权重、文档数、加权占比、校区分布、渠道占比、代表性原文 document/site/person/channel/text）、`ambiguous_assignments[]`（归属模糊文档及候选主题概率）、`assignment_detail`（逐文档归属与权重，便于校区回查）、`fit_info`（候选主题数与一致性分数）。
- `transitions[]`：相邻学期的衔接事件，状态为 `continues`、`split`、`merge`、`split_and_merge`、`continuation_unconfirmed`、`emerged`、`discontinued`。每条边含 `confidence`（数值或 null）与 `confidence_level`（high/medium/low/unconfirmed）；弱衔接只保留 `continuation_unconfirmed`，不强行连线，也不进入轨迹拼接。边内 `diagnostics` 区分变化来源：`vocabulary_substitution`（术语替换份额）、`channel_composition`（渠道全变差距离）、`content_change`（内容距离）或 `stable`，并列出新词、消失词和判定理由。
- `trajectories[]`：仅由强衔接连通的主题轨迹；无法确认延续的主题不会被拼接。
- `parameters`：本次实际生效的完整参数；`schema_version` 为结果结构版本。

## 主题版本留存

不同建模选择应以不同 request_key（或不同 parameters）分别保存为独立分析记录；相同 request_key 改参数返回 409，旧记录保持原样。回读时通过 result.parameters 与 alternative_comparisons.parameters_fingerprint 区分版本。
