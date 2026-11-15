# 课程意见研读资料约定

资料全部是本地合成业务记录，不含真实个人信息。`term-a` 为完整版本，`term-b` 保持业务值不变而调整表行次序，用于检验分析是否不恰当地依赖到达顺序；`limited` 保留登记资料并移除一类观测，表示证据不足的版本。版本内容不可就地修改。

- `feedback`：业务唯一键为 `document`。
- `terminology`：业务唯一键为 `document, original, applies_from`。

document 是不可变反馈编号，person 可重复；term 为先后次序。文本用空格分隔是当前导入方式，保留中文原文用于追溯。terminology 是指定文档引出的人工建议，applies_from 表示建议可用于当期及未来分析，不能用于较早期训练。channel 为采集渠道，不代表内容质量高低。资料版本会增加学期及术语意见，旧分析保持原样。

字段之外不能从示例的编号顺序或随机生成规律推导业务关系。涉及权重、相关范围、模型支撑的判断必须使用明确登记的字段和业务假设。

## topic_evolution 使用资料的方式

- 按 `term` 分期独立拟合；`document` 是结果中代表性原文与逐文档归属的回查键，`site`、`channel` 原样进入校区与渠道分布，不做质量高低推断。
- `person` 同学期重复留言、完全重复的 `text` 只通过显式权重假设下调影响（可在 parameters 中关闭），不删除证据；权重与逐文档权重随结果返回。
- 短文本（词数低于 parameters.min_tokens）降低权重而非剔除，避免短留言被静默排除。
- terminology 的 original/preferred 只在 `applies_from <= term` 时替换建模词项，原始 text 永不改写；每条替换记录 occurrences 与证据 document，冲突的 preferred 会在 terminology_conflicts 中报告。limited 版本没有 terminology 行，等价于不启用术语，原始词项直接进入模型。
- 主题跨期衔接只比较相邻两期的主题词项分布（Jensen-Shannon 相似度），相似度落在弱区间时输出 `continuation_unconfirmed`，不允许用更晚学期的留言或人工改词回溯重判历史主题。
