"""跨学期主题演化分析。

设计约束（与 docs/api.md、docs/data-contract.md 对应）：

* 每个学期独立拟合主题模型，主题编号只是本次拟合的内部下标；主题身份由
  内容摘要派生，跨学期衔接只依据相邻两期的词项分布相似度，不把相同编号的
  簇视为同一主题。
* 第 t 期的拟合只使用第 t 期的资料；衔接诊断只使用相邻两期的并表词表，
  不读取更晚学期的留言。人工术语意见带有 applies_from，只在当期及以后
  生效，永不回溯改写更早的学期。
* 重复转述、同一人多次留言、短文本通过显式权重假设下调影响；权重假设是
  parameters 的一部分，可整体关闭以便比较。
* 相似度不足时保留“无法确认延续”，不强行连线。
"""
from collections import Counter, defaultdict
import hashlib
import math
import warnings

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score

from learning.codec import digest

MODEL_LDA = "lda"
MODEL_NMF = "nmf"
WEIGHTING_ADJUSTED = "adjusted"
WEIGHTING_NONE = "none"

STATUS_CONTINUES = "continues"
STATUS_SPLIT = "split"
STATUS_MERGE = "merge"
STATUS_MIXTURE = "split_and_merge"
STATUS_UNCONFIRMED = "continuation_unconfirmed"
STATUS_EMERGED = "emerged"
STATUS_DISCONTINUED = "discontinued"

DEFAULT_RANDOM_STATE = 20261115
_DEFAULTS = {
    "terms": None,
    "weighting": WEIGHTING_ADJUSTED,
    "person_cap": 3.0,
    "duplicate_text_cap": 3.0,
    "min_tokens": 3,
    "short_weight": 0.5,
    "min_token_count": 1,
    "use_terminology": True,
    "align_min": 0.55,
    "align_weak": 0.35,
    "split_merge_min": 0.5,
    "ambiguity_margin": 0.15,
    "top_words": 8,
    "representatives": 3,
    "max_evidence": 25,
    "alternatives": [],
}
_MODEL_DEFAULTS = {
    "type": MODEL_LDA,
    "n_topics": "auto",
    "auto_min": 2,
    "auto_max": 4,
    "max_iter": 80,
    "random_state": DEFAULT_RANDOM_STATE,
}


# --------------------------------------------------------------------------- #
# 参数校验与规格合并
# --------------------------------------------------------------------------- #
def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_parameters(parameters, available_terms, _nested=False):
    _check(isinstance(parameters, dict), "分析参数必须是对象")
    unknown = set(parameters) - set(_DEFAULTS) - {"model"}
    _check(not unknown, f"未知分析参数：{sorted(unknown)}")
    _check(not (_nested and parameters.get("alternatives")), "备选规格不能再嵌套备选规格")
    spec = dict(_DEFAULTS)
    spec.update({key: value for key, value in parameters.items() if key != "model"})
    model = dict(_MODEL_DEFAULTS)
    if "model" in parameters:
        _check(isinstance(parameters["model"], dict), "model 必须是对象")
        _check(not (set(parameters["model"]) - set(_MODEL_DEFAULTS)), "model 含未知字段")
        model.update(parameters["model"])

    _check(spec["weighting"] in (WEIGHTING_ADJUSTED, WEIGHTING_NONE), "weighting 取值不合法")
    for name in ("person_cap", "duplicate_text_cap"):
        _check(_is_number(spec[name]) and spec[name] >= 1, f"{name} 必须是不小于 1 的有限数")
    _check(isinstance(spec["min_tokens"], int) and spec["min_tokens"] >= 1, "min_tokens 必须是正整数")
    _check(_is_number(spec["short_weight"]) and 0 < spec["short_weight"] <= 1, "short_weight 必须在 (0,1] 内")
    _check(isinstance(spec["min_token_count"], int) and spec["min_token_count"] >= 1, "min_token_count 必须是正整数")
    _check(isinstance(spec["use_terminology"], bool), "use_terminology 必须是布尔值")
    for name in ("align_min", "align_weak", "split_merge_min", "ambiguity_margin"):
        _check(_is_number(spec[name]) and 0 < spec[name] < 1, f"{name} 必须在 (0,1) 内")
    _check(spec["align_weak"] <= spec["split_merge_min"] <= spec["align_min"], "衔接阈值需满足 align_weak <= split_merge_min <= align_min")
    _check(isinstance(spec["top_words"], int) and 1 <= spec["top_words"] <= 30, "top_words 超出范围")
    _check(isinstance(spec["representatives"], int) and 1 <= spec["representatives"] <= 10, "representatives 超出范围")
    _check(isinstance(spec["max_evidence"], int) and spec["max_evidence"] >= 1, "max_evidence 必须是正整数")
    _check(isinstance(spec["alternatives"], list), "alternatives 必须是数组")
    _check(len(spec["alternatives"]) <= 5, "备选规格最多 5 个")

    if spec["terms"] is None:
        spec["terms"] = sorted(available_terms)
    _check(isinstance(spec["terms"], list) and spec["terms"], "terms 必须是非空数组")
    _check(len(spec["terms"]) <= 12, "学期数量超出范围")
    _check(all(isinstance(term, int) and not isinstance(term, bool) for term in spec["terms"]), "学期编号必须是整数")
    _check(spec["terms"] == sorted(spec["terms"]) and len(set(spec["terms"])) == len(spec["terms"]), "学期编号必须严格递增且不重复")
    _check(set(spec["terms"]) <= set(available_terms), "terms 含资料中不存在的学期")

    _validate_model(model)
    spec["model"] = model
    alternative_specs = []
    for index, override in enumerate(spec["alternatives"]):
        _check(isinstance(override, dict), f"备选规格 {index} 必须是对象")
        _check("alternatives" not in override, "备选规格不能再嵌套备选规格")
        alt_params = {key: value for key, value in parameters.items()
                      if key not in ("alternatives", "model")}
        alt_params.update({key: value for key, value in override.items()
                           if key not in ("model", "name")})
        if "model" in override:
            alt_params["model"] = override["model"]
        alt = validate_parameters(alt_params, available_terms, _nested=True)
        alt["label"] = override.get("name", f"alternative-{index + 1}")
        alternative_specs.append(alt)
    spec["alternatives"] = alternative_specs
    spec["label"] = "baseline"
    return spec


def _validate_model(model):
    _check(model["type"] in (MODEL_LDA, MODEL_NMF), "模型类型必须是 lda 或 nmf")
    if model["n_topics"] != "auto":
        _check(isinstance(model["n_topics"], int) and not isinstance(model["n_topics"], bool)
               and 2 <= model["n_topics"] <= 12, "n_topics 必须是 2..12 的整数或 auto")
    _check(isinstance(model["auto_min"], int) and isinstance(model["auto_max"], int)
           and 2 <= model["auto_min"] <= model["auto_max"] <= 12, "auto 主题数范围不合法")
    _check(isinstance(model["max_iter"], int) and model["max_iter"] >= 10, "max_iter 不合法")
    _check(isinstance(model["random_state"], int) and 0 <= model["random_state"] < 2**32, "random_state 不合法")


# --------------------------------------------------------------------------- #
# 纯数值小工具
# --------------------------------------------------------------------------- #
def _round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def _clean(value):
    """把 numpy 类型递归转成原生 JSON 类型，并拒绝非有限数。"""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            raise ValueError("结果含非有限数值，应改为 null 并说明原因")
        return float(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("结果含非有限数值，应改为 null 并说明原因")
    return value


def _js_similarity(left, right):
    """两概率分布的 Jensen-Shannon 相似度（1 - JS 散度，底 2，区间 [0,1]）。"""
    middle = (left + right) / 2.0

    def kl_divergence(side, ref):
        mask = side > 0
        return float(np.sum(side[mask] * np.log2(side[mask] / ref[mask])))

    return max(0.0, 1.0 - (kl_divergence(left, middle) + kl_divergence(right, middle)) / 2.0)


def _total_variation(left, right):
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _topic_id(term, word_weights):
    payload = [(word, _round(weight, 6)) for word, weight in word_weights[:10]]
    return f"t{term}-" + hashlib.sha256(digest(payload).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# 证据准备：术语、权重、矩阵
# --------------------------------------------------------------------------- #
def _term_rows(feedback, term):
    return sorted((row for row in feedback if row["term"] == term), key=lambda row: row["document"])


def _terminology_mapping(terminology, term):
    """返回当期生效的 original -> preferred 映射；只收集 applies_from <= term 的意见。"""
    mapping = {}
    conflicts = Counter()
    for row in terminology:
        if row["applies_from"] > term:
            continue
        previous = mapping.get(row["original"])
        if previous is not None and previous != row["preferred"]:
            conflicts[row["original"]] += 1
        mapping.setdefault(row["original"], row["preferred"])
    return mapping, conflicts


def _normalize_text(text, mapping):
    raw_tokens = text.split()
    if not mapping:
        return raw_tokens, raw_tokens, Counter()
    normalized = []
    applied = Counter()
    for token in raw_tokens:
        preferred = mapping.get(token)
        if preferred is None:
            normalized.append(token)
        else:
            normalized.append(preferred)
            applied[(token, preferred)] += 1
    return raw_tokens, normalized, applied


def _document_weights(rows, spec):
    if spec["weighting"] == WEIGHTING_NONE:
        return [1.0] * len(rows), {"weighting": WEIGHTING_NONE, "person_adjustments": 0,
                                   "duplicate_adjustments": 0, "short_documents": 0}
    person_counts = Counter(row["person"] for row in rows)
    duplicate_counts = Counter(" ".join(row["text"].split()) for row in rows)
    weights = []
    person_hits = duplicate_hits = short_hits = 0
    for row in rows:
        person_factor = min(1.0, spec["person_cap"] / person_counts[row["person"]])
        duplicate_factor = min(1.0, spec["duplicate_text_cap"] / duplicate_counts[" ".join(row["text"].split())])
        short = len(row["text"].split()) < spec["min_tokens"]
        short_factor = spec["short_weight"] if short else 1.0
        if person_factor < 1.0:
            person_hits += 1
        if duplicate_factor < 1.0:
            duplicate_hits += 1
        if short:
            short_hits += 1
        weights.append(person_factor * duplicate_factor * short_factor)
    return weights, {"weighting": WEIGHTING_ADJUSTED, "person_adjustments": person_hits,
                     "duplicate_adjustments": duplicate_hits, "short_documents": short_hits}


def _build_matrix(rows, weights, mapping, spec):
    raw_streams, modeled_streams, applied_counters = [], [], []
    for row in rows:
        raw_tokens, normalized, applied = _normalize_text(row["text"], mapping)
        raw_streams.append(raw_tokens)
        modeled_streams.append(normalized)
        applied_counters.append(applied)
    token_frequency = Counter(token for stream in modeled_streams for token in stream)
    vocab = sorted(token for token, count in token_frequency.items() if count >= spec["min_token_count"])
    index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(rows), len(vocab)))
    for i, stream in enumerate(modeled_streams):
        for token, count in Counter(stream).items():
            if token in index:
                matrix[i, index[token]] = weights[i] * count
    return vocab, matrix, raw_streams, modeled_streams, applied_counters


# --------------------------------------------------------------------------- #
# 模型拟合与主题数选择
# --------------------------------------------------------------------------- #
def _fit_one(model_spec, matrix, random_offset=0):
    seed = model_spec["random_state"] + random_offset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if model_spec["type"] == MODEL_LDA:
            estimator = LatentDirichletAllocation(
                n_components=model_spec["n_topics"], random_state=seed,
                learning_method="batch", max_iter=model_spec["max_iter"], n_jobs=1)
            assignment = estimator.fit_transform(matrix)
            quality_metric = "perplexity"
            quality_value = estimator.perplexity(matrix)
            converged = estimator.n_iter_ < estimator.max_iter
        else:
            estimator = NMF(n_components=model_spec["n_topics"], init="nndsvda",
                            random_state=seed, max_iter=max(model_spec["max_iter"] * 5, 500))
            assignment = estimator.fit_transform(matrix)
            quality_metric = "reconstruction_error"
            quality_value = float(estimator.reconstruction_err_)
            converged = estimator.n_iter_ < estimator.max_iter
    assignment = assignment / np.maximum(assignment.sum(axis=1, keepdims=True), 1e-12)
    topic_word = estimator.components_ / np.maximum(estimator.components_.sum(axis=1, keepdims=True), 1e-12)
    return assignment, topic_word, quality_metric, quality_value, converged


def _umass_coherence(matrix, topic_word, top_n=10):
    present = (matrix > 0).astype(float)
    doc_counts = present.sum(axis=0)
    scores = []
    for topic in topic_word:
        top_indices = np.argsort(-topic)[:top_n]
        pair_scores = []
        for pos in range(1, len(top_indices)):
            later = top_indices[pos]
            for earlier in top_indices[:pos]:
                cooccur = float(present[:, earlier].dot(present[:, later]))
                pair_scores.append(math.log((cooccur + 1.0) / max(doc_counts[later], 1.0)))
        if pair_scores:
            scores.append(sum(pair_scores) / len(pair_scores))
    return float(np.mean(scores)) if scores else None


def _fit_term(model_spec, matrix):
    if model_spec["n_topics"] != "auto":
        assignment, topic_word, metric, value, converged = _fit_one(model_spec, matrix)
        coherence = _umass_coherence(matrix, topic_word)
        return assignment, topic_word, {"chosen_k": model_spec["n_topics"],
                                         "candidate_coherence": [],
                                         "fit_quality": {"metric": metric, "value": _round(value),
                                                         "converged": converged},
                                         "coherence": _round(coherence)}
    candidates = []
    fits = {}
    for k in range(model_spec["auto_min"], model_spec["auto_max"] + 1):
        candidate_spec = dict(model_spec, n_topics=k)
        assignment, topic_word, metric, value, converged = _fit_one(candidate_spec, matrix, random_offset=k)
        coherence = _umass_coherence(matrix, topic_word)
        candidates.append({"k": k, "umass_coherence": _round(coherence),
                           "fit_quality": {"metric": metric, "value": _round(value),
                                           "converged": converged}})
        fits[k] = (assignment, topic_word, coherence, metric, value, converged)
    best_k = max(candidates, key=lambda item: (item["umass_coherence"], -item["k"]))["k"]
    assignment, topic_word, coherence, metric, value, converged = fits[best_k]
    return assignment, topic_word, {"chosen_k": best_k, "candidate_coherence": candidates,
                                    "fit_quality": {"metric": metric, "value": _round(value),
                                                    "converged": converged},
                                    "coherence": _round(coherence)}


# --------------------------------------------------------------------------- #
# 单期主题刻画
# --------------------------------------------------------------------------- #
def _describe_term(term, rows, weights, spec, mapping, terminology_conflicts):
    vocab, matrix, raw_streams, modeled_streams, applied_counters = _build_matrix(rows, weights, mapping, spec)
    _check(matrix.shape[0] >= 2 and matrix.shape[1] >= 2, "该学期可用证据不足，无法拟合主题模型")
    assignment, topic_word, fit_info = _fit_term(spec["model"], matrix)
    if spec["model"]["n_topics"] == "auto":
        _check(fit_info["chosen_k"] <= len(rows), "主题数超过该学期留言数")
    else:
        _check(spec["model"]["n_topics"] <= len(rows), "主题数超过该学期留言数")

    topics = []
    ambiguous = []
    order = np.argsort(-topic_word, axis=1)
    for topic_index in range(topic_word.shape[0]):
        word_weights = [(vocab[j], float(topic_word[topic_index, j])) for j in order[topic_index]
                        if topic_word[topic_index, j] > 0][:spec["top_words"]]
        topic = {
            "topic_id": _topic_id(term, word_weights),
            "term": term,
            "fit_index": int(topic_index),
            "top_words": [(word, _round(weight)) for word, weight in word_weights],
            "document_count": 0,
            "weighted_share": _round(float(assignment[:, topic_index].sum() / max(assignment.sum(), 1e-12))),
            "sites": {},
            "channels": {},
            "representatives": [],
        }
        topics.append(topic)

    assigned = assignment.argmax(axis=1)
    topic_channels = defaultdict(Counter)
    topic_sites = defaultdict(Counter)
    for i, row in enumerate(rows):
        label = int(assigned[i])
        topics[label]["document_count"] += 1
        topic_sites[label][row["site"]] += 1
        topic_channels[label][row["channel"]] += 1
        ranked = np.argsort(-assignment[i])
        margin = float(assignment[i, ranked[0]] - assignment[i, ranked[1]])
        if margin < spec["ambiguity_margin"]:
            ambiguous.append({
                "document": row["document"], "site": row["site"], "assigned_topic": topics[label]["topic_id"],
                "margin": _round(margin),
                "candidate_topics": [{"topic_id": topics[int(j)]["topic_id"],
                                      "probability": _round(assignment[i, j])} for j in ranked[:2]],
            })
    for label, topic in enumerate(topics):
        topic["sites"] = dict(sorted(topic_sites[label].items()))
        channel_counts = topic_channels[label]
        total = sum(channel_counts.values())
        topic["channels"] = {channel: _round(count / total) for channel, count in sorted(channel_counts.items())}
        members = [i for i in range(len(rows)) if assigned[i] == label]
        members.sort(key=lambda i: (-assignment[i, label], rows[i]["document"]))
        topic["representatives"] = [{
            "document": rows[i]["document"], "site": rows[i]["site"],
            "person": rows[i]["person"], "channel": rows[i]["channel"],
            "probability": _round(assignment[i, label]), "text": rows[i]["text"],
        } for i in members[:spec["representatives"]]]

    applied = Counter()
    evidence = defaultdict(list)
    for row, counter in zip(rows, applied_counters):
        for pair, count in counter.items():
            applied[pair] += count
            if len(evidence[pair]) < spec["max_evidence"]:
                evidence[pair].append(row["document"])
    terminology_used = [{
        "original": pair[0], "preferred": pair[1], "occurrences": count,
        "evidence_documents": sorted(evidence[pair]),
    } for pair, count in sorted(applied.items())]

    effective_n = (sum(weights) ** 2) / max(sum(w * w for w in weights), 1e-12)
    channel_share = Counter()
    for row in rows:
        channel_share[row["channel"]] += 1
    raw_vocab = sorted({token for stream in raw_streams for token in stream})
    summary = {
        "term": term,
        "document_count": len(rows),
        "person_count": len({row["person"] for row in rows}),
        "effective_sample_size": _round(effective_n),
        "vocabulary_size": len(raw_vocab),
        "channel_share": {channel: _round(count / len(rows)) for channel, count in sorted(channel_share.items())},
        "terminology_applied": terminology_used,
        "terminology_conflicts": [{"original": word, "conflicts": count}
                                  for word, count in sorted(terminology_conflicts.items())],
    }
    return {"summary": summary, "topics": topics,
            "ambiguous_assignments": ambiguous, "fit_info": fit_info,
            "_internal": {"vocab": vocab, "topic_word": topic_word, "assignment": assignment,
                          "rows": rows, "weights": weights, "raw_vocab": raw_vocab,
                          "modeled_streams": modeled_streams,
                          "weight_diagnostics": {"effective_sample_size": _round(effective_n)}}}


# --------------------------------------------------------------------------- #
# 相邻学期衔接
# --------------------------------------------------------------------------- #
def _similarity_matrix(left_internal, right_internal):
    left_vocab, right_vocab = left_internal["vocab"], right_internal["vocab"]
    union = sorted(set(left_vocab) | set(right_vocab))
    left_index = {word: i for i, word in enumerate(left_vocab)}
    right_index = {word: i for i, word in enumerate(right_vocab)}
    left_pad = np.zeros((left_internal["topic_word"].shape[0], len(union)))
    right_pad = np.zeros((right_internal["topic_word"].shape[0], len(union)))
    for j, word in enumerate(union):
        if word in left_index:
            left_pad[:, j] = left_internal["topic_word"][:, left_index[word]]
        if word in right_index:
            right_pad[:, j] = right_internal["topic_word"][:, right_index[word]]
    matrix = np.zeros((left_pad.shape[0], right_pad.shape[0]))
    for i in range(left_pad.shape[0]):
        for j in range(right_pad.shape[0]):
            matrix[i, j] = _js_similarity(left_pad[i], right_pad[j])
    return matrix


def _confidence(similarity, spec):
    if similarity < spec["align_weak"]:
        return None, "相邻学期主题词项分布相似度低于弱衔接阈值，无法确认延续"
    if similarity >= 0.85:
        level = "high"
    elif similarity >= 0.6:
        level = "medium"
    else:
        level = "low"
    return _round(similarity), level


def _edge_diagnostics(left_term, right_term, left_topic, right_topic, sim_matrix,
                      left_internal, right_internal, spec, terminology_words):
    li, ri = left_topic["fit_index"], right_topic["fit_index"]
    similarity = float(sim_matrix[li, ri])
    left_assigned = left_internal["assignment"].argmax(axis=1)
    right_assigned = right_internal["assignment"].argmax(axis=1)
    left_channels = Counter(row["channel"] for i, row in enumerate(left_internal["rows"])
                            if left_assigned[i] == li)
    right_channels = Counter(row["channel"] for i, row in enumerate(right_internal["rows"])
                             if right_assigned[i] == ri)
    left_total = max(sum(left_channels.values()), 1)
    right_total = max(sum(right_channels.values()), 1)
    left_dist = {channel: count / left_total for channel, count in left_channels.items()}
    right_dist = {channel: count / right_total for channel, count in right_channels.items()}
    channel_tv = _total_variation(left_dist, right_dist)

    previous_vocab = set(left_internal["raw_vocab"])
    novel_words = Counter()
    disappeared_counter = Counter()
    substitution_mass = 0
    right_streams = right_internal["modeled_streams"]
    right_tokens = 0
    for i, stream in enumerate(right_streams):
        if right_assigned[i] != ri:
            continue
        right_tokens += len(stream)
        for token in stream:
            if token not in previous_vocab:
                novel_words[token] += 1
                if token in terminology_words:
                    substitution_mass += 1
    left_streams = left_internal["modeled_streams"]
    current_vocab = set(right_internal["raw_vocab"])
    for i, stream in enumerate(left_streams):
        if left_assigned[i] == li:
            for token in stream:
                if token not in current_vocab:
                    disappeared_counter[token] += 1
    novel_mass = sum(novel_words.values()) / max(right_tokens, 1)
    substitution_share = substitution_mass / max(right_tokens, 1)
    terminology_hits = sorted(set(novel_words) & terminology_words)

    column_margin = float(sim_matrix[li, ri] - np.max(np.delete(sim_matrix[li], ri)))
    row_margin = float(sim_matrix[li, ri] - np.max(np.delete(sim_matrix[:, ri], li)))
    content_distance = 1.0 - similarity
    has_substitution = substitution_share >= 0.05
    has_novel_vocabulary = novel_mass >= 0.1
    has_channel_shift = channel_tv >= 0.2
    has_content_shift = content_distance >= 0.3
    reasons = []
    if has_substitution:
        reasons.append("承接文档中出现人工术语对应的新说法（术语替换），词面变化不一定代表内容变化")
    elif has_novel_vocabulary:
        reasons.append("承接文档中出现上期没有的词项，存在新增说法")
    if has_channel_shift:
        reasons.append("衔接主题的渠道构成变化明显，部分差异可能来自采集渠道")
    if has_content_shift:
        reasons.append("词项分布相似度偏低，内容本身发生变化")
    if not reasons:
        reasons.append("词项分布与渠道构成均保持稳定")
    if has_substitution and not has_content_shift:
        driver = "vocabulary_substitution"
    elif has_channel_shift and not has_content_shift and not has_novel_vocabulary:
        driver = "channel_composition"
    elif has_content_shift:
        driver = "content_change"
    else:
        driver = "stable"
    return {
        "js_similarity": _round(similarity),
        "content_distance": _round(content_distance),
        "channel_tv_distance": _round(channel_tv),
        "novel_word_mass": _round(novel_mass),
        "terminology_substitution_share": _round(substitution_share),
        "novel_words": novel_words.most_common(spec["top_words"]),
        "disappeared_words": disappeared_counter.most_common(spec["top_words"]),
        "terminology_words_in_novel_set": terminology_hits,
        "assignment_margin_min": _round(min(column_margin, row_margin)),
        "change_driver": driver,
        "reasons": reasons,
    }


def _align_terms(left_bundle, right_bundle, spec, terminology_words):
    left_internal, right_internal = left_bundle["_internal"], right_bundle["_internal"]
    sim_matrix = _similarity_matrix(left_internal, right_internal)
    n_left, n_right = sim_matrix.shape
    strong = sim_matrix >= spec["align_min"]
    weak = sim_matrix >= spec["align_weak"]

    # 以强衔接构建二部图连通分量，允许一个主题同时连向多个主题（分裂/合并）。
    adjacency = defaultdict(set)
    for i in range(n_left):
        for j in range(n_right):
            if strong[i, j]:
                adjacency[("L", i)].add(("R", j))
                adjacency[("R", j)].add(("L", i))
    seen, components = set(), []
    for node in list(adjacency):
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(component)

    edges, covered_left, covered_right = [], set(), set()
    for component in components:
        left_nodes = sorted(i for side, i in component if side == "L")
        right_nodes = sorted(j for side, j in component if side == "R")
        covered_left.update(left_nodes)
        covered_right.update(right_nodes)
        links = [{"from_topic": left_bundle["topics"][i]["topic_id"],
                  "to_topic": right_bundle["topics"][j]["topic_id"],
                  "js_similarity": _round(float(sim_matrix[i, j])),
                  "diagnostics": _edge_diagnostics(
                      left_bundle["summary"]["term"], right_bundle["summary"]["term"],
                      left_bundle["topics"][i], right_bundle["topics"][j],
                      sim_matrix, left_internal, right_internal, spec, terminology_words)}
                 for i in left_nodes for j in right_nodes if strong[i, j]]
        mean_sim = float(np.mean([link["js_similarity"] for link in links]))
        confidence, level = _confidence(mean_sim, spec)
        if len(left_nodes) == 1 and len(right_nodes) == 1:
            status = STATUS_CONTINUES
        elif len(left_nodes) == 1:
            status = STATUS_SPLIT
        elif len(right_nodes) == 1:
            status = STATUS_MERGE
        else:
            status = STATUS_MIXTURE
        edges.append({
            "transition": [left_bundle["summary"]["term"], right_bundle["summary"]["term"]],
            "status": status,
            "from_topics": [left_bundle["topics"][i]["topic_id"] for i in left_nodes],
            "to_topics": [right_bundle["topics"][j]["topic_id"] for j in right_nodes],
            "confidence": confidence,
            "confidence_level": level,
            "links": links,
        })

    # 只有弱衔接、没有强衔接的主题对：保留“无法确认延续”。
    for i in range(n_left):
        if i in covered_left:
            continue
        weak_targets = [j for j in range(n_right) if weak[i, j]]
        if weak_targets:
            j = max(weak_targets, key=lambda candidate: (float(sim_matrix[i, candidate]), -candidate))
            edges.append({
                "transition": [left_bundle["summary"]["term"], right_bundle["summary"]["term"]],
                "status": STATUS_UNCONFIRMED,
                "from_topics": [left_bundle["topics"][i]["topic_id"]],
                "to_topics": [right_bundle["topics"][j]["topic_id"]],
                "confidence": _round(float(sim_matrix[i, j])),
                "confidence_level": "unconfirmed",
                "links": [{"from_topic": left_bundle["topics"][i]["topic_id"],
                          "to_topic": right_bundle["topics"][j]["topic_id"],
                          "js_similarity": _round(float(sim_matrix[i, j])),
                          "diagnostics": _edge_diagnostics(
                              left_bundle["summary"]["term"], right_bundle["summary"]["term"],
                              left_bundle["topics"][i], right_bundle["topics"][j],
                              sim_matrix, left_internal, right_internal, spec, terminology_words)}],
            })
            covered_right.add(j)
        else:
            edges.append({
                "transition": [left_bundle["summary"]["term"], right_bundle["summary"]["term"]],
                "status": STATUS_DISCONTINUED,
                "from_topics": [left_bundle["topics"][i]["topic_id"]],
                "to_topics": [],
                "confidence": None,
                "confidence_level": None,
                "links": [],
            })
    for j in range(n_right):
        if j not in covered_right:
            edges.append({
                "transition": [left_bundle["summary"]["term"], right_bundle["summary"]["term"]],
                "status": STATUS_EMERGED,
                "from_topics": [],
                "to_topics": [right_bundle["topics"][j]["topic_id"]],
                "confidence": None,
                "confidence_level": None,
                "links": [],
            })
    edges.sort(key=lambda edge: (edge["transition"], edge["status"], edge["from_topics"], edge["to_topics"]))
    return edges, sim_matrix


def _build_trajectories(bundles, transitions):
    neighbors = defaultdict(set)
    nodes = {(bundle["summary"]["term"], topic["topic_id"])
             for bundle in bundles for topic in bundle["topics"]}
    for edge in transitions:
        if edge["status"] in (STATUS_UNCONFIRMED, STATUS_EMERGED, STATUS_DISCONTINUED):
            continue
        for link in edge["links"]:
            neighbors[link["from_topic"]].add(link["to_topic"])
            neighbors[link["to_topic"]].add(link["from_topic"])
    seen, trajectories = set(), []
    for term, topic_id in sorted(nodes):
        if topic_id in seen:
            continue
        stack, component = [topic_id], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(neighbors[current] - seen)
        ordered = sorted(component, key=lambda topic: topic)
        trajectories.append({
            "trajectory_id": hashlib.sha256("|".join(ordered).encode()).hexdigest()[:12],
            "topic_ids": ordered,
            "span_terms": sorted(int(topic.split("-", 1)[0][1:]) for topic in ordered),
        })
    return trajectories


# --------------------------------------------------------------------------- #
# 规格间比较
# --------------------------------------------------------------------------- #
def _compare_specifications(baseline, alternative, spec_terms):
    per_term = []
    for term in spec_terms:
        base_bundle = next(bundle for bundle in baseline["bundles"] if bundle["summary"]["term"] == term)
        alt_bundle = next(bundle for bundle in alternative["bundles"] if bundle["summary"]["term"] == term)
        base_labels = base_bundle["_internal"]["assignment"].argmax(axis=1)
        alt_labels = alt_bundle["_internal"]["assignment"].argmax(axis=1)
        base_docs = [row["document"] for row in base_bundle["_internal"]["rows"]]
        alt_docs = [row["document"] for row in alt_bundle["_internal"]["rows"]]
        assert base_docs == alt_docs
        per_term.append({"term": term,
                         "assignment_ari": _round(adjusted_rand_score(base_labels, alt_labels))})
    alt_statuses = [edge["status"] for edge in alternative["result"]["transitions"]]
    base_statuses = [edge["status"] for edge in baseline["result"]["transitions"]]
    agree = sum(1 for left, right in zip(base_statuses, alt_statuses) if left == right)
    return {
        "label": alternative["spec"]["label"],
        "parameters_fingerprint": digest(_public_parameters(alternative["spec"])),
        "mean_assignment_ari": _round(np.mean([item["assignment_ari"] for item in per_term])),
        "assignment_ari_by_term": per_term,
        "event_status_agreement": _round(agree / max(len(base_statuses), 1)),
        "baseline_event_counts": dict(Counter(base_statuses)),
        "alternative_event_counts": dict(Counter(alt_statuses)),
    }


def _public_parameters(spec):
    return {key: value for key, value in spec.items() if key not in ("alternatives", "label")}


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def topic_evolution(tables, parameters):
    feedback = tables["feedback"]
    available_terms = sorted({row["term"] for row in feedback})
    spec = validate_parameters(parameters or {}, available_terms)

    def run(single_spec):
        bundles = []
        terminology_words = set()
        for term in single_spec["terms"]:
            rows = _term_rows(feedback, term)
            mapping, conflicts = _terminology_mapping(
                tables["terminology"] if single_spec["use_terminology"] else [], term)
            terminology_words.update(mapping.values())
            weights, weight_note = _document_weights(rows, single_spec)
            bundle = _describe_term(term, rows, weights, single_spec, mapping, conflicts)
            bundle["fit_info"]["weighting"] = weight_note
            bundles.append(bundle)

        transitions = []
        for left_bundle, right_bundle in zip(bundles, bundles[1:]):
            edges, _ = _align_terms(left_bundle, right_bundle, single_spec, terminology_words)
            transitions.extend(edges)
        trajectories = _build_trajectories(bundles, transitions)

        term_models = []
        for bundle in bundles:
            internal = bundle["_internal"]
            term_models.append({
                "summary": bundle["summary"],
                "topics": bundle["topics"],
                "ambiguous_assignments": bundle["ambiguous_assignments"],
                "fit_info": bundle["fit_info"],
                "assignment_detail": {
                    "documents": [row["document"] for row in internal["rows"]],
                    "assigned_fit_index": [int(label) for label in internal["assignment"].argmax(axis=1)],
                    "weights": [_round(weight) for weight in internal["weights"]],
                }})
        result = {
            "schema_version": 1,
            "label": single_spec["label"],
            "parameters": _public_parameters(single_spec),
            "term_models": term_models,
            "transitions": transitions,
            "trajectories": trajectories,
            "interpretation_notes": [
                "主题身份由内容摘要派生，fit_index 仅为当期拟合的内部编号，不跨期继承",
                "第 t 期拟合只使用第 t 期证据；术语意见仅在 applies_from 及以后学期生效",
                "continuation_unconfirmed 表示证据不足，不能据此宣称主题延续",
            ],
        }
        return {"spec": single_spec, "bundles": bundles, "result": result}

    baseline = run(spec)
    comparisons = [_compare_specifications(baseline, run(alt), spec["terms"])
                   for alt in spec["alternatives"]]
    baseline["result"]["alternative_comparisons"] = comparisons
    return _clean(baseline["result"])
