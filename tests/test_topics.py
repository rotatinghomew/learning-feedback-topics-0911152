import copy
import json
from pathlib import Path
import unittest

from learning.codec import digest, encode
from learning.domain import ANALYSES, validate
from learning.topics import (
    STATUS_CONTINUES,
    STATUS_DISCONTINUED,
    STATUS_EMERGED,
    STATUS_UNCONFIRMED,
    validate_parameters,
)


def build_tables(rows, terminology=None):
    return {
        "feedback": [
            {"document": f"d{i:03d}", "person": person, "term": term, "site": site,
             "channel": channel, "text": text}
            for i, (term, person, site, channel, text) in enumerate(rows)
        ],
        "terminology": terminology or [],
    }


# 三个内容清晰的主题；alpha/beta/gamma 三组词在同一文档内不混杂。
THEME_A = "手机 挂号 扫码 操作 慢 老师"
THEME_B = "书法 作品 交流 同伴 活动 练习"
THEME_C = "预约 交通 名额 时间 调整 课程"


def balanced_term(term, size=12, seed_shift=0):
    rows = []
    for i in range(size):
        theme = (THEME_A, THEME_B, THEME_C)[i % 3]
        words = theme.split()
        shift = (i + seed_shift) % len(words)
        text = " ".join(words[shift:] + words[:shift])
        rows.append((term, f"p{term}-{i}", f"s{i % 4}",
                     "phone" if i % 4 == 0 else "written", text))
    return rows


class TopicEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.release = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]
        validate(self.release)

    def test_result_is_serializable_and_deterministic(self):
        first = ANALYSES["topic_evolution"](self.release, {"model": {"n_topics": 3}})
        second = ANALYSES["topic_evolution"](self.release, {"model": {"n_topics": 3}})
        encode(first)
        self.assertEqual(first, second)

    def test_does_not_mutate_evidence(self):
        tables = copy.deepcopy(self.release)
        before = digest(tables)
        ANALYSES["topic_evolution"](tables, {"model": {"n_topics": 3}})
        self.assertEqual(before, digest(tables))

    def test_order_independence(self):
        params = {"model": {"n_topics": 3}}
        before = ANALYSES["topic_evolution"](self.release, params)
        shuffled = {name: list(reversed(rows)) for name, rows in self.release.items()}
        after = ANALYSES["topic_evolution"](shuffled, params)
        self.assertEqual(digest(before), digest(after))

    def test_topics_are_aligned_by_content_not_fit_index(self):
        result = ANALYSES["topic_evolution"](self.release, {"model": {"n_topics": 3}})
        ids = [[topic["topic_id"] for topic in model["topics"]] for model in result["term_models"]]
        for earlier, later in zip(ids, ids[1:]):
            self.assertEqual(set(earlier) & set(later), set(), "topic_id 必须随内容重新派生，不能继承编号")
        for edge in result["transitions"]:
            for link in edge["links"]:
                self.assertTrue(link["from_topic"].startswith("t"))
            for topic_id in edge["from_topics"] + edge["to_topics"]:
                self.assertRegex(topic_id, r"^t\d+-[0-9a-f]{10}$")
        # 每条衔接都能回查到代表性原文与校区。
        for model in result["term_models"]:
            for topic in model["topics"]:
                self.assertTrue(topic["representatives"])
                for representative in topic["representatives"]:
                    self.assertTrue({"document", "site", "text"} <= set(representative))
                self.assertTrue(topic["sites"])

    def test_strict_threshold_keeps_unconfirmed_without_forcing_links(self):
        result = ANALYSES["topic_evolution"](
            self.release,
            {"model": {"n_topics": 3}, "align_min": 0.97, "split_merge_min": 0.95, "align_weak": 0.3})
        statuses = [edge["status"] for edge in result["transitions"]]
        self.assertIn(STATUS_UNCONFIRMED, statuses)
        unconfirmed = [edge for edge in result["transitions"]
                       if edge["status"] == STATUS_UNCONFIRMED]
        for edge in unconfirmed:
            self.assertEqual(edge["confidence_level"], "unconfirmed")
            self.assertIsNotNone(edge["confidence"])
            self.assertEqual(len(edge["to_topics"]), 1)
        # 无法确认的衔接不得把两侧主题拼进同一条轨迹。
        trajectories = result["trajectories"]
        for edge in unconfirmed:
            self.assertFalse(any(
                set(edge["from_topics"]) <= set(trajectory["topic_ids"])
                and set(edge["to_topics"]) <= set(trajectory["topic_ids"])
                for trajectory in trajectories))

    def test_terminology_only_applies_from_applies_from(self):
        result = ANALYSES["topic_evolution"](self.release, {"model": {"n_topics": 3}})
        self.assertEqual(result["term_models"][0]["summary"]["terminology_applied"], [])
        self.assertEqual(result["term_models"][1]["summary"]["terminology_applied"], [])
        applied = result["term_models"][2]["summary"]["terminology_applied"]
        originals = {item["original"] for item in applied}
        self.assertEqual(originals, {"云课", "打卡"})
        for item in applied:
            self.assertTrue(item["evidence_documents"])
            # 证据文档必须真实存在且属于该学期。
            docs = {row["document"] for row in self.release["feedback"]
                    if row["term"] == 2}
            self.assertTrue(set(item["evidence_documents"]) <= docs)
        # 关闭术语开关后结果不同，但原始资料未被改写。
        raw_tokens = [token for row in self.release["feedback"] for token in row["text"].split()]
        self.assertIn("云课", raw_tokens)
        without = ANALYSES["topic_evolution"](
            self.release, {"model": {"n_topics": 3}, "use_terminology": False})
        self.assertEqual(without["term_models"][2]["summary"]["terminology_applied"], [])

    def test_terminology_is_evidence_not_permanent_dictionary(self):
        # limited 版本没有术语表：同样的原始文本不得被静默改写。
        limited = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["limited"]
        result = ANALYSES["topic_evolution"](limited, {"model": {"n_topics": 3}})
        for model in result["term_models"]:
            self.assertEqual(model["summary"]["terminology_applied"], [])
        words = {word for model in result["term_models"] for topic in model["topics"]
                 for word, _ in topic["top_words"]}
        self.assertIn("云课", words)
        self.assertNotIn("线上课程", words)

    def test_emergence_and_discontinuation_on_synthetic_topics(self):
        def term_rows(term, themes, size_per=8):
            rows = []
            index = 0
            for theme in themes:
                words = theme.split()
                for _ in range(size_per):
                    shift = index % len(words)
                    text = " ".join(words[shift:] + words[:shift])
                    rows.append((term, f"p{term}-{index}", f"s{index % 4}",
                                 "phone" if index % 4 == 0 else "written", text))
                    index += 1
            return rows

        extra_theme = "直播 回放 弹幕 连麦 课件 笔记"
        rows = term_rows(0, [THEME_A, THEME_B, THEME_C])
        rows += term_rows(1, [THEME_A, THEME_B, THEME_C])
        # 第三期：主题 A 消失，全新词项主题出现，B/C 保留。
        rows += term_rows(2, [THEME_B, THEME_C, extra_theme])
        tables = build_tables(rows)
        validate(tables)
        result = ANALYSES["topic_evolution"](
            tables, {"model": {"type": "nmf", "n_topics": 3}, "align_weak": 0.2})
        second_transition = [edge for edge in result["transitions"] if edge["transition"] == [1, 2]]
        statuses = {edge["status"] for edge in second_transition}
        self.assertTrue({STATUS_EMERGED, STATUS_DISCONTINUED} <= statuses)
        emerged = next(edge for edge in second_transition if edge["status"] == STATUS_EMERGED)
        self.assertEqual(emerged["from_topics"], [])
        self.assertEqual(len(emerged["to_topics"]), 1)
        discontinued = next(edge for edge in second_transition if edge["status"] == STATUS_DISCONTINUED)
        self.assertEqual(discontinued["to_topics"], [])

    def test_duplicate_and_repeated_person_are_downweighted(self):
        rows = balanced_term(0, size=9)
        # 同一人连续 6 条、且文本完全相同的转述，不应主导主题。
        repeated_text = THEME_A
        rows += [(0, "p-spam", "s0", "written", repeated_text) for _ in range(6)]
        tables = build_tables(rows)
        validate(tables)
        weighted = ANALYSES["topic_evolution"](tables, {"model": {"n_topics": 3}})
        unweighted = ANALYSES["topic_evolution"](
            tables, {"model": {"n_topics": 3}, "weighting": "none"})
        note = weighted["term_models"][0]["fit_info"]["weighting"]
        self.assertGreater(note["person_adjustments"], 0)
        self.assertGreater(note["duplicate_adjustments"], 0)
        self.assertLess(weighted["term_models"][0]["summary"]["effective_sample_size"],
                        unweighted["term_models"][0]["summary"]["effective_sample_size"])
        self.assertEqual(unweighted["term_models"][0]["summary"]["effective_sample_size"], 15)

    def test_short_documents_get_reduced_weight(self):
        rows = balanced_term(0, size=9)
        rows.append((0, "p-short", "s1", "written", "手机 慢"))
        tables = build_tables(rows)
        validate(tables)
        result = ANALYSES["topic_evolution"](tables, {"model": {"n_topics": 3}, "min_tokens": 3})
        note = result["term_models"][0]["fit_info"]["weighting"]
        self.assertEqual(note["short_documents"], 1)
        detail = result["term_models"][0]["assignment_detail"]
        short_index = next(i for i, row in enumerate(tables["feedback"]) if row["document"] == "d009")
        self.assertLess(detail["weights"][short_index], 1.0)

    def test_ambiguous_assignments_are_exposed(self):
        # 高主题数迫使部分文档归属模糊。
        result = ANALYSES["topic_evolution"](
            self.release,
            {"model": {"n_topics": 6}, "ambiguity_margin": 0.35})
        self.assertTrue(any(model["ambiguous_assignments"] for model in result["term_models"]))
        for model in result["term_models"]:
            for item in model["ambiguous_assignments"]:
                self.assertEqual(len(item["candidate_topics"]), 2)
                self.assertIn(item["assigned_topic"],
                              [candidate["topic_id"] for candidate in item["candidate_topics"]])

    def test_alternatives_are_compared_without_overwriting_baseline(self):
        result = ANALYSES["topic_evolution"](
            self.release,
            {"model": {"n_topics": 3},
             "alternatives": [{"name": "nmf", "model": {"type": "nmf", "n_topics": 3}}]})
        self.assertEqual(len(result["alternative_comparisons"]), 1)
        comparison = result["alternative_comparisons"][0]
        self.assertEqual(comparison["label"], "nmf")
        self.assertTrue(0 <= comparison["mean_assignment_ari"] <= 1)
        self.assertEqual(result["parameters"]["model"]["type"], "lda")
        self.assertNotEqual(comparison["parameters_fingerprint"],
                            digest(result["parameters"]))

    def test_nested_alternatives_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_parameters(
                {"alternatives": [{"alternatives": []}]}, [0, 1])
        with self.assertRaises(ValueError):
            validate_parameters(
                {"alternatives": [{} for _ in range(6)]}, [0, 1])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_parameters({"weighting": "mystery"}, [0, 1])
        with self.assertRaises(ValueError):
            validate_parameters({"model": {"n_topics": 1}}, [0, 1])
        with self.assertRaises(ValueError):
            validate_parameters({"align_min": 0.4, "align_weak": 0.5}, [0, 1])
        with self.assertRaises(ValueError):
            validate_parameters({"terms": [1, 0]}, [0, 1])
        with self.assertRaises(ValueError):
            validate_parameters({"unknown_option": 1}, [0, 1])

    def test_term_subset_only_uses_selected_evidence(self):
        result = ANALYSES["topic_evolution"](
            self.release, {"terms": [2, 3], "model": {"n_topics": 3}})
        self.assertEqual([model["summary"]["term"] for model in result["term_models"]], [2, 3])
        self.assertEqual({tuple(edge["transition"]) for edge in result["transitions"]}, {(2, 3)})
        for model in result["term_models"]:
            self.assertEqual(model["summary"]["document_count"], 90)

    def test_continuation_carries_change_diagnostics(self):
        result = ANALYSES["topic_evolution"](self.release, {"model": {"n_topics": 3}})
        for edge in result["transitions"]:
            self.assertEqual(edge["status"], STATUS_CONTINUES)
            for link in edge["links"]:
                diagnostics = link["diagnostics"]
                self.assertIn(diagnostics["change_driver"],
                              {"stable", "vocabulary_substitution", "channel_composition", "content_change"})
                self.assertTrue(0 <= diagnostics["js_similarity"] <= 1)
                self.assertTrue(diagnostics["reasons"])

    def test_channel_shock_is_attributed_to_channel_composition(self):
        def term_rows(term, themes, channel_flip=False, size_per=8):
            rows = []
            index = 0
            for theme in themes:
                words = theme.split()
                for _ in range(size_per):
                    shift = index % len(words)
                    text = " ".join(words[shift:] + words[:shift])
                    channel = "phone" if channel_flip or index % 4 == 0 else "written"
                    rows.append((term, f"p{term}-{index}", f"s{index % 4}", channel, text))
                    index += 1
            return rows

        rows = term_rows(0, [THEME_A, THEME_B, THEME_C])
        rows += term_rows(1, [THEME_A, THEME_B, THEME_C], channel_flip=True)
        tables = build_tables(rows)
        validate(tables)
        result = ANALYSES["topic_evolution"](
            tables, {"model": {"type": "nmf", "n_topics": 3}})
        for edge in result["transitions"]:
            for link in edge["links"]:
                diagnostics = link["diagnostics"]
                self.assertEqual(diagnostics["change_driver"], "channel_composition")
                self.assertGreater(diagnostics["channel_tv_distance"], 0.2)
                self.assertLess(diagnostics["content_distance"], 0.3)

    def test_content_drift_is_not_branded_vocabulary_substitution(self):
        drifted = "手机 挂号 网络 信号 卡顿 杂音"

        def base_rows(term, themes, size_per=8):
            rows = []
            index = 0
            for theme in themes:
                words = theme.split()
                for _ in range(size_per):
                    shift = index % len(words)
                    rows.append((term, f"p{term}-{index}", f"s{index % 4}",
                                 "phone" if index % 4 == 0 else "written",
                                 " ".join(words[shift:] + words[:shift])))
                    index += 1
            return rows

        drift_rows = []
        for i in range(8):
            words = drifted.split()
            shift = i % len(words)
            drift_rows.append((1, f"p1-d{i}", "s0", "written",
                               " ".join(words[shift:] + words[:shift])))
        rows = base_rows(0, [THEME_A, THEME_B, THEME_C])
        rows += drift_rows + base_rows(1, [THEME_B, THEME_C])
        tables = build_tables(rows)
        validate(tables)
        result = ANALYSES["topic_evolution"](
            tables, {"model": {"type": "nmf", "n_topics": 3}, "align_weak": 0.1})
        drivers = {link["diagnostics"]["change_driver"]
                   for edge in result["transitions"] for link in edge["links"]}
        self.assertIn("content_change", drivers)
        self.assertNotIn("vocabulary_substitution", drivers)
        drift_edge = next(link for edge in result["transitions"] for link in edge["links"]
                          if link["diagnostics"]["change_driver"] == "content_change")
        self.assertIn("网络", dict(drift_edge["diagnostics"]["novel_words"]))


if __name__ == "__main__":
    unittest.main()
