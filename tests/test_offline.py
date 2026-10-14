#!/usr/bin/env python
"""Offline sanity checks - no GPU, no model download, no network needed.

Covers the pieces most likely to break silently: the JSON parser ladder (both
Task 1's label-only schema and Task 2's label+confidence+evidence schema),
the shared metrics/reliability accounting, Task 2's evidence/calibration/
format-compliance quality metrics, prompt construction (Task 1's plain
zero-shot prompt, its CARP ablation, and both Task 2 strategies), dataset
cleaning (against a mocked HF loader, so no network call happens),
`.env`/HF token lookup, and Task 3's instruction formatting + training
helpers (prompt/answer masking, the data collator, log tidying - against a
fake whitespace tokenizer, no model download). Run before any long GPU job.

Usage:
    python tests/test_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import tempfile

import pandas as pd

from src import (
    dataset, eval_task2, instruction_format, json_parser, metrics, prompt_sensitivity,
    prompts, prompts2, prompts_basic, prompts_carp, self_consistency, train_task3, utils,
)
from src.utils import LABELS

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((PASS if condition else FAIL, name, detail))


# --------------------------------------------------------------------------
# 1. JSON parser ladder
# --------------------------------------------------------------------------
def test_parser() -> None:
    cases = [
        # (raw model output, expected label, expected strategy)
        ('{"label": "positive"}', "positive", "strict"),
        ('{"label":"negative"}', "negative", "strict"),
        ('{ "label" : "Positive" }', "positive", "strict"),
        ('```json\n{"label": "positive"}\n```', "positive", "fenced"),
        ('```\n{"label": "negative"}\n```', "negative", "fenced"),
        ('Sure! Here is the answer: {"label": "negative"}', "negative", "substring"),
        ('{"label": "positive", "confidence": 0.9}', "positive", "strict"),
        ("label: negative", "negative", "regex_kv"),
        ('{"label" = "positive"', "positive", "regex_kv"),
        ("positive", "positive", "bare_label"),
        ("The sentiment is clearly Negative.", "negative", "bare_label"),
        ('{"label": "pos"}', "positive", "strict"),          # alias
    ]
    for raw, want_label, want_strategy in cases:
        r = json_parser.parse(raw)
        check(f"parse {raw[:34]!r}",
              r.ok and r.label == want_label and r.strategy == want_strategy,
              f"got ok={r.ok} label={r.label} strategy={r.strategy}")

    # Failures must fail, not guess.
    for raw, why in [
        ("", "empty_output"),
        ("I cannot determine this.", "no_label_found"),
        ("It is either positive or negative.", "ambiguous_multiple_labels"),
    ]:
        r = json_parser.parse(raw)
        check(f"reject {raw[:34]!r}", (not r.ok) and r.error == why,
              f"got ok={r.ok} error={r.error}")

    # strict_schema mode rejects anything that was not clean JSON.
    lenient = json_parser.parse_and_validate("positive", strict_schema=True)
    check("strict_schema rejects bare_label", not lenient.ok, f"got {lenient.error}")
    extra = json_parser.parse_and_validate('{"label":"positive","why":"x"}', strict_schema=True)
    check("strict_schema rejects extra keys", not extra.ok, f"got {extra.error}")


# --------------------------------------------------------------------------
# 2. Prompt construction
# --------------------------------------------------------------------------
def test_prompts() -> None:
    msgs = prompts.build_messages("this movie rules", attempt=0)
    check("prompt has system+user", len(msgs) == 2 and msgs[0]["role"] == "system")
    check("prompt mentions both labels",
          all(l in msgs[0]["content"].lower() for l in LABELS))
    check("attempt 0 has no retry suffix", "previous reply" not in msgs[1]["content"])
    check("review text injected", "this movie rules" in msgs[1]["content"])
    check("no neutral label mentioned (IMDB is binary)",
          "neutral" not in msgs[0]["content"].lower())

    retry = prompts.build_messages("y", attempt=1)
    check("attempt 1 adds a stricter suffix", "not valid JSON" in retry[1]["content"])
    final = prompts.build_messages("y", attempt=9)  # clamps to the last suffix
    check("attempt index clamps", "FINAL ATTEMPT" in final[1]["content"])

    meta = prompts.prompt_metadata()
    check("prompt metadata says zero-shot", meta["n_examples_in_prompt"] == 0)


# --------------------------------------------------------------------------
# 2a. Task 1 ablation: CARP (Clue And Reasoning Prompting) prompt construction
# --------------------------------------------------------------------------
def test_prompts_carp() -> None:
    msgs = prompts_carp.build_messages("this movie rules", attempt=0)
    check("carp prompt has system+user", len(msgs) == 2 and msgs[0]["role"] == "system")
    check("carp prompt mentions both labels",
          all(l in msgs[0]["content"].lower() for l in LABELS))
    check("carp prompt requires clue and reasoning",
          "clue" in msgs[0]["content"].lower() and "reasoning" in msgs[0]["content"].lower())
    check("carp review text injected", "this movie rules" in msgs[1]["content"])

    retry = prompts_carp.build_messages("y", attempt=1)
    check("carp attempt 1 adds a stricter suffix", "not valid JSON" in retry[1]["content"])
    final = prompts_carp.build_messages("y", attempt=9)  # clamps to the last suffix
    check("carp attempt index clamps", "FINAL ATTEMPT" in final[1]["content"])

    meta = prompts_carp.prompt_metadata()
    check("carp prompt metadata identifies the method", meta["method"] == "carp")

    # The whole point of reusing json_parser.parse() unchanged for CARP: a
    # clue/reasoning/label reply must still resolve to a clean "strict" parse,
    # with clue/reasoning surfaced as extra_keys rather than breaking anything.
    carp_reply = ('{"clue": "brilliant acting, loved it", "reasoning": "The reviewer '
                  'repeatedly praises the acting and states they loved the film, with '
                  'no negative remarks, so the verdict is positive.", "label": "positive"}')
    r = json_parser.parse(carp_reply)
    check("json_parser recovers the label from a CARP reply with no code changes",
          r.ok and r.label == "positive" and r.strategy == "strict",
          f"got ok={r.ok} label={r.label} strategy={r.strategy}")
    check("clue/reasoning surface as extra_keys, not a parse failure",
          set(r.extra_keys) == {"clue", "reasoning"}, f"got {r.extra_keys}")


# --------------------------------------------------------------------------
# 2a-bis. Task 1 ablation: the minimal basic prompt
# --------------------------------------------------------------------------
def test_prompts_basic() -> None:
    msgs = prompts_basic.build_messages("this movie rules", attempt=0)
    check("basic prompt has system+user", len(msgs) == 2 and msgs[0]["role"] == "system")
    check("basic prompt asks for POSITIVE/NEGATIVE",
          "POSITIVE" in msgs[0]["content"] and "NEGATIVE" in msgs[0]["content"])
    check("basic prompt has no label definitions or rules (that's the point)",
          "disambiguat" not in msgs[0]["content"].lower()
          and "rule" not in msgs[0]["content"].lower())
    check("basic review text injected", "this movie rules" in msgs[1]["content"])
    check("attempt 0 ends with a bare Sentiment: cue",
          msgs[1]["content"].rstrip().endswith("Sentiment:"))

    retry = prompts_basic.build_messages("y", attempt=1)
    check("basic attempt 1 adds a stricter suffix", "exactly one word" in retry[1]["content"])
    final = prompts_basic.build_messages("y", attempt=9)  # clamps to the last suffix
    check("basic attempt index clamps", "FINAL ATTEMPT" in final[1]["content"])

    meta = prompts_basic.prompt_metadata()
    check("basic prompt metadata identifies the method", meta["method"] == "basic")

    # The whole point: a bare "POSITIVE" reply (no JSON at all) must still
    # resolve via json_parser's most lenient rung, unmodified.
    r = json_parser.parse("POSITIVE")
    check("json_parser recovers a bare POSITIVE/NEGATIVE reply via bare_label",
          r.ok and r.label == "positive" and r.strategy == "bare_label",
          f"got ok={r.ok} label={r.label} strategy={r.strategy}")


# --------------------------------------------------------------------------
# 2b. Task 2: few-shot prompt loading + parse_fewshot
# --------------------------------------------------------------------------
def test_fewshot() -> None:
    for strategy in prompts2.STRATEGIES:
        msgs = prompts2.build_messages(strategy, "this movie rules")
        check(f"prompts2[{strategy}] has system+user",
              len(msgs) == 2 and msgs[0]["role"] == "system")
        check(f"prompts2[{strategy}] mentions both labels",
              all(l in msgs[0]["content"].lower() for l in LABELS))
        check(f"prompts2[{strategy}] requires confidence+evidence",
              "confidence" in msgs[0]["content"] and "evidence" in msgs[0]["content"])
        check(f"prompts2[{strategy}] review text injected",
              "this movie rules" in msgs[1]["content"])

    cases = [
        ('{"label": "positive", "confidence": 0.9, "evidence": "loves it"}',
         "positive", 0.9, "loves it", "strict"),
        ('```json\n{"label": "negative", "confidence": 0.8, "evidence": "hates it"}\n```',
         "negative", 0.8, "hates it", "fenced"),
        ('Sure! {"label": "positive", "confidence": 1.5, "evidence": "just a fact"}',
         "positive", 1.0, "just a fact", "substring"),  # confidence clamped to [0, 1]
    ]
    for raw, want_label, want_conf, want_evidence, want_strategy in cases:
        r = json_parser.parse_fewshot(raw)
        check(f"parse_fewshot {raw[:30]!r}",
              r.ok and r.label == want_label and r.confidence == want_conf
              and r.evidence == want_evidence and r.strategy == want_strategy
              and r.confidence_valid and not r.extra_keys,
              f"got ok={r.ok} label={r.label} conf={r.confidence} "
              f"evidence={r.evidence} strategy={r.strategy}")

    extra = json_parser.parse_fewshot('{"label": "positive", "confidence": 0.9, "evidence": "x", "why": "y"}')
    check("parse_fewshot flags extra keys", extra.ok and extra.extra_keys == ["why"], f"got {extra}")

    bad_conf = json_parser.parse_fewshot('{"label": "positive", "confidence": "high", "evidence": "x"}')
    check("parse_fewshot tolerates a non-numeric confidence",
          bad_conf.ok and bad_conf.confidence is None and not bad_conf.confidence_valid,
          f"got {bad_conf}")

    bad_label = json_parser.parse_fewshot('{"label": "unknown", "confidence": 0.5, "evidence": "x"}')
    check("parse_fewshot rejects an invalid label",
          not bad_label.ok and bad_label.error == "invalid_label", f"got {bad_label}")

    empty = json_parser.parse_fewshot("")
    check("parse_fewshot rejects empty output",
          not empty.ok and empty.error == "empty_output", f"got {empty}")


# --------------------------------------------------------------------------
# 3. Metrics: accuracy, macro-F1, per-class, confusion matrix, invalid rate
# --------------------------------------------------------------------------
def _mock_preds() -> pd.DataFrame:
    # 4 correct, 2 wrong, 2 unparsed -> accuracy 4/6 = 0.6667 over valid rows only
    rows = [
        ("positive", "positive"),
        ("positive", "positive"),
        ("positive", "negative"),
        ("negative", "negative"),
        ("negative", "negative"),
        ("negative", "positive"),
        ("positive", "unparsed"),
        ("negative", "unparsed"),
    ]
    return pd.DataFrame(rows, columns=["gold", "predicted"])


def test_metrics() -> None:
    m = metrics.compute(_mock_preds())
    check("n == 8", m["n"] == 8, str(m["n"]))
    check("n_valid == 6", m["n_valid"] == 6, str(m["n_valid"]))
    # Accuracy excludes the 2 unparsed rows from both numerator and
    # denominator: 4 correct / 6 valid, not 4 / 8.
    check("accuracy == 0.6667 (valid rows only)", m["accuracy"] == 0.6667, str(m["accuracy"]))
    # Overall accuracy counts the 2 unparsed rows as wrong: 4 correct / 8 total.
    check("accuracy_overall == 0.5 (unparsed counted as wrong)",
          m["accuracy_overall"] == 0.5, str(m["accuracy_overall"]))
    # Regression check: macro_f1 must average over the 2 real labels only, not
    # let the "unparsed" rows count as a silent 3rd pseudo-class.
    check("macro_f1 excludes the unparsed pseudo-class",
          m["macro_f1"] == 0.5714, str(m["macro_f1"]))
    check("invalid_output_rate == 0.25",
          m["invalid_output_rate"] == 0.25, str(m["invalid_output_rate"]))
    check("per-class covers both labels", set(m["per_class"]) == set(LABELS))
    cm = m["confusion_matrix"]
    check("confusion matrix gold rows are the 2 labels", cm["gold_labels"] == LABELS)
    check("confusion matrix predicted columns include unparsed",
          "unparsed" in cm["predicted_labels"])

    md = metrics.to_markdown(m, {"checkpoint": "test", "seed": 42})
    check("markdown renders", "Macro F1" in md and "Confusion matrix" in md)


# --------------------------------------------------------------------------
# 3b. Task 2 quality metrics: evidence relevance, calibration, format compliance
# --------------------------------------------------------------------------
def _mock_fewshot_preds() -> pd.DataFrame:
    rows = [
        # text, gold, predicted, confidence, confidence_valid, evidence, parse_strategy, extra_keys
        ("a wonderful uplifting film with great acting", "positive", "positive",
         0.9, True, "great acting and an uplifting story", "strict", []),
        ("a terrible boring waste of time", "negative", "negative",
         0.85, True, "the movie is boring and a waste of time", "strict", []),
        # wrong prediction + evidence that just echoes the label instead of citing the review
        ("an okay film with some decent moments", "positive", "negative",
         0.6, True, "the label is negative", "strict", []),
        # never parsed at all
        ("fantastic cinematography and story", "positive", "unparsed",
         None, False, None, None, []),
        # parsed via the lenient ladder, with an unexpected extra key -> not schema-compliant
        ("not great, kind of disappointing", "negative", "negative",
         0.55, True, None, "fenced", ["reason"]),
    ]
    cols = ["text", "gold", "predicted", "confidence", "confidence_valid",
            "evidence", "parse_strategy", "extra_keys"]
    return pd.DataFrame(rows, columns=cols)


def test_eval_task2() -> None:
    preds = _mock_fewshot_preds()
    q = eval_task2.evaluate(preds)

    ev = q["evidence"]
    check("evidence_present_rate == 0.6", ev["evidence_present_rate"] == 0.6, str(ev))
    check("label_echo_rate_among_present == 0.3333 (1 of 3 evidence-bearing rows echoes the label)",
          ev["label_echo_rate_among_present"] == round(1 / 3, 4), str(ev))
    check("mean_relevance is a real overlap score in (0, 1]",
          ev["mean_relevance"] is not None and 0 < ev["mean_relevance"] <= 1, str(ev))
    check("null_relevance is present and mean_relevance clears it (real lift over random pairing)",
          ev["null_relevance"] is not None and ev["relevance_lift"] is not None
          and ev["relevance_lift"] > 0, str(ev))
    check("null baseline uses derangements (no evidence paired with its own review) -> null == 0",
          ev["null_relevance"] == 0.0, str(ev))

    # _derangement is deterministic and has no fixed points for n > 1.
    for _seed in (0, 1, 42, 43, 99):
        for _n in (2, 3, 5, 17):
            d = eval_task2._derangement(_n, _seed)
            check(f"_derangement(n={_n}, seed={_seed}) is a fixed-point-free permutation",
                  sorted(d.tolist()) == list(range(_n)) and all(d[i] != i for i in range(_n)))
    check("_derangement(n=1) is None (no derangement exists)",
          eval_task2._derangement(1, 42) is None)
    check("mean_evidence_length_words is a positive word count",
          ev["mean_evidence_length_words"] is not None and ev["mean_evidence_length_words"] > 0, str(ev))

    cal = q["calibration"]
    check("calibration scores only the 4 rows with a numeric confidence",
          cal["n"] == 4 and cal["n_excluded_no_confidence"] == 1, str(cal))
    check("calibration mean_confidence == 0.725", cal["mean_confidence"] == 0.725, str(cal))
    check("calibration accuracy_on_scored_rows == 0.75", cal["accuracy_on_scored_rows"] == 0.75, str(cal))
    check("ece is a non-negative float", cal["ece"] is not None and cal["ece"] >= 0, str(cal))
    check("brier_score is a non-negative float", cal["brier_score"] is not None and cal["brier_score"] >= 0, str(cal))
    check("calibration coverage == 0.8 (4 of 5 rows scored)", cal["coverage"] == 0.8, str(cal))
    check("confidence_value_counts is a non-empty distribution",
          isinstance(cal["confidence_value_counts"], dict) and len(cal["confidence_value_counts"]) > 0, str(cal))

    # common_index restricts scoring to a shared row set so strategies with
    # different invalid rates are compared on an identical population.
    cal_common = eval_task2.evaluate_calibration(preds, common_index=preds.index[:3])
    check("calibration common_index restricts to the shared rows",
          cal_common["n"] == 3 and cal_common["coverage"] == 1.0, str(cal_common))
    check("evaluate() forwards common_index through to calibration",
          eval_task2.evaluate(preds, common_index=preds.index[:3])["calibration"]["n"] == 3)

    fmt = q["format_compliance"]
    # schema-compliant = strict JSON, no extra keys, numeric confidence, evidence present:
    # rows 1-2 yes, row 3 yes (echoed evidence is still schema-compliant, just low quality),
    # row 4 no (never parsed), row 5 no (fenced, not strict).
    check("schema_compliance_rate == 0.6", fmt["schema_compliance_rate"] == 0.6, str(fmt))
    check("confidence_present_rate == 0.8", fmt["confidence_present_rate"] == 0.8, str(fmt))
    check("evidence_present_rate == 0.6", fmt["evidence_present_rate"] == 0.6, str(fmt))
    check("extra_keys_rate == 0.2", fmt["extra_keys_rate"] == 0.2, str(fmt))
    check("strategy_breakdown counts all 3 extraction outcomes",
          fmt["strategy_breakdown"] == {"strict": 3, "fenced": 1, "no_json_found": 1},
          str(fmt["strategy_breakdown"]))
    check("n_valid == 4 (3 strict + 1 fenced, the no_json_found row excluded)",
          fmt["n_valid"] == 4, str(fmt))
    check("strict_rate_among_valid == 0.75 (3 of 4 parseable rows were strict JSON)",
          fmt["strict_rate_among_valid"] == 0.75, str(fmt))
    check("repaired_rate_among_valid == 0.25 (fenced row; complement of strict within valid)",
          fmt["repaired_rate_among_valid"] == 0.25, str(fmt))

    md = eval_task2.to_markdown(q)
    check("quality markdown renders", "Evidence relevance" in md and "Confidence calibration" in md
          and "Format compliance" in md)

    # A confident, well-calibrated, empty run shouldn't crash.
    empty_q = eval_task2.evaluate(preds.iloc[0:0])
    check("evaluate() handles an empty frame without raising",
          empty_q["calibration"]["n"] == 0 and empty_q["format_compliance"]["n"] == 0)


# --------------------------------------------------------------------------
# 3c. Prompt sensitivity: the winning-strategy perturbation harness
# --------------------------------------------------------------------------
def test_prompt_sensitivity() -> None:
    ps = prompt_sensitivity

    check("SUPPORTED_STRATEGIES is curated + random",
          set(ps.SUPPORTED_STRATEGIES) == {"curated", "random"})

    for strategy in ps.SUPPORTED_STRATEGIES:
        base = ps.baseline_for(strategy)

        # The baseline combination must rebuild the strategy's .txt byte for byte,
        # and its instruction block must equal the shared PHRASINGS["p1"].
        try:
            ps.assert_baseline_matches(strategy)
            check(f"[{strategy}] baseline reproduces the .txt + shares p1", True)
        except AssertionError as e:
            check(f"[{strategy}] baseline reproduces the .txt + shares p1", False, str(e))

        variants = ps.build_variants(strategy)
        labels = [v["label"] for v in variants]
        check(f"[{strategy}] build_variants includes the baseline once",
              labels.count("baseline") == 1, str(labels))
        check(f"[{strategy}] build_variants covers all 3 axes",
              {v["axis"] for v in variants} == {"baseline", "phrasing", "demo_order", "schema"},
              str({v["axis"] for v in variants}))
        check(f"[{strategy}] build_variants perturbs one axis at a time",
              all(sum(v[a] != base[a] for a in base) <= 1 for v in variants))
        check(f"[{strategy}] the baseline demo_order is never also a perturbation",
              [v["variant"] for v in variants].count(base["demo_order"]) == 0)

        only = ps.build_variants(strategy, only_axis="demo_order")
        check(f"[{strategy}] only_axis keeps the baseline + that axis",
              {v["axis"] for v in only} == {"baseline", "demo_order"}, str(only))

        # Demo order reorders the same 4 demos, nothing added or dropped.
        base_demos = ps._ordered_demos(strategy, base["demo_order"])
        check(f"[{strategy}] baseline demo order has 2 positive + 2 negative",
              sorted(d["label"] for d in base_demos)
              == ["negative", "negative", "positive", "positive"])
        for order in ps.DEMO_ORDERS:
            reordered = ps._ordered_demos(strategy, order)
            check(f"[{strategy}] demo_order[{order}] is a permutation of the demos",
                  sorted(d["text"] for d in reordered) == sorted(d["text"] for d in base_demos))
        check(f"[{strategy}] demo_order[neg_first] leads with a negative demo",
              ps._ordered_demos(strategy, "neg_first")[0]["label"] == "negative")
        check(f"[{strategy}] demo_order[alternating] alternates classes",
              [d["label"] for d in ps._ordered_demos(strategy, "alternating")]
              == ["positive", "negative", "positive", "negative"])

        # Renamed schema: prompt shows the new field names everywhere.
        renamed_sys = ps.build_system_prompt(strategy, "p1", base["demo_order"], "renamed")
        check(f"[{strategy}] renamed schema prints the new field names",
              '"sentiment"' in renamed_sys and '"justification"' in renamed_sys
              and '"label"' not in renamed_sys)

        # Reordered schema keeps canonical names.
        reordered_sys = ps.build_system_prompt(strategy, "p1", base["demo_order"], "reordered")
        check(f"[{strategy}] reordered schema keeps the canonical field names",
              '"label"' in reordered_sys and '"evidence"' in reordered_sys)

        # Every phrasing yields a usable chat message with the review injected.
        for phrasing in ps.PHRASINGS:
            msgs = ps.build_messages(strategy, phrasing, base["demo_order"], "baseline",
                                     "this film rules")
            check(f"[{strategy}] build_messages[{phrasing}] injects the review",
                  msgs[0]["role"] == "system" and "this film rules" in msgs[1]["content"])

    # parse_variant is strategy-independent.
    r = ps.parse_variant(
        '{"sentiment": "negative", "certainty": 0.8, "justification": "boring and slow"}',
        "renamed",
    )
    check("parse_variant maps a renamed reply back to canonical fields",
          r.ok and r.label == "negative" and r.confidence == 0.8
          and r.evidence == "boring and slow" and not r.extra_keys, str(r.as_dict()))
    bad = ps.parse_variant("no json here", "renamed")
    check("parse_variant still rejects unparseable output",
          not bad.ok and bad.error == "no_json_found", str(bad))
    rr = ps.parse_variant(
        '{"confidence": 0.7, "evidence": "great script", "label": "positive"}', "reordered",
    )
    check("parse_variant reads a reordered reply unchanged",
          rr.ok and rr.label == "positive" and rr.confidence == 0.7, str(rr.as_dict()))

    # Paraphrases must differ from the shared baseline and keep the definitional
    # content: the 4 example snippets and the sarcasm / mixed-review rules.
    _SNIPPETS = (
        "the pacing drags a bit, but the ending floored me",
        "an underrated gem I'll rewatch",
        "two hours I'll never get back",
        "great premise, wasted on a lazy script",
    )
    for name, text in ps.PHRASINGS.items():
        if name == "p1":
            continue
        check(f"phrasing[{name}] differs from the baseline wording", text != ps.PHRASINGS["p1"])
        check(f"phrasing[{name}] keeps all 4 example snippets",
              all(s in text for s in _SNIPPETS))
        check(f"phrasing[{name}] keeps the sarcasm + mixed-review rules",
              "sarcas" in text.lower() and "mixed" in text.lower())


# --------------------------------------------------------------------------
# 3d. Self-consistency vote aggregation
# --------------------------------------------------------------------------
def test_self_consistency() -> None:
    sc = self_consistency
    P = '{"label": "positive", "confidence": %s, "evidence": "%s"}'
    N = '{"label": "negative", "confidence": %s, "evidence": "%s"}'

    # Clean majority: 2 positive vs 1 negative.
    a = sc.aggregate([P % (0.9, "great acting"), P % (0.8, "loved the ending"), N % (0.7, "slow")])
    check("majority label wins", a["_label"] == "positive" and a["parse_ok"], str(a))
    check("n_valid_votes counts every parsed candidate", a["n_valid_votes"] == 3, str(a))
    check("vote_margin == winning / valid", a["vote_margin"] == round(2 / 3, 4), str(a))
    check("confidence = mean of the agreeing candidates only",
          a["confidence"] == 0.85, str(a))       # mean(0.9, 0.8)
    check("evidence = first agreeing candidate", a["evidence"] == "great acting", str(a))

    # One candidate is garbage: the row is still valid, vote is over the rest.
    b = sc.aggregate(["not json at all", P % (0.9, "x"), P % (0.6, "y")])
    check("row valid when >=1 candidate parses; invalids excluded from the vote",
          b["parse_ok"] and b["_label"] == "positive" and b["n_valid_votes"] == 2
          and b["vote_margin"] == 1.0, str(b))

    # Tie -> highest-confidence candidate's label.
    c = sc.aggregate([P % (0.55, "meh positive"), N % (0.95, "clearly bad")])
    check("tie broken by the highest-confidence candidate",
          c["_label"] == "negative" and c["vote_margin"] == 0.5, str(c))

    # No candidate parses at all -> unparsed row, compatible shape.
    d = sc.aggregate(["nope", "still nope", "{bad json"])
    check("all-invalid -> unparsed row",
          not d["parse_ok"] and d["_label"] is None and d["n_valid_votes"] == 0
          and d["vote_margin"] == 0.0 and d["parse_error"] == "no_valid_candidate", str(d))

    # The row (minus _label) must carry exactly the FewShotResult.as_dict()
    # keys plus the two new columns, so eval_task2.evaluate() stays happy.
    keys = set(sc.aggregate([P % (0.9, "x")]).keys()) - {"_label"}
    expected = set(json_parser.parse_fewshot(P % (0.9, "x")).as_dict()) | {"vote_margin", "n_valid_votes"}
    check("aggregate row is column-compatible with parse_fewshot + 2 extras",
          keys == expected, f"{keys ^ expected}")

    # A single candidate behaves like a plain parse (N=1 semantics if ever routed here).
    e = sc.aggregate([N % (0.77, "bad pacing")])
    check("single candidate -> that candidate, full margin",
          e["_label"] == "negative" and e["vote_margin"] == 1.0 and e["n_valid_votes"] == 1, str(e))


# --------------------------------------------------------------------------
# 4. Dataset: cleaning rules, label mapping, holdout independence
#
# `dataset._load_hf_split` is monkeypatched so these checks never touch the
# network - they exercise the same cleaning logic that runs on the real
# `stanfordnlp/imdb` splits, against small synthetic DataFrames shaped exactly
# like what `datasets.load_dataset(...).to_pandas()` returns (a `text` column
# and an integer `label` column, 0/1, with -1 marking "no gold label").
# --------------------------------------------------------------------------
def test_dataset() -> None:
    train_df = pd.DataFrame({
        "text": [
            "Absolutely great acting and story, loved every minute.",
            "One of the best films I have seen this year!",
            "Utter trash, waste of two hours of my life.",
            "Terribly written, terribly acted, avoid at all costs.",
            "<br /><br />A solid movie with <br />good pacing.",
            "ok",                                                     # too short
            "Absolutely great acting and story, loved every minute.",  # exact duplicate
            "Some unlabeled review text with no gold label at all.",
        ],
        "label": [1, 1, 0, 0, 1, 1, 1, -1],  # -1 = "unsupervised" split placeholder
    })
    test_df = pd.DataFrame({
        "text": [
            "A charming, well-acted little gem.",
            "Painfully dull from start to finish.",
            "Absolutely great acting and story, loved every minute.",  # overlaps the pool
        ],
        "label": [1, 0, 1],
    })
    splits = {"train": train_df, "test": test_df}

    def fake_load_hf_split(dataset_id: str, split: str, hf_token):
        return splits[split].copy()

    orig_loader = dataset._load_hf_split
    dataset._load_hf_split = fake_load_hf_split
    try:
        cfg = {
            "seed": 42,
            "data": {
                "hf_dataset": "stanfordnlp/imdb",
                "pool_split": "train",
                "holdout_split": "test",
                "min_chars": 3,
                "dedupe_on": ["text"],
            },
        }

        pool = dataset.load_clean(cfg)
        check("pool drops unlabeled + short + duplicate rows",
              len(pool) == 5, f"got {len(pool)}")
        check("pool has only positive/negative labels",
              set(pool["label"]) <= set(LABELS), str(pool["label"].unique()))
        check("HTML line breaks collapsed",
              not pool["text"].str.contains("<br", regex=False).any())
        check("uid is stable and unique", pool["uid"].is_unique)

        holdout = dataset.load_holdout(cfg, pool=pool)
        check("holdout drops the row shared with the pool",
              len(holdout) == 2, f"got {len(holdout)}")
        check("holdout is disjoint from the pool",
              not set(holdout["uid"]) & set(pool["uid"]))

        # Without a pool reference nothing is de-overlapped.
        raw_holdout = dataset.load_holdout(cfg, pool=None)
        check("holdout without a pool keeps every row", len(raw_holdout) == 3)

        # A missing/unset holdout split must degrade to None, not raise.
        cfg_missing = {**cfg, "data": {**cfg["data"], "holdout_split": None}}
        check("holdout_split=null returns None",
              dataset.load_holdout(cfg_missing, pool=pool) is None)
    finally:
        dataset._load_hf_split = orig_loader

    check("PARTITIONS lists every accepted source",
          set(dataset.PARTITIONS) == {"all", "train", "val", "test", "holdout"})


# --------------------------------------------------------------------------
# 5. Env: .env parsing and HF token lookup
# --------------------------------------------------------------------------
def test_env() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / ".env"
        env_file.write_text(
            "# a comment\n"
            "\n"
            "TEST_PLAIN=plain_value\n"
            'HF_TOKEN="hf_from_file"\n'
            "MALFORMED_NO_EQUALS\n",
            encoding="utf-8",
        )
        os.environ.pop("TEST_PLAIN", None)
        os.environ.pop("HF_TOKEN", None)

        orig_root = utils.PROJECT_ROOT
        utils.PROJECT_ROOT = Path(tmp)
        try:
            check("get_hf_token reads the .env file",
                  utils.get_hf_token() == "hf_from_file")
            check("load_env strips quotes", os.environ.get("HF_TOKEN") == "hf_from_file")
            check("load_env skips malformed lines", "MALFORMED_NO_EQUALS" not in os.environ)
        finally:
            utils.PROJECT_ROOT = orig_root
            os.environ.pop("TEST_PLAIN", None)
            os.environ.pop("HF_TOKEN", None)


# --------------------------------------------------------------------------
# 6. Task 3: instruction formatting + training helpers (fake tokenizer)
# --------------------------------------------------------------------------
class _FakeTokenizer:
    """Whitespace tokenizer with a chat template just rich enough to exercise
    instruction_format's masking logic - no model, no network."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _id(self, tok: str) -> int:
        return self._vocab.setdefault(tok, len(self._vocab) + 2)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(t) for t in str(text).split()]}

    def decode(self, ids, skip_special_tokens=True):
        rev = {v: k for k, v in self._vocab.items()}
        return " ".join(rev.get(i, "") for i in ids).strip()

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, return_dict=True, **kw):
        parts = [f"<{m['role']}> {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        ids = [self._id(t) for t in " ".join(parts).split()]
        return {"input_ids": ids} if return_dict else ids


def test_instruction_format() -> None:
    tok = _FakeTokenizer()

    ex = instruction_format.build_supervised_example(
        "great movie loved it", "positive", tok, max_review_tokens=50, max_seq_len=400
    )
    n_masked = ex["labels"].count(-100)
    target_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    check("supervised example masks the prompt with -100", n_masked > 0)
    check("supervised labels align with input_ids", len(ex["labels"]) == len(ex["input_ids"]))
    check("supervised target decodes to the assistant answer",
          tok.decode(target_ids) == instruction_format.target_text("positive"))

    prompt_ids = instruction_format.build_prompt_ids("great movie", tok, max_review_tokens=50)
    full_ex = instruction_format.build_supervised_example("great movie", "positive", tok, 50, 400)
    check("inference prompt is a strict prefix of the supervised example, minus the answer",
          full_ex["input_ids"][: len(prompt_ids)] == prompt_ids
          and len(full_ex["input_ids"]) > len(prompt_ids))

    long_review = " ".join(["word"] * 500)
    trunc = instruction_format._truncate_review(long_review, tok, 20)
    check("long review truncated to max_review_tokens", len(trunc.split()) == 20)
    check("review under the cap is untouched",
          instruction_format._truncate_review("short one", tok, 20) == "short one")

    check("example over max_seq_len is dropped (returns None)",
          instruction_format.build_supervised_example(
              long_review, "negative", tok, max_review_tokens=400, max_seq_len=50) is None)

    check("target_text rejects an unknown label",
          _raises(lambda: instruction_format.target_text("neutral")))


def test_train_task3_helpers() -> None:
    tok = _FakeTokenizer()
    ex_a = instruction_format.build_supervised_example("a b c", "positive", tok, 50, 400)
    ex_b = instruction_format.build_supervised_example("longer review here now", "negative", tok, 50, 400)
    batch = train_task3.CausalDataCollator(tok.pad_token_id)([ex_a, ex_b])
    n = max(len(ex_a["input_ids"]), len(ex_b["input_ids"]))
    check("collator pads to the longest sequence in the batch",
          tuple(batch["input_ids"].shape) == (2, n))
    check("collator pads labels with -100, inputs with pad_token_id",
          int((batch["labels"] == -100).sum()) >= n - len(ex_a["input_ids"])
          and int((batch["input_ids"][0] == tok.pad_token_id).sum()) == n - len(ex_a["input_ids"]))
    check("collator attention mask is 1 on real tokens, 0 on padding",
          int(batch["attention_mask"][0].sum()) == len(ex_a["input_ids"]))

    log = [
        {"loss": 0.5, "learning_rate": 2e-4, "step": 10, "epoch": 0.1},
        {"eval_loss": 0.4, "eval_macro_f1": 0.8, "step": 10, "epoch": 0.1},
        {"loss": 0.3, "learning_rate": 1e-4, "step": 20, "epoch": 0.2},
    ]
    tidy = train_task3.tidy_log_history(log)
    row10 = tidy[tidy["step"] == 10].iloc[0]
    check("tidy_log_history merges train+eval records for the same step",
          row10["train_loss"] == 0.5 and row10["eval_macro_f1"] == 0.8)
    check("tidy_log_history keeps one row per step, sorted",
          list(tidy["step"]) == [10, 20])


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:  # noqa: BLE001
        return True


# --------------------------------------------------------------------------
def main() -> int:
    test_parser()
    test_prompts()
    test_prompts_carp()
    test_prompts_basic()
    test_fewshot()
    test_metrics()
    test_eval_task2()
    test_prompt_sensitivity()
    test_self_consistency()
    test_dataset()
    test_env()
    test_instruction_format()
    test_train_task3_helpers()

    width = max(len(n) for _, n, _ in _results)
    for status, name, detail in _results:
        line = f"[{status}] {name.ljust(width)}"
        if status == FAIL:
            line += f"  <- {detail}"
        print(line)

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
