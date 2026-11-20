#!/usr/bin/env python3
"""CS5491 milestone runner for context-aware NSC (CA-NSC).

Run from this folder (Python 3.9+, standard library only):

    python3 run_milestone.py [--repeats 5]

Steps
  1. randomized property checks of the reweighting   -> results/checks_synthetic.json
  2. hand-computable FFN reordering example           -> results/toy_reordering.{json,md}
  3. course starter availability                      -> results/status.{json,md}
  4. if the starter is bound (cansc/course_adapter.py):
       a. freeze the alpha grid from the development configurations alone
          -> results/protocol_frozen.json (never overwritten)
       b. load development labels and evaluate the official development split
          -> results/dev_results.json, dev_scores.csv, main_table.md,
             size_control.md, reversal_example.md
       c. cross-check with the unmodified course evaluator and its reference
          output -> results/course_evaluator_development.json, crosscheck.md
  5. numbers for the LaTeX report                     -> report/generated/*.tex

The final split is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cansc import checks, course_adapter, protocol, report_tex, toy
from cansc.evaluate import evaluate_development
from cansc.records import DEVELOPMENT

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GENERATED = HERE / "report" / "generated"
EXPECTED_DEV_SIZE = 100
EXPECTED_DEV_PAIRS = 180  # starter README, "Evaluation protocol" item 3

BLOCKED_RESULTS = (
    "course NSC scores of the development architectures (direct call to the course scorer)",
    "alpha = 0 versus course NSC on every development architecture (maximum absolute error)",
    "first-layer weight and single-layer checks on the real development architectures",
    "unlabelled C_l distribution of the development split and the numeric alpha grid",
    "Spearman rho of course NSC and of every alpha, and the selected alpha",
    "NSC score ranges, scoring time, FFN weight minimum / median",
    "rank change between course NSC and the selected alpha",
    "size-control pairs (same width, parameter ratio >= 0.90), their verdicts and a reversal case",
)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def _alpha(a) -> str:
    return "--" if a is None else ("0" if a == 0 else f"{a:g}")


def main_table_md(res: dict, frozen: dict) -> str:
    """Markdown version of the main development table and the checks."""
    sel = res["selected_alpha"]
    q = frozen["c_quantiles"]
    lines = [
        f"# Development split results (n = {res['n']}, official split; final split not used)",
        "",
        f"Frozen grid: {', '.join(_alpha(a) for a in frozen['grid'])} (rule {frozen['rule_version']}; "
        f"C_ref = {frozen['c_ref']:.4f}, the median of {frozen['n_pooled_layers']} unlabelled C_l values, "
        f"layers l >= 2; C_l quantiles min {q['min']:.3f}, q25 {q['q25']:.3f}, q75 {q['q75']:.3f}, "
        f"max {q['max']:.3f}).",
        f"Timing: {res['timing']}.",
        "Spearman and Kendall use scores canonicalised to 12 significant digits (course evaluator); "
        "score ranges are unrounded.",
        "",
        "| method | alpha | n | Spearman (score, -ppl) | Kendall tau-b | score min | score max | time (ms) "
        "| FFN w min | FFN w median | retained FFN (median) | FFN share of score (median) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        star = " (selected)" if r["alpha"] is not None and r["alpha"] == sel else ""
        w = lambda k: f"{r[k]:.4f}" if r.get(k) is not None else "--"
        lines.append(
            f"| {r['method']}{star} | {_alpha(r['alpha'])} | {r['n']} | {r['spearman']:.6f} | "
            f"{r['kendall_tau_b']:.6f} | {r['score_min']:.6g} | {r['score_max']:.6g} | {r['time_ms']:.1f} | "
            f"{w('w_min')} | {w('w_median')} | {w('retained_median')} | {w('ffn_share_median')} |")
    rc = res["rank_change_course_vs_selected"]
    chk = res["checks"]
    ident = chk["alpha0_vs_course"]
    lines += [
        "",
        "FFN weight statistics: all (architecture, layer l >= 2) pairs of the development split; "
        "retained FFN = sum_l w_l F_l / sum_l F_l per architecture.",
        "",
        f"Selected alpha (max unrounded Spearman, ties to smaller alpha): **{_alpha(sel)}**.",
        f"Rank change course -> selected (canonical scores): Spearman between scores "
        f"{rc['spearman_between']:.6f}; {rc['items_rank_changed']} architectures changed rank; "
        f"mean |shift| {rc['mean_abs_rank_shift']:.3f}; max |shift| {rc['max_abs_rank_shift']:.1f}; "
        f"{rc['reversed_pairs']} of {rc['total_pairs']} pairs reversed.",
        "",
        "## Correctness checks on the development architectures",
        f"- alpha = 0 vs course NSC (`nsc.original_nsc`): max abs error {ident['max_abs_err']:.3e} "
        f"(bitwise equal for {ident['n_bitwise_equal']} of {ident['n']}); tolerance {ident['tolerance']}; "
        f"failures {ident['n_fail']}.",
    ]
    if "per_layer_decomposition" in chk:
        pl = chk["per_layer_decomposition"]
        lines.append(f"- A_l + F_l vs `nsc.layer_capacities`: {pl['n_layers']} layers, "
                     f"{pl['n_not_bitwise_equal']} not bitwise equal, max abs error {pl['max_abs_err']:.3e}.")
    lines.append(f"- first-layer weight exactly 1 in all {chk['first_layer_weight']['n_checked']} "
                 f"(architecture, alpha) cases: {chk['first_layer_weight']['all_exactly_one']}.")
    sa = chk["single_layer_archived"]
    lines.append(f"- archived single-layer architectures: {sa['n_single_layer_architectures']}.")
    if "single_layer_constructed" in chk:
        sc = chk["single_layer_constructed"]
        lines.append(f"- one-block copies ({sc['construction']}): {sc['n_configs']} configurations x "
                     f"alphas {', '.join(_alpha(a) for a in sc['alphas'])}: max abs error "
                     f"{sc['max_abs_err']:.3e}; all bitwise equal: {sc['all_bitwise_equal']}.")
    diffs = [r["spearman_crosscheck_abs_diff"] for r in res["rows"]]
    lines.append(f"- own Spearman implementation vs course `ranks`/`correlation`: max abs difference "
                 f"{max(diffs):.1e}.")
    return "\n".join(lines) + "\n"


def size_control_md(res: dict) -> str:
    sc = res["size_control"]
    lines = [
        "# Size-control check (development split)",
        "",
        f"Rule: {sc['rule']}. Qualifying pairs: **{sc['n_pairs']}**; pairs with identical recorded "
        f"perplexity (truth ties, excluded from the counts below): {sc['course']['truth_tie']}.",
        "Scores canonicalised to 12 significant digits, as in the course evaluator.",
        "",
        "| method | correct | wrong | score tie | tau-b (course pair_summary) |",
        "|---|---|---|---|---|",
        f"| NSC (course) | {sc['course']['correct']} | {sc['course']['wrong']} | {sc['course']['score_tie']} "
        f"| {sc['official_course']['tau_b']:.6f} |",
        f"| CA-NSC (alpha = {_alpha(res['selected_alpha'])}) | {sc['selected']['correct']} | "
        f"{sc['selected']['wrong']} | {sc['selected']['score_tie']} | {sc['official_selected']['tau_b']:.6f} |",
        "",
        f"Counts equal the course `pair_summary` counts: {sc['counts_match_official']}.",
        "",
        "Verdict transitions course -> selected: " + ", ".join(f"{k}: {v}" for k, v in sc["transitions"].items()),
        "",
        f"Pairs that the selected alpha reverses from correct to wrong: {sc['n_reversed_correct_to_wrong']}"
        + (f" ({'; '.join(' vs '.join(p) for p in sc['reversals'])})." if sc["reversals"] else "."),
        "",
        "## Pairs whose verdict changed (better = lower perplexity)",
        "",
        "| transition | better | worse | depth b/w | attention b/w | raw FFN b/w | ppl b/w |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in sc["changed_pairs"]:
        lines.append(
            f"| {p['transition']} | {p['better']} | {p['worse']} | {p['depth_better']}/{p['depth_worse']} | "
            f"{p['attention_better']:.2f}/{p['attention_worse']:.2f} | {p['ffn_better']:.2f}/{p['ffn_worse']:.2f} | "
            f"{p['ppl_better']:.3f}/{p['ppl_worse']:.3f} |")
    return "\n".join(lines) + "\n"


def reversal_md(res: dict) -> str:
    ex = res["size_control"]["reversal_example"]
    if not ex:
        return "# Reversal example\n\nNo qualifying pair was reversed from correct to wrong.\n"
    lines = [f"# Reversal example (alpha = {_alpha(res['selected_alpha'])}; largest perplexity gap among "
             "correct -> wrong pairs)", ""]
    for a in ex["architectures"]:
        lines += [
            f"## {a['arch_id']}: width {a['width']}, depth {a['depth']}, total params {a['params']:.0f}, "
            f"perplexity {a['perplexity']}",
            f"n_head {a['n_head']}; d_inner {a['d_inner']}",
            f"course NSC {a['course_nsc']:.6f}; CA-NSC {a['ca_nsc']:.6f}; FFN total {a['ffn_total']:.6f}; "
            f"FFN loss {a['ffn_loss']:.6f}; attention total {a['attention_total']:.6f}",
            "",
            "| layer | A_l | F_l | C_l | w_l | w_l F_l |",
            "|---|---|---|---|---|---|",
        ]
        lines += [f"| {int(r['layer'])} | {r['A']:.4f} | {r['F']:.4f} | {r['C']:.4f} | {r['w']:.4f} | "
                  f"{r['wF']:.4f} |" for r in a["layers"]]
        lines.append("")
    return "\n".join(lines)


def write_scores_csv(res: dict, path: Path) -> None:
    cols = [k for k in res["scores"] if k != "ids"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["arch_id"] + cols)
        for i, arch_id in enumerate(res["scores"]["ids"]):
            writer.writerow([arch_id] + [repr(res["scores"][c][i]) for c in cols])


def course_evaluator_crosscheck(res: dict) -> dict:
    """Run the unmodified course evaluator on CA-NSC and compare its numbers with ours."""
    starter = course_adapter.STARTER_DIR
    out = RESULTS / "course_evaluator_development.json"
    env = dict(os.environ, CANSC_ALPHA=repr(float(res["selected_alpha"])), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, str(starter / "evaluate.py"), "--split", DEVELOPMENT,
                           "--student", str(HERE / "student_ca_nsc.py"), "--output", str(out)],
                          cwd=HERE, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"course evaluator failed:\n{proc.stdout}\n{proc.stderr}")
    official = json.loads(out.read_text(encoding="utf-8"))["ranking"]
    expected = json.loads((starter / "expected_development.json").read_text(encoding="utf-8"))["ranking"]
    rows = {r["alpha"]: r for r in res["rows"][1:]}
    course_row = res["rows"][0]
    sel_row = rows[res["selected_alpha"]]
    sc = res["size_control"]

    def diff(a, b):
        return None if a is None or b is None else abs(a - b)

    out_rows = {
        "original_nsc_rho": diff(official["original_nsc"]["spearman_rho"], course_row["spearman"]),
        "ablation_rho_vs_course_row": diff(official["ablation"]["spearman_rho"], course_row["spearman"]),
        "ablation_rho_vs_alpha0_row": diff(official["ablation"]["spearman_rho"], rows[0.0]["spearman"]),
        "student_rho_vs_selected_row": diff(official["student"]["spearman_rho"], sel_row["spearman"]),
        "student_tau_vs_selected_row": diff(official["student"]["all_pairs"]["tau_b"], sel_row["kendall_tau_b"]),
    }
    official_pairs = official["original_nsc"]["size_controlled_pairs"]["pairs"]
    report = {
        "selected_alpha": res["selected_alpha"],
        "abs_differences": out_rows,
        "max_abs_difference": max(v for v in out_rows.values() if v is not None),
        "size_controlled_pairs_course_evaluator": official_pairs,
        "size_controlled_pairs_ours": sc["n_pairs"],
        "size_controlled_pairs_readme": EXPECTED_DEV_PAIRS,
        "student_size_controlled_equals_ours": {
            k: official["student"]["size_controlled_pairs"][k] == sc["official_selected"][k]
            for k in ("concordant", "discordant", "score_only_ties", "quality_only_ties", "both_ties")},
        "original_nsc_equals_reference_output": official["original_nsc"] == expected["original_nsc"],
        "reference_original_nsc_rho": expected["original_nsc"]["spearman_rho"],
        "note": "course_evaluator_development.json also holds the evaluator's budgeted-selection output; "
                "it is not analysed in the milestone.",
    }
    return report


def course_reorder_pair(alpha: float) -> dict:
    """The starter README's reorder pair, scored with the course capacities.

    d = 128, H = [4, 4], FFN widths [256, 512] versus [512, 256]: NSC ties by
    construction; the ablation (alpha = 0) must reproduce that tie, and CA-NSC
    at alpha > 0 must prefer the larger FFN first (property (vi)).
    """
    from cansc.context_nsc import context_aware_score
    nsc = course_adapter.course_nsc()
    rows = []
    for inner in ([256, 512], [512, 256]):
        cfg = {"d_model": 128, "n_layer": 2, "n_head": [4, 4], "d_inner": list(inner)}
        terms = course_adapter.terms_from_config(f"toy{inner}", cfg)
        rows.append({"d_inner": inner, "original_nsc": nsc.original_nsc(cfg),
                     "ablation": context_aware_score(terms, 0.0),
                     "ca_nsc": context_aware_score(terms, alpha),
                     "position_example": nsc.position_example(cfg)})
    back, front = rows  # [256, 512] puts the larger FFN last, [512, 256] first
    return {"alpha": alpha, "rows": rows,
            "nsc_tie": back["original_nsc"] == front["original_nsc"],
            "ablation_equals_nsc": all(r["ablation"] == r["original_nsc"] for r in rows),
            "ca_prefers_larger_first": front["ca_nsc"] > back["ca_nsc"],
            "position_prefers_larger_last": back["position_example"] > front["position_example"]}


def crosscheck_md(cc: dict) -> str:
    lines = ["# Cross-check with the unmodified course evaluator", "",
             f"`CANSC_ALPHA={cc['selected_alpha']!r} python3 starter/evaluate.py --split development "
             f"--student student_ca_nsc.py` (score = CA-NSC at the selected alpha, ablation = alpha 0).", ""]
    lines += [f"- |{k}|: {'n/a' if v is None else f'{v:.1e}'}" for k, v in cc["abs_differences"].items()]
    lines += [
        f"- size-controlled pairs: course evaluator {cc['size_controlled_pairs_course_evaluator']}, ours "
        f"{cc['size_controlled_pairs_ours']}, starter README {cc['size_controlled_pairs_readme']}",
        f"- student size-controlled counts equal ours: {cc['student_size_controlled_equals_ours']}",
        f"- original-NSC ranking block identical to expected_development.json: "
        f"{cc['original_nsc_equals_reference_output']} (reference rho {cc['reference_original_nsc_rho']:.6f})",
        "", cc["note"]]
    return "\n".join(lines) + "\n"


def blocked_md(missing) -> str:
    lines = ["# Development evaluation: BLOCKED", "",
             "The course starter is not available or not bound, so no development-split result exists yet.",
             "", "Missing:"] + [f"- {m}" for m in missing] + [
             "", "Results that cannot be produced until then:"] + [f"- {b}" for b in BLOCKED_RESULTS] + [
             "", "Bind cansc/course_adapter.py (see its docstring) and rerun `python3 run_milestone.py`."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats per method (default 5)")
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    chk = checks.run_synthetic_checks()
    write_json(RESULTS / "checks_synthetic.json", chk)
    print(f"[1] synthetic property checks: {chk['float_cases']} float + {chk['exact_cases']} exact cases, "
          f"{chk['total_violations']} violations")

    toy_res = toy.run()
    write_json(RESULTS / "toy_reordering.json", toy_res)
    (RESULTS / "toy_reordering.md").write_text(toy.to_markdown(toy_res), encoding="utf-8")
    print(f"[2] toy reordering: exact match with hand calculation = {toy_res['all_exact_match']}")
    for r in toy_res["rows"]:
        print(f"      order {tuple(r['order'])}, d={r['width']}: S(0)={r['S_alpha0']}, "
              f"S(1/2)={r['S_alpha']} = {r['S_alpha_float']:.4f}")

    missing = course_adapter.missing_items()
    status = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "missing": missing,
        "dev_evaluation": "blocked" if missing else "run",
        "blocked_results": list(BLOCKED_RESULTS) if missing else [],
        "final_split_used": False,
    }
    if missing:
        print("[3] course starter: MISSING -> development evaluation blocked")
        for m in missing:
            print(f"      - {m}")
        if (RESULTS / "main_table.md").exists():
            # A clone without the starter keeps the committed development results.
            print("      existing development results in results/ are left unchanged")
        else:
            (RESULTS / "main_table.md").write_text(blocked_md(missing), encoding="utf-8")
    else:
        print(f"[3] course starter: bound ({course_adapter._display(course_adapter.STARTER_DIR)}/)")
        # (a) grid from configurations only; no label file has been opened yet.
        unlabelled = course_adapter.unlabelled_terms(DEVELOPMENT)
        frozen = protocol.freeze_or_load(unlabelled, RESULTS / "protocol_frozen.json")
        q = frozen["c_quantiles"]
        print(f"[4a] alpha grid {'loaded' if frozen['loaded_existing'] else 'frozen'}: {frozen['grid']} "
              f"(C_ref = {frozen['c_ref']:.4f}; C_l from {q['min']:.3f} to {q['max']:.3f})")
        # (b) labels and evaluation.
        records = course_adapter.load_split(DEVELOPMENT)
        if len(records) != EXPECTED_DEV_SIZE:
            print(f"      WARNING: {len(records)} development architectures, expected {EXPECTED_DEV_SIZE}")
        res = evaluate_development(records, course_adapter.original_nsc, course_adapter.layer_terms,
                                   frozen["grid"], repeats=args.repeats, harness=course_adapter.harness(),
                                   reset_fn=course_adapter.clear_cache,
                                   layer_capacities_fn=course_adapter.layer_capacities,
                                   single_layer_fn=course_adapter.single_block)
        write_json(RESULTS / "dev_results.json", res)
        write_scores_csv(res, RESULTS / "dev_scores.csv")
        (RESULTS / "main_table.md").write_text(main_table_md(res, frozen), encoding="utf-8")
        (RESULTS / "size_control.md").write_text(size_control_md(res), encoding="utf-8")
        (RESULTS / "reversal_example.md").write_text(reversal_md(res), encoding="utf-8")
        ident = res["checks"]["alpha0_vs_course"]
        print(f"[4b] n = {res['n']}; alpha = 0 vs course NSC: max abs error {ident['max_abs_err']:.1e} "
              f"(passed = {ident['passed']}); selected alpha = {_alpha(res['selected_alpha'])}")
        for r in res["rows"]:
            print(f"      {r['method']:13s} alpha={_alpha(r['alpha']):>6s}  rho={r['spearman']:.6f}  "
                  f"w_min={r.get('w_min', float('nan')):.3f}")
        sc = res["size_control"]
        print(f"      size control: {sc['n_pairs']} pairs; course {sc['course']}; selected {sc['selected']}")
        # (c) cross-check with the unmodified course evaluator and its reference output.
        cc = course_evaluator_crosscheck(res)
        write_json(RESULTS / "crosscheck.json", cc)
        (RESULTS / "crosscheck.md").write_text(crosscheck_md(cc), encoding="utf-8")
        print(f"[4c] course evaluator cross-check: max abs difference {cc['max_abs_difference']:.1e}; "
              f"pairs {cc['size_controlled_pairs_course_evaluator']}; original NSC equals reference output = "
              f"{cc['original_nsc_equals_reference_output']}")
        # (d) the starter's own reorder pair (mechanism check with course capacities).
        pair = course_reorder_pair(float(res["selected_alpha"]))
        write_json(RESULTS / "course_reorder_pair.json", pair)
        lines = ["# Starter reorder pair (course capacities, d = 128, H = [4, 4])", "",
                 f"| d_inner | original NSC | ablation (alpha = 0) | CA-NSC (alpha = {pair['alpha']:g}) "
                 "| position example |", "|---|---|---|---|---|"]
        lines += [f"| {r['d_inner']} | {r['original_nsc']:.9f} | {r['ablation']:.9f} | {r['ca_nsc']:.9f} | "
                  f"{r['position_example']:.9f} |" for r in pair["rows"]]
        lines += ["", f"NSC tie: {pair['nsc_tie']}; ablation equals NSC: {pair['ablation_equals_nsc']}; "
                      f"CA-NSC prefers the larger FFN first: {pair['ca_prefers_larger_first']}; "
                      f"the positional example prefers it last: {pair['position_prefers_larger_last']}."]
        (RESULTS / "course_reorder_pair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[4d] starter reorder pair: NSC tie {pair['nsc_tie']}, ablation = NSC "
              f"{pair['ablation_equals_nsc']}, CA-NSC prefers larger FFN first {pair['ca_prefers_larger_first']}")
    write_json(RESULTS / "status.json", status)
    (RESULTS / "status.md").write_text(
        blocked_md(missing) if missing else "# Status\n\nDevelopment evaluation ran; see main_table.md.\n",
        encoding="utf-8")

    report_tex.write_generated(RESULTS, GENERATED)
    print(f"[5] report numbers written to {GENERATED.relative_to(HERE)}/")
    print(f"done in {time.perf_counter() - start:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
