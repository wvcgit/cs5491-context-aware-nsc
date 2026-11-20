"""Write the numbers used by report/milestone_report.tex.

Every number in the report comes from a file in results/, through the LaTeX
snippets written here to report/generated/. Missing results are rendered with
the report's ``\\Pending`` macro (red text), never with invented values. The
report preamble declares the switches ``\\ifdevresults``, ``\\ifselzero`` and
``\\ifreversal``; the generated macros only set them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

PENDING = r"\Pending"


def _load(path: Path) -> Optional[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _tex(s: object) -> str:
    """Escape LaTeX special characters in identifiers taken from the archive."""
    out = str(s)
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                       ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                       ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        out = out.replace(char, repl)
    return out


def _sci(x: float) -> str:
    mant, exp = f"{x:.2e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def _num(x, digits: int = 4) -> str:
    """Fixed-point number with ``digits`` decimals; tiny or huge values in scientific form."""
    if x is None:
        return PENDING
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}".replace(",", "{,}")
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return _sci(x)
    return f"{x:.{digits}f}"


def _sig4(x) -> str:
    """About four significant figures without an exponent when 1e-3 <= |x| < 1e7."""
    if x is None:
        return PENDING
    x = float(x)
    if x == 0:
        return "0"
    if not 1e-3 <= abs(x) < 1e7:
        return _sci(x)
    if x.is_integer():
        return f"{int(x):,}".replace(",", "{,}")
    decimals = max(0, 3 - int(math.floor(math.log10(abs(x)))))
    return f"{x:,.{decimals}f}".replace(",", "{,}")


def _pct(x) -> str:
    """Fraction as a whole percentage for LaTeX (0.4103 -> 41\\%)."""
    return PENDING if x is None else f"{100 * float(x):.0f}\\%"


def _ms(x: float) -> str:
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 1000 else f"{x:,.0f}".replace(",", "{,}"))


def _alpha(a) -> str:
    return "0" if float(a) == 0 else f"{float(a):g}"


def _cmd(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _switch(name: str, on: bool) -> str:
    return "\\" + name + ("true" if on else "false")


def _dev_macros(dev: Dict[str, object]) -> List[str]:
    rows = dev["rows"]
    course = rows[0]
    ca_rows = rows[1:]
    sel = next(r for r in ca_rows if r["alpha"] == dev["selected_alpha"])
    nonzero = [r for r in ca_rows if r["alpha"] != 0]
    best_nz = max(nonzero, key=lambda r: (r["spearman"], -r["alpha"])) if nonzero else None
    strong = max(nonzero, key=lambda r: r["alpha"]) if nonzero else None
    rc = dev["rank_change_course_vs_selected"]
    sc = dev["size_control"]
    ident = dev["checks"]["alpha0_vs_course"]
    chk = dev["checks"]
    alpha0 = next(r for r in ca_rows if r["alpha"] == 0)
    by_alpha = sorted(ca_rows, key=lambda r: r["alpha"])
    changed = sc.get("changed_pairs", [])
    fixed = [p for p in changed if p["transition"] == "wrong->correct"]
    broken = [p for p in changed if p["transition"] == "correct->wrong"]
    m = [_switch("selmax", dev["selected_alpha"] == max(r["alpha"] for r in ca_rows)),
         _cmd("RhoSecond", _num(by_alpha[-2]["spearman"]) if len(by_alpha) > 1 else PENDING),
         _cmd("RhoGainEdge", f"{by_alpha[-1]['spearman'] - by_alpha[-2]['spearman']:.4f}"
              if len(by_alpha) > 1 else PENDING),
         _cmd("RhoGainTotal", f"{sel['spearman'] - course['spearman']:.4f}"),
         _cmd("AlphaSecond", _alpha(by_alpha[-2]["alpha"]) if len(by_alpha) > 1 else PENDING),
         _cmd("TauCourse", _num(course["kendall_tau_b"])), _cmd("TauSel", _num(sel["kendall_tau_b"])),
         _cmd("TimeCourse", _ms(course["time_ms"])), _cmd("TimeSel", _ms(sel["time_ms"])),
         _cmd("FFNShareCourse", _pct(alpha0.get("ffn_share_median"))),
         _cmd("FFNShareSel", _pct(sel.get("ffn_share_median"))),
         _cmd("RetainedSel", _pct(sel.get("retained_median"))),
         _cmd("NFixed", str(len(fixed))), _cmd("NBroken", str(len(broken))),
         _cmd("NFixedAttnRich", str(sum(p["better_is_attention_rich"] for p in fixed))),
         # fixed pairs whose better model is attention-rich AND deeper (the sentence's subset)
         _cmd("NFixedDeeper", str(sum(p["better_is_attention_rich"] and p["depth_better"] > p["depth_worse"]
                                      for p in fixed))),
         _cmd("NBrokenAttnRich", str(sum(p["worse_is_attention_rich"] for p in broken))),
         _cmd("SizeTauCourse", _num(sc["official_course"]["tau_b"])),
         _cmd("SizeTauSel", _num(sc["official_selected"]["tau_b"])),
         _cmd("IdBitwise", str(ident["n_bitwise_equal"])),
         _cmd("FirstN", _num(chk["first_layer_weight"]["n_checked"])),
         _cmd("PerLayerN", _num(chk.get("per_layer_decomposition", {}).get("n_layers"))),
         _cmd("PerLayerMismatch", _num(chk.get("per_layer_decomposition", {}).get("n_not_bitwise_equal"))),
         _cmd("SingleN", _num(chk.get("single_layer_constructed", {}).get("n_configs"))),
         _cmd("SingleMaxErr", _num(chk.get("single_layer_constructed", {}).get("max_abs_err"))),
         _cmd("UnitsByWidth", ", ".join(f"{float(v):.2f} ($d={w}$)" for w, v in
                                        dev.get("ffn_capacity_per_width_median", {}).items()) or PENDING)]
    m += [_switch("selzero", dev["selected_alpha"] == 0),
         _cmd("DevN", _num(dev["n"])), _cmd("SelAlpha", _alpha(dev["selected_alpha"])),
         _cmd("RhoCourse", _num(course["spearman"])), _cmd("RhoSel", _num(sel["spearman"])),
         _cmd("WMinSel", _num(sel.get("w_min"), 3)), _cmd("WMedSel", _num(sel.get("w_median"), 3)),
         _cmd("RankChanged", _num(rc["items_rank_changed"])),
         _cmd("MeanShift", _num(rc["mean_abs_rank_shift"], 2)),
         _cmd("MaxShift", _sig4(rc["max_abs_rank_shift"])),
         _cmd("RevPairs", _num(rc["reversed_pairs"])), _cmd("TotPairs", _num(rc["total_pairs"])),
         _cmd("NPairs", _num(sc["n_pairs"])), _cmd("TruthTies", _num(sc["course"]["truth_tie"])),
         _cmd("NReversed", _num(sc["n_reversed_correct_to_wrong"])),
         _cmd("NReversedPairs", f"{sc['n_reversed_correct_to_wrong']} pair"
              + ("" if sc["n_reversed_correct_to_wrong"] == 1 else "s")),
         _cmd("IdMaxErr", _num(ident["max_abs_err"])),
         _cmd("IdPassed", "passed" if ident["passed"] else r"\textbf{failed}")]
    if best_nz:
        m += [_cmd("BestNZAlpha", _alpha(best_nz["alpha"])), _cmd("BestNZRho", _num(best_nz["spearman"])),
              _cmd("StrongAlpha", _alpha(strong["alpha"])), _cmd("StrongRho", _num(strong["spearman"])),
              _cmd("StrongWMin", _num(strong.get("w_min"), 3)),
              _cmd("StrongWMed", _num(strong.get("w_median"), 3)),
              _cmd("StrongRetained", _num(strong.get("retained_median"), 3))]
    else:
        m += [_cmd(n, PENDING) for n in ("BestNZAlpha", "BestNZRho", "StrongAlpha", "StrongRho",
                                         "StrongWMin", "StrongWMed", "StrongRetained")]
    ex = sc.get("reversal_example")
    m.append(_switch("reversal", bool(ex)))
    if ex:
        a, b = ex["architectures"]
        better, worse = (a, b) if a["perplexity"] < b["perplexity"] else (b, a)
        kept_b = better["ffn_total"] - better["ffn_loss"]
        kept_w = worse["ffn_total"] - worse["ffn_loss"]
        m += [_cmd("RevBetter", r"\texttt{%s}" % _tex(better["arch_id"])),
              _cmd("RevWorse", r"\texttt{%s}" % _tex(worse["arch_id"])),
              _cmd("RevWidth", _sig4(better["width"])),
              _cmd("RevParamsBetter", f"{better['params'] / 1e6:.2f}M"),
              _cmd("RevParamsWorse", f"{worse['params'] / 1e6:.2f}M"),
              _cmd("RevFFNGap", f"{better['ffn_total'] - worse['ffn_total']:.1f}"),
              _cmd("RevAttGap", f"{worse['attention_total'] - better['attention_total']:.1f}"),
              _cmd("RevKeptGap", f"{kept_b - kept_w:.1f}"),
              _cmd("RevDepthBetter", str(better["depth"])), _cmd("RevDepthWorse", str(worse["depth"])),
              _cmd("RevPplBetter", _sig4(better["perplexity"])), _cmd("RevPplWorse", _sig4(worse["perplexity"])),
              _cmd("RevCourseBetter", _sig4(better["course_nsc"])), _cmd("RevCourseWorse", _sig4(worse["course_nsc"])),
              _cmd("RevCABetter", _sig4(better["ca_nsc"])), _cmd("RevCAWorse", _sig4(worse["ca_nsc"])),
              _cmd("RevLossBetter", _sig4(better["ffn_loss"])), _cmd("RevLossWorse", _sig4(worse["ffn_loss"]))]
    return m


def _main_rows(dev: Optional[Dict[str, object]]) -> List[str]:
    if not dev:
        p = r"\multicolumn{6}{c}{\Pending[requires the course archive]}"
        return [rf"NSC (course) & -- & {p} \\", rf"CA-NSC (ablation) & 0 & {p} \\",
                rf"CA-NSC & $\alpha_1,\dots,\alpha_4$ & {p} \\"]
    lines = []
    for r in dev["rows"]:
        is_course = r["alpha"] is None
        mark = r"$^\star$" if not is_course and r["alpha"] == dev["selected_alpha"] else ""
        lines.append(
            f"{'NSC (course)' if is_course else 'CA-NSC'}{mark} & {'--' if is_course else _alpha(r['alpha'])} & "
            f"{r['n']} & {r['spearman']:.4f} & {_sig4(r['score_min'])}--{_sig4(r['score_max'])} & "
            f"{_ms(r['time_ms'])} & {'--' if is_course else _num(r.get('w_min'), 3)} & "
            f"{'--' if is_course else _num(r.get('w_median'), 3)} \\\\")
    return lines


def _size_rows(dev: Optional[Dict[str, object]]) -> List[str]:
    if not dev:
        p = r"\multicolumn{4}{c}{\Pending[requires the course archive]}"
        return [rf"NSC (course) & {p} \\", rf"CA-NSC (selected $\alpha$) & {p} \\"]
    sc = dev["size_control"]
    c, s = sc["course"], sc["selected"]
    return [f"NSC (course) & {c['correct']} & {c['wrong']} & {c['score_tie']} & "
            f"{sc['official_course']['tau_b']:.4f} \\\\",
            f"CA-NSC ($\\alpha={_alpha(dev['selected_alpha'])}$) & {s['correct']} & {s['wrong']} & "
            f"{s['score_tie']} & {sc['official_selected']['tau_b']:.4f} \\\\"]


def write_generated(results_dir: Path, out_dir: Path) -> None:
    """Regenerate report/generated/*.tex from results/*.json."""
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = _load(results_dir / "checks_synthetic.json")
    toy = _load(results_dir / "toy_reordering.json")
    status = _load(results_dir / "status.json") or {}
    # Real-data results are used only if the latest run actually evaluated the
    # development split, so stale files never reach the report.
    current = status.get("dev_evaluation") == "run"
    dev = _load(results_dir / "dev_results.json") if current else None
    frozen = _load(results_dir / "protocol_frozen.json") if current else None

    m = [r"% Generated by run_milestone.py from results/ -- do not edit by hand.",
         _switch("devresults", bool(dev))]
    if checks:
        m += [_cmd("SynthCases", _num(checks["float_cases"])),
              _cmd("SynthExactCases", _num(checks["exact_cases"])),
              _cmd("SynthSeed", str(checks["seed"])),
              _cmd("SynthViolations", _num(checks["total_violations"])),
              _cmd("SynthNAlphas", _num(len(checks["alphas"]))),
              _cmd("SynthOrderErr", _num(checks["max_errors"]["alpha0_order_rel"])),
              _cmd("SynthLossErr", _num(checks["max_errors"]["loss_identity_rel"]))]
    if toy:
        m += [_cmd("ToyAllMatch", "yes" if toy["all_exact_match"] else r"\textbf{NO}"),
              _cmd("ToyFloatErr", _num(toy["max_float_abs_err"]))]
    if frozen:
        nonzero = [a for a in frozen["grid"] if a != 0]
        m += [_cmd("GridValues", ", ".join(_alpha(a) for a in frozen["grid"])),
              _cmd("CRef", _sig4(frozen["c_ref"])),
              _cmd("NPooled", str(frozen["n_pooled_layers"])),
              _cmd("CMinVal", f"{frozen['c_quantiles']['min']:.2f}"),
              _cmd("CMaxVal", f"{frozen['c_quantiles']['max']:.1f}"),
              _cmd("WeightsAtCRef", ", ".join(f"{frozen['weight_at_c_ref'][str(a)]:.2f}" for a in nonzero))]
    else:
        m += [_cmd("GridValues", r"\Pending[grid instantiated from development $C_l$]")]
        m += [_cmd(n, PENDING) for n in ("CRef", "NPooled", "CMinVal", "CMaxVal", "WeightsAtCRef")]
    cross = _load(results_dir / "crosscheck.json") if current else None
    if dev:
        m += _dev_macros(dev)
    else:
        m += [_cmd(n, PENDING) for n in (
            "DevN", "SelAlpha", "RhoCourse", "RhoSel", "WMinSel", "WMedSel", "RankChanged", "MeanShift",
            "MaxShift", "RevPairs", "TotPairs", "NPairs", "TruthTies", "NReversed", "NReversedPairs",
            "IdMaxErr", "IdPassed",
            "BestNZAlpha", "BestNZRho", "StrongAlpha", "StrongRho", "StrongWMin", "StrongWMed",
            "StrongRetained", "RhoSecond", "RhoGainEdge", "RhoGainTotal", "AlphaSecond", "TauCourse",
            "TauSel", "TimeCourse", "TimeSel",
            "FFNShareCourse", "FFNShareSel", "RetainedSel", "NFixed", "NBroken", "NFixedAttnRich",
            "NFixedDeeper", "NBrokenAttnRich", "SizeTauCourse", "SizeTauSel", "IdBitwise", "FirstN",
            "PerLayerN", "PerLayerMismatch", "SingleN", "SingleMaxErr", "UnitsByWidth")]
    pair = _load(results_dir / "course_reorder_pair.json") if current else None
    if pair:
        back, front = pair["rows"]
        m += [_cmd("PairNSC", f"{back['original_nsc']:.6f}"),
              _cmd("PairCAFront", f"{front['ca_nsc']:.4f}"), _cmd("PairCABack", f"{back['ca_nsc']:.4f}"),
              _cmd("PairPosFront", f"{front['position_example']:.4f}"),
              _cmd("PairPosBack", f"{back['position_example']:.4f}"),
              _cmd("PairAlpha", _alpha(pair["alpha"])),
              _cmd("PairAblationOK", "reproduces this tie exactly" if pair["ablation_equals_nsc"]
                   else r"\textbf{does not reproduce the tie}")]
    else:
        m += [_cmd(n, PENDING) for n in ("PairNSC", "PairCAFront", "PairCABack", "PairPosFront",
                                         "PairPosBack", "PairAlpha", "PairAblationOK")]
    if cross:
        m += [_cmd("CrossMaxDiff", _num(cross["max_abs_difference"])),
              _cmd("CrossPairs", str(cross["size_controlled_pairs_course_evaluator"])),
              _cmd("RefEqual", "identical" if cross["original_nsc_equals_reference_output"]
                   else r"\textbf{different}")]
    else:
        m += [_cmd(n, PENDING) for n in ("CrossMaxDiff", "CrossPairs", "RefEqual")]
    (out_dir / "macros.tex").write_text("\n".join(m) + "\n", encoding="utf-8")

    if toy:
        toy_lines = [
            f"({', '.join(map(str, r['order']))}) & {r['width']} & $({', '.join(r['C'])})$ & "
            f"$({', '.join(r['w'])})$ & ${r['sum_wF']}$ & ${r['S_alpha0']}$ & ${r['hand_S_alpha']}$ & "
            f"{r['S_alpha_float']:.4f} \\\\" for r in toy["rows"]]
        (out_dir / "toy_rows.tex").write_text("\n".join(toy_lines) + "\n", encoding="utf-8")
    (out_dir / "main_rows.tex").write_text("\n".join(_main_rows(dev)) + "\n", encoding="utf-8")
    (out_dir / "size_rows.tex").write_text("\n".join(_size_rows(dev)) + "\n", encoding="utf-8")
