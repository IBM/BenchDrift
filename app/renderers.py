"""
BenchDrift App — HTML renderers.

All HTML rendering for the terminal-aesthetic UI lives here:
analysis cards, result cards, baseline, summary, drift chart.
"""

import re
from typing import Dict, List, Optional, Set

from benchdrift.pipeline.feature_relevance import (
    ALL_AXIS_NAMES,
    TRANSFORMATION_TO_AXIS,
    distribute_slots,
    LLM_FEATURE_NAMES,
    TAXONOMY,
    WITHIN_AXIS_AFFINITY,
    _get_valid_axes,
    _rank_axes_by_features,
)
from benchdrift.pipeline.unified_variation_engine_batched import UnifiedVariationEngine

ALL_TRANSFORMATION_TYPES = UnifiedVariationEngine.get_all_transformation_types()

# Ordered list for the UI checkbox group
ALL_AXES_LIST = [
    "linguistic", "referential", "pragmatic", "structural",
    "constraint_targeted", "persona", "long_context",
]
DEFAULT_AXES_LIST = [
    "linguistic", "referential", "pragmatic", "structural", "constraint_targeted",
]


def _filter_types_by_axes(enabled_axes: Set[str]) -> dict:
    """Return transformation types dict filtered to only the given axes."""
    return {k: v for k, v in ALL_TRANSFORMATION_TYPES.items()
            if TRANSFORMATION_TO_AXIS.get(k) in enabled_axes}


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Axis-feature map (built once at import time)
# ---------------------------------------------------------------------------

def _build_axis_feature_map() -> Dict[str, List[str]]:
    axis_features = {}
    for axis_name, axis_def in TAXONOMY.items():
        feats = set(axis_def.get("affinity_features", []))
        for trans in axis_def.get("transformations", []):
            for f in WITHIN_AXIS_AFFINITY.get(trans, []):
                feats.add(f)
        axis_features[axis_name] = sorted(feats)
    return axis_features


AXIS_FEATURE_MAP = _build_axis_feature_map()


# ---------------------------------------------------------------------------
# Analysis panel
# ---------------------------------------------------------------------------

def render_analysis(features: dict, llm_features: Optional[dict] = None,
                    discovered_features: Optional[dict] = None,
                    ranked_axes: Optional[List[str]] = None,
                    enabled_axes: Optional[set] = None,
                    top_k: int = 8,
                    ref_candidates: Optional[List[dict]] = None) -> str:
    """Render unified axis-grouped analysis."""
    if not features:
        return ""

    valid_axes = _get_valid_axes(features, enabled_axes=enabled_axes)
    if ranked_axes:
        ordered = [a for a in ranked_axes if a in valid_axes]
        for a in valid_axes:
            if a not in ordered:
                ordered.append(a)
    else:
        ordered = _rank_axes_by_features(features, valid_axes)

    slots = distribute_slots(ordered, top_k)

    all_feats = dict(features)
    if llm_features:
        all_feats.update(llm_features)
    if discovered_features:
        all_feats.update(discovered_features)

    rendered_features = set()
    html_parts = []

    for rank_idx, axis_name in enumerate(ordered):
        axis_def = TAXONOMY.get(axis_name, {})
        desc = axis_def.get("description", "")
        transforms = axis_def.get("transformations", [])
        n_slots = slots.get(axis_name, 0)
        rank_num = rank_idx + 1

        rank_color = "#60a5fa" if rank_num <= 2 else ("#93c5fd" if rank_num <= 4 else "#555")
        slot_label = f"{n_slots} slot{'s' if n_slots != 1 else ''}" if n_slots > 0 else "0 slots"

        card = f'<div class="axis-group">'
        card += f'<div class="axis-group-header">'
        card += f'<span class="axis-group-rank" style="color:{rank_color}">#{rank_num}</span>'
        card += f'<span class="axis-group-name">{axis_name}</span>'
        card += f'<span class="axis-group-slots">{slot_label}</span>'
        card += f'</div>'
        card += f'<div class="axis-group-desc">{_esc(desc)}</div>'

        if axis_name == "referential" and ref_candidates:
            composites = [c for c in ref_candidates if c.get('is_composite') or c.get('merged_from')]
            individuals = [c for c in ref_candidates if not c.get('is_composite') and not c.get('merged_from')]

            ref_html = '<div class="ref-entities">'
            if composites:
                ref_html += '<div><span class="ref-label">clusters (varied together):</span> '
                for c in composites:
                    topic = c.get('topic', '')
                    ref_html += (
                        f'<span class="ref-fragment">'
                        f'{_esc(c["text"])}'
                        f'<span class="ref-fragment-type">{topic}</span>'
                        f'</span>'
                    )
                ref_html += '</div>'

            if individuals:
                by_topic: Dict[str, List[str]] = {}
                for c in individuals:
                    key = c.get('topic', c.get('domain', 'other'))
                    by_topic.setdefault(key, []).append(c['text'])
                for topic, texts in by_topic.items():
                    ref_html += f'<div style="margin-top:2px"><span class="ref-label">{topic}:</span> '
                    seen = set()
                    for t in texts:
                        if t not in seen:
                            seen.add(t)
                            ref_html += f'<span class="ref-entity">{_esc(t)}</span>'
                    ref_html += '</div>'

            if not composites and not individuals:
                ref_html += '<span style="color:#555">no entities detected</span>'

            ref_html += '</div>'
            card += ref_html
        else:
            axis_feats = AXIS_FEATURE_MAP.get(axis_name, [])
            badges = []
            for feat_name in axis_feats:
                active = bool(all_feats.get(feat_name, False))
                if discovered_features and feat_name in discovered_features:
                    cls = "badge-disc-on" if active else "badge-disc-off"
                    source = "disc"
                elif feat_name in LLM_FEATURE_NAMES:
                    if llm_features is None:
                        cls = "badge-llm-off"
                        source = "llm"
                    else:
                        cls = "badge-llm-on" if active else "badge-llm-off"
                        source = "llm"
                else:
                    cls = "badge-on" if active else "badge-off"
                    source = "regex"

                badges.append(
                    f'<span class="feature-badge {cls}" '
                    f'data-feature="{feat_name}" data-source="{source}" '
                    f'onclick="toggleFeature(this)">{feat_name}</span>'
                )
                rendered_features.add(feat_name)

            if badges:
                card += f'<div class="axis-group-features">'
                card += f'<div class="feature-badges">{"".join(badges)}</div>'
                card += f'</div>'

        if transforms:
            available = set(ALL_TRANSFORMATION_TYPES.keys())
            trans_parts = []
            for t in transforms:
                if t in available:
                    trans_parts.append(t)
                else:
                    trans_parts.append(f'<span style="text-decoration:line-through">{t}</span>')
            card += f'<div class="axis-group-transforms">transforms: {", ".join(trans_parts)}</div>'

        card += f'</div>'
        html_parts.append(card)

    # Unassigned features
    all_bool_feats = {k: v for k, v in all_feats.items() if isinstance(v, bool)}
    unassigned = {k: v for k, v in all_bool_feats.items() if k not in rendered_features}
    if unassigned:
        ub = []
        for feat_name, active in sorted(unassigned.items()):
            if discovered_features and feat_name in discovered_features:
                cls = "badge-disc-on" if active else "badge-disc-off"
                source = "disc"
            elif feat_name in LLM_FEATURE_NAMES:
                cls = "badge-llm-on" if active else "badge-llm-off"
                source = "llm"
            else:
                cls = "badge-on" if active else "badge-off"
                source = "regex"
            ub.append(
                f'<span class="feature-badge {cls}" '
                f'data-feature="{feat_name}" data-source="{source}" '
                f'onclick="toggleFeature(this)">{feat_name}</span>'
            )
        html_parts.append(
            f'<div class="unassigned-section">'
            f'<div class="badge-section-label">unassigned features</div>'
            f'<div class="feature-badges">{"".join(ub)}</div>'
            f'</div>'
        )

    if discovered_features:
        disc_not_shown = {k: v for k, v in discovered_features.items()
                          if k not in rendered_features and k not in unassigned}
        if disc_not_shown:
            db = []
            for feat_name, active in sorted(disc_not_shown.items()):
                cls = "badge-disc-on" if active else "badge-disc-off"
                db.append(
                    f'<span class="feature-badge {cls}" '
                    f'data-feature="{feat_name}" data-source="disc" '
                    f'onclick="toggleFeature(this)">{feat_name}</span>'
                )
            html_parts.append(
                f'<div class="unassigned-section">'
                f'<div class="badge-section-label">discovered features (click to toggle)</div>'
                f'<div class="feature-badges">{"".join(db)}</div>'
                f'</div>'
            )

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Result cards, baseline, summary, drift chart
# ---------------------------------------------------------------------------

def render_cards(results: list) -> str:
    if not results:
        return ""
    cards = []
    for r in results:
        status = r.get("status", "")
        name = r["name"]
        axis = r.get("axis", "")
        ref_kw = r.get("ref_keywords", "")

        if axis == "referential" and ref_kw:
            display_name = f'{_esc(ref_kw)} [{axis}]'
        elif axis:
            display_name = f'{_esc(name)} [{axis}]'
        else:
            display_name = _esc(name)

        header = f'<div class="card-header"><span class="card-name">{display_name}</span></div>'

        if status == "generating":
            body = '<span class="st st-wait">generating...</span>'
        elif status == "testing":
            variant = r.get("variant", "")
            body = (
                f'<span class="st st-wait">testing...</span>'
                f'<details><summary style="color:#555;font-size:0.85em;cursor:pointer;margin-top:4px">show variant</summary>'
                f'<div class="card-variant">{_esc(variant)}</div></details>'
            )
        elif status == "re-testing":
            variant = r.get("variant", "")
            body = (
                f'<span class="st st-wait">re-testing...</span>'
                f'<details><summary style="color:#555;font-size:0.85em;cursor:pointer;margin-top:4px">show variant</summary>'
                f'<div class="card-variant">{_esc(variant)}</div></details>'
            )
        elif status == "error":
            err = r.get("error", "Unknown error")
            body = f'<span class="st st-err">error: {_esc(err)}</span>'
        else:
            variant = r.get("variant", "")
            answer = r.get("answer", "")
            correct = r.get("correct", False)
            positive_drift = r.get("positive_drift", False)
            negative_drift = r.get("negative_drift", False)
            retries = r.get("retries", 0)

            if positive_drift:
                badge = '<span class="st st-pos-drift">POSITIVE DRIFT</span>'
            elif negative_drift:
                badge = '<span class="st st-drift">NEGATIVE DRIFT</span>'
            else:
                badge = '<span class="st st-neutral">no drift</span>'

            retry_note = f' <span style="color:#555">(retry x{retries})</span>' if retries else ''
            reasoning = r.get("reasoning", "")

            # Show CoT indicator in summary if reasoning exists
            cot_note = ""
            if reasoning:
                word_count = len(reasoning.split())
                cot_note = (
                    f' <span class="card-divider">|</span>'
                    f' <span style="color:#8b5cf6;font-size:0.85em">CoT: {word_count}w</span>'
                )

            # Compact one-line summary with expandable details
            summary_line = (
                f'<div class="card-summary-line">'
                f'{badge}'
                f'<span class="card-divider">|</span>'
                f'answered: <b>{_esc(answer)}</b>'
                f'{retry_note}{cot_note}'
                f'</div>'
            )

            # Expandable details section — auto-expand when reasoning exists
            detail_parts = []
            if variant:
                detail_parts.append(
                    f'<div class="card-detail-section">'
                    f'<div class="card-detail-label">variant</div>'
                    f'<div class="card-variant">{_esc(variant)}</div>'
                    f'</div>'
                )
            if reasoning:
                detail_parts.append(
                    f'<div class="card-detail-section">'
                    f'<div class="card-detail-label">chain-of-thought ({len(reasoning.split())} words)</div>'
                    f'<div class="card-variant" style="border-left:2px solid #8b5cf6;padding-left:8px;'
                    f'max-height:300px;overflow-y:auto;white-space:pre-wrap">{_esc(reasoning)}</div>'
                    f'</div>'
                )

            details_html = ""
            if detail_parts:
                open_attr = " open" if reasoning else ""
                details_html = (
                    f'<details class="card-expand"{open_attr}>'
                    f'<summary>details</summary>'
                    f'<div class="card-details-body">{"".join(detail_parts)}</div>'
                    f'</details>'
                )

            body = f'{summary_line}{details_html}'

        cards.append(f'<div class="result-card">{header}{body}</div>')
    return f'<div class="results-grid">{"".join(cards)}</div>'


def render_baseline(baseline_answer: str, baseline_correct: bool,
                    gen_model: str, target_model: str) -> str:
    cls = "st-ok" if baseline_correct else "st-drift"
    label = "correct" if baseline_correct else "WRONG"
    return (
        f'<div class="baseline-card">'
        f'<b>baseline</b> &mdash; '
        f'<span class="st {cls}">{label}</span> '
        f'answered: <b>{_esc(baseline_answer)}</b>'
        f'<div style="font-size:0.8em;color:#555;margin-top:2px">target: {_esc(target_model)}</div>'
        f'</div>'
    )


def render_summary(results: list, baseline_correct: bool, baseline_answer: str) -> str:
    completed = [r for r in results if r.get("status") == "done"]
    if not completed:
        return ""

    total = len(completed)
    drift_count = sum(1 for r in completed if r.get("drifted", False))
    pos_count = sum(1 for r in completed if r.get("positive_drift", False))
    neg_count = sum(1 for r in completed if r.get("negative_drift", False))
    consistent_count = total - drift_count
    error_count = sum(1 for r in results if r.get("status") == "error")

    parts = ['<div class="summary-box">', '<div class="sum-title">Summary</div>']

    baseline_label = "correct" if baseline_correct else "WRONG"
    parts.append(f'<p><b>baseline:</b> {baseline_label}</p>')

    pct = (drift_count / total * 100) if total > 0 else 0
    parts.append(f'<p><b>drift_rate:</b> {drift_count}/{total} ({pct:.0f}%)</p>')

    if pos_count > 0:
        pos_names = [r["name"] for r in completed if r.get("positive_drift", False)]
        parts.append(
            f'<p style="margin-left:12px"><span style="color:#4ade80">positive:</span> '
            f'{pos_count} (hidden capability &mdash; wrong baseline &rarr; correct variant)'
            f'<br/><span style="color:#555;font-size:0.9em">{", ".join(pos_names)}</span></p>'
        )
    if neg_count > 0:
        neg_names = [r["name"] for r in completed if r.get("negative_drift", False)]
        parts.append(
            f'<p style="margin-left:12px"><span style="color:#f87171">negative:</span> '
            f'{neg_count} (hidden fragility &mdash; correct baseline &rarr; wrong variant)'
            f'<br/><span style="color:#555;font-size:0.9em">{", ".join(neg_names)}</span></p>'
        )
    parts.append(
        f'<p style="margin-left:12px"><span style="color:#94a3b8">no drift:</span> '
        f'{consistent_count} (same as baseline)</p>'
    )

    if drift_count == 0:
        parts.append(f'<p>No drift detected across {total} variations.</p>')

    if error_count > 0:
        parts.append(f'<p><b>errors:</b> {error_count}</p>')

    parts.append('</div>')
    return "\n".join(parts)


def render_drift_chart(results: list, baseline_correct: bool) -> str:
    """Render a compact summary drift chart grouped by axis/category."""
    completed = [r for r in results if r.get("status") == "done"]
    errors = [r for r in results if r.get("status") == "error"]
    if not completed and not errors:
        return ""

    total_done = len(completed)
    total_pos = sum(1 for r in completed if r.get("positive_drift", False))
    total_neg = sum(1 for r in completed if r.get("negative_drift", False))
    total_consistent = total_done - total_pos - total_neg
    total_err = len(errors)

    # Build per-axis breakdown
    axis_stats = {}
    for r in completed:
        axis = r.get("axis", "other") or "other"
        if axis not in axis_stats:
            axis_stats[axis] = {"pos": 0, "neg": 0, "ok": 0, "total": 0}
        axis_stats[axis]["total"] += 1
        if r.get("positive_drift", False):
            axis_stats[axis]["pos"] += 1
        elif r.get("negative_drift", False):
            axis_stats[axis]["neg"] += 1
        else:
            axis_stats[axis]["ok"] += 1

    # Overall stacked bar
    max_bar_width = 300
    parts = ['<div class="drift-chart">']
    parts.append('<div class="badge-section-label">drift overview</div>')

    if total_done > 0:
        pos_w = int(total_pos / total_done * max_bar_width) if total_pos else 0
        neg_w = int(total_neg / total_done * max_bar_width) if total_neg else 0
        ok_w = max_bar_width - pos_w - neg_w

        parts.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:8px 0">'
            f'<span style="font-size:0.78em;color:#888;font-family:monospace;min-width:50px">overall</span>'
            f'<div style="display:flex;height:18px;border-radius:3px;overflow:hidden;flex:1;max-width:{max_bar_width}px">'
        )
        if pos_w > 0:
            parts.append(f'<div style="width:{pos_w}px;background:rgba(74,222,128,0.4)"></div>')
        if neg_w > 0:
            parts.append(f'<div style="width:{neg_w}px;background:rgba(248,113,113,0.4)"></div>')
        if ok_w > 0:
            parts.append(f'<div style="width:{ok_w}px;background:rgba(148,163,184,0.15)"></div>')
        parts.append('</div>')

        drift_total = total_pos + total_neg
        pct = drift_total / total_done * 100
        color = "#f87171" if drift_total > 0 else "#4ade80"
        parts.append(
            f'<span style="font-size:0.78em;font-family:monospace;color:{color}">'
            f'{drift_total}/{total_done} ({pct:.0f}%)</span>'
            f'</div>'
        )

    # Per-axis stacked bars
    for axis_name in sorted(axis_stats.keys()):
        s = axis_stats[axis_name]
        if s["total"] == 0:
            continue
        pos_w = int(s["pos"] / s["total"] * max_bar_width) if s["pos"] else 0
        neg_w = int(s["neg"] / s["total"] * max_bar_width) if s["neg"] else 0
        ok_w = max_bar_width - pos_w - neg_w

        axis_drift = s["pos"] + s["neg"]
        axis_pct = axis_drift / s["total"] * 100
        axis_color = "#f87171" if axis_drift > 0 else "#4ade80"

        parts.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<span style="font-size:0.72em;color:#555;font-family:monospace;min-width:50px;text-align:right">{_esc(axis_name)}</span>'
            f'<div style="display:flex;height:12px;border-radius:2px;overflow:hidden;flex:1;max-width:{max_bar_width}px">'
        )
        if pos_w > 0:
            parts.append(f'<div style="width:{pos_w}px;background:rgba(74,222,128,0.4)"></div>')
        if neg_w > 0:
            parts.append(f'<div style="width:{neg_w}px;background:rgba(248,113,113,0.4)"></div>')
        if ok_w > 0:
            parts.append(f'<div style="width:{ok_w}px;background:rgba(148,163,184,0.15)"></div>')
        parts.append('</div>')
        parts.append(
            f'<span style="font-size:0.7em;font-family:monospace;color:{axis_color}">'
            f'{axis_drift}/{s["total"]}</span>'
            f'</div>'
        )

    # Legend
    parts.append(
        '<div style="display:flex;gap:14px;margin-top:8px;font-size:0.7em;font-family:monospace;color:#555">'
        '<span><span style="display:inline-block;width:10px;height:10px;background:rgba(74,222,128,0.4);border-radius:2px;vertical-align:middle"></span> positive drift</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;background:rgba(248,113,113,0.4);border-radius:2px;vertical-align:middle"></span> negative drift</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;background:rgba(148,163,184,0.15);border:1px solid #333;border-radius:2px;vertical-align:middle"></span> no drift</span>'
    )
    if total_err > 0:
        parts.append(
            f'<span><span style="display:inline-block;width:10px;height:10px;background:rgba(168,85,247,0.3);border-radius:2px;vertical-align:middle"></span> {total_err} error(s)</span>'
        )
    parts.append('</div>')

    parts.append('</div>')
    return "\n".join(parts)
