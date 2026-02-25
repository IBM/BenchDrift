"""
BenchDrift App — Gradio UI layout and event wiring.

This is the only file that imports gradio. All logic lives in other modules:
- app.theme: CSS, JS, Gradio theme
- app.ollama: Ollama HTTP calls
- app.renderers: HTML rendering
- app.hf_loader: HuggingFace + JSON upload
- app.runner: unified run engine (uses pipeline core — no duplicate logic)
"""

import json

import gradio as gr

from benchdrift.pipeline.feature_relevance import (
    ALL_AXIS_NAMES,
    enrich_features_with_llm,
    get_problem_features,
    parse_axes,
    rank_axes_with_llm,
    TRANSFORMATION_TO_AXIS,
    _get_valid_axes,
    _rank_axes_by_features,
)

from app.ollama import (
    BACKENDS, OLLAMA_BASE_URL, call_llm, get_available_models,
    get_merged_models, parse_model_selection,
)
from app.renderers import ALL_AXES_LIST, DEFAULT_AXES_LIST, _filter_types_by_axes, render_analysis
from app.runner import (
    RunMode,
    build_meta,
    detect_referential_candidates,
    eval_changed,
    generation_changed,
    inject_meta,
    parse_prev,
    run,
    strip_instruction_tag,
    testing_changed,
)
from app.hf_loader import (
    HF_AVAILABLE,
    hf_fetch_problems,
    hf_load_dataset_info,
    hf_on_config_change,
    hf_select_problem,
    json_upload_handler,
)
from app.theme import BADGE_JS, CUSTOM_CSS, make_theme

# Examples with ground-truth answers
EXAMPLES = [
    [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells the remainder "
        "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
        "does she make every day at the farmers' market?",
        "18",
    ],
    [
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "3",
    ],
    [
        "If it takes 30 minutes to walk 1.5 miles to the store, and I leave at 2:00 PM, "
        "what time will I arrive?",
        "2:30 PM",
    ],
    [
        "A bridge can support 5000 pounds. Cars weigh 2500 pounds each. "
        "How many cars can the bridge hold at once?",
        "2",
    ],
    [
        "The longest side of a right triangle is 5 cm, and the shortest side is 3 cm. "
        "Find the area of the triangle in square cm.",
        "6",
    ],
]


def build_app():
    available_models = get_merged_models(["ollama"])

    with gr.Blocks(
        title="BenchDrift",
        theme=make_theme(),
        css=CUSTOM_CSS,
        js=BADGE_JS,
    ) as app:

        # --- Settings sidebar ---
        with gr.Sidebar(label="settings", open=False, position="right", width=340):
            gr.Markdown("### backends")
            cfg_backend = gr.CheckboxGroup(
                choices=BACKENDS, value=["ollama"], label="active backends",
                info="check all backends whose models should appear in dropdowns",
            )
            gr.Markdown("### evaluation")
            cfg_judge_model = gr.Dropdown(
                choices=[""] + available_models, value="",
                label="judge model", info="empty = same as generator",
            )
            cfg_eval_method = gr.Radio(
                choices=["string matching", "llm judge"],
                value="string matching", label="eval method",
            )
            cfg_validate = gr.Checkbox(
                label="validate variations (LLM)", value=True,
                info="LLM checks each variant preserves the answer",
            )
            gr.Markdown("### generation")
            cfg_gen_temp = gr.Slider(0.0, 1.5, 0.5, step=0.05, label="temperature")
            cfg_gen_max_tokens = gr.Slider(128, 4096, 1024, step=128, label="max tokens")
            cfg_gen_max_retries = gr.Slider(0, 5, 2, step=1, label="max retries")
            gr.Markdown("### solver")
            cfg_solver_temp = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="temperature")
            cfg_solver_max_tokens = gr.Slider(32, 2048, 256, step=32, label="max tokens")
            gr.Markdown("### reasoning")
            cfg_enable_reasoning = gr.Checkbox(
                label="enable reasoning (CoT)", value=False,
                info="model thinks step-by-step before answering",
            )
            cfg_think_max_tokens = gr.Slider(
                128, 8192, 2048, step=128,
                label="thinking token budget",
                info="total tokens for thinking + answer (model-dependent max)",
            )
            gr.Markdown("### advanced")
            cfg_ollama_url = gr.Textbox(value=OLLAMA_BASE_URL, label="Ollama URL")
            cfg_ollama_timeout = gr.Slider(30, 600, 120, step=10, label="timeout (s)")
            cfg_hf_max_rows = gr.Slider(10, 500, 50, step=10, label="HF max rows")
            cfg_llm_trait_fallback = gr.Checkbox(
                label="LLM trait fallback", value=False,
                info="when regex can't classify the answer style, use a single LLM call (cached)",
            )

        # --- Header ---
        gr.HTML(
            '<div class="app-header">'
            '<h1>BenchDrift</h1>'
            '<div class="subtitle">does rephrasing a problem change the answer?</div>'
            '<div class="header-actions">'
            '<button id="theme-toggle-btn" onclick="toggleBenchDriftTheme()">light mode</button>'
            '</div>'
            '</div>'
        )

        # --- State ---
        results_json_state = gr.State("[]")
        features_state = gr.State({})
        llm_features_state = gr.State({})
        discovered_features_state = gr.State({})
        axis_ranking_state = gr.State([])
        hf_rows_state = gr.State([])
        hf_answer_instruction_state = gr.State("")
        json_problem_col_state = gr.State("")
        json_answer_col_state = gr.State("")
        features_sync_box = gr.Textbox(value="{}", visible=False, elem_id="features-sync-box")

        # --- Input ---
        gr.HTML('<div class="section-label">problem &amp; models</div>')
        with gr.Row():
            with gr.Column(scale=3):
                problem_input = gr.Textbox(
                    label="problem",
                    placeholder="Paste your math/reasoning problem here...",
                    lines=3, max_lines=8,
                )
                answer_input = gr.Textbox(
                    label="expected answer", placeholder="e.g. 18", lines=1,
                )
            with gr.Column(scale=1):
                gen_model = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="generator", info="rephrases the problem",
                )
                target_model = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="target", info="tested for drift",
                )
                refresh_btn = gr.Button("refresh models", size="sm")
        with gr.Row():
            variation_mode = gr.Dropdown(
                choices=["axes-based", "free-form"],
                value="axes-based",
                label="variation mode",
                info="axes-based = taxonomy-driven | free-form = LLM generates all",
                scale=1,
            )
            top_k_slider = gr.Slider(1, 63, 8, step=1, label="top-k variations", scale=1)
            enabled_axes_cb = gr.CheckboxGroup(
                choices=ALL_AXES_LIST,
                value=DEFAULT_AXES_LIST,
                label="axes",
                elem_classes=["axes-checkboxgroup"],
                scale=3,
            )

        gr.HTML('<div class="section-label">data</div>')
        with gr.Accordion("examples", open=False):
            gr.Examples(
                examples=EXAMPLES,
                inputs=[problem_input, answer_input],
                label="examples",
                elem_id="benchdrift-examples",
            )

        # --- HuggingFace Loader ---
        with gr.Accordion("load from huggingface", open=False):
            if not HF_AVAILABLE:
                gr.HTML('<p style="color:#fbbf24;font-size:0.85em">pip install datasets to enable HuggingFace loading</p>')
            with gr.Row():
                hf_dataset_name = gr.Textbox(label="dataset name", placeholder="e.g. gsm8k, cais/mmlu", scale=3)
                hf_load_btn = gr.Button("load", size="sm", scale=1)
            with gr.Row():
                hf_config = gr.Dropdown(label="config", choices=[], scale=1)
                hf_split = gr.Dropdown(label="split", choices=[], scale=1)
                hf_problem_col = gr.Dropdown(label="problem col", choices=[], scale=1)
                hf_answer_col = gr.Dropdown(label="answer col", choices=[], scale=1)
            with gr.Row():
                hf_override_problem = gr.Textbox(
                    label="override problem field(s)",
                    placeholder="e.g. question  or  instruction+context",
                    info="priority over dropdown — use + to combine columns",
                    scale=2,
                )
                hf_override_answer = gr.Textbox(
                    label="override answer field",
                    placeholder="e.g. response",
                    info="priority over dropdown",
                    scale=1,
                )
            hf_fetch_btn = gr.Button("fetch problems", size="sm")
            hf_problem_selector = gr.Dropdown(label="problems", choices=[], interactive=True)
            hf_status = gr.HTML(value="")

        # --- Custom JSON Upload ---
        with gr.Accordion("upload custom JSON", open=False):
            json_upload = gr.File(label="JSON or JSONL file", file_types=[".json", ".jsonl"])
            json_problem_selector = gr.Dropdown(label="select problem", choices=[], interactive=True)
            json_upload_status = gr.HTML(value="")

        gr.HTML('<div class="section-label">analysis &amp; configuration</div>')
        # --- Feature analysis ---
        with gr.Accordion("feature analysis", open=True) as analysis_accordion:
            analysis_html = gr.HTML(value="")
            with gr.Row():
                enrich_btn = gr.Button("enrich features & rerank (LLM)", size="sm", variant="secondary")
                discover_btn = gr.Button("discover new features", size="sm", variant="secondary")

        # --- Action buttons ---
        with gr.Row():
            run_btn = gr.Button("generate & analyze", variant="primary", size="lg", scale=4)
            stop_btn = gr.Button("stop", variant="stop", size="lg", scale=1)
            force_rerun_btn = gr.Button("force full re-run", variant="secondary", size="lg", scale=2)

        # --- Results ---
        gr.HTML('<div class="badge-section-label" style="margin-top:16px;margin-bottom:4px;font-size:0.72em">results</div>')
        cards_html = gr.HTML(value="")
        with gr.Row():
            summary_html = gr.HTML(value="")
            drift_chart_html = gr.HTML(value="")

        # ===================================================================
        # Event wiring
        # ===================================================================

        # Refresh models — merges all checked backends with client/model prefix
        def _refresh_models(backends):
            backends = backends or ["ollama"]
            models = get_merged_models(backends)
            return (gr.update(choices=models, value=models[0] if models else None),
                    gr.update(choices=models, value=models[0] if models else None),
                    gr.update(choices=[""] + models))

        refresh_btn.click(
            fn=_refresh_models, inputs=[cfg_backend],
            outputs=[gen_model, target_model, cfg_judge_model],
        )

        # Auto-refresh models when backends change
        cfg_backend.change(
            fn=_refresh_models, inputs=[cfg_backend],
            outputs=[gen_model, target_model, cfg_judge_model],
        )

        # Disable axes checkboxes when free-form is selected
        def _on_variation_mode_change(mode):
            if mode == "free-form":
                return gr.update(interactive=False, label="axes (disabled in free-form mode)")
            return gr.update(interactive=True, label="axes")

        variation_mode.change(
            fn=_on_variation_mode_change, inputs=[variation_mode],
            outputs=[enabled_axes_cb],
        )

        # Feature detection on problem change
        def show_features(problem_text, axes_list, top_k_val):
            if not problem_text or not problem_text.strip():
                return "", {}, {}, {}, [], "{}", "[]", gr.update(label="feature analysis")
            text, _ = strip_instruction_tag(problem_text.strip())
            features = get_problem_features(text)
            ref_cands = detect_referential_candidates(text)
            sync = {k: v for k, v in features.items() if isinstance(v, bool)}
            ea = set(axes_list) if axes_list else set(DEFAULT_AXES_LIST)
            valid_axes = _get_valid_axes(features, enabled_axes=ea)
            n_feats = sum(1 for v in features.values() if isinstance(v, bool) and v)
            html = render_analysis(
                features, llm_features=None, discovered_features=None,
                ranked_axes=None, enabled_axes=ea,
                top_k=int(top_k_val), ref_candidates=ref_cands,
            )
            acc_label = f"feature analysis — {len(valid_axes)} axes, {n_feats} features"
            return html, features, {}, {}, [], json.dumps(sync), "[]", gr.update(label=acc_label)

        problem_input.change(
            fn=show_features,
            inputs=[problem_input, enabled_axes_cb, top_k_slider],
            outputs=[analysis_html, features_state, llm_features_state,
                     discovered_features_state, axis_ranking_state,
                     features_sync_box, results_json_state, analysis_accordion],
        )

        # LLM enrichment + axis ranking
        def enrich_with_llm(problem_text, regex_feats, disc_feats,
                            model_name, axes_list, top_k_val,
                            c_url, c_tout):
            if not problem_text or not problem_text.strip():
                return "", {}, [], "{}", gr.update(label="feature analysis")
            text, _ = strip_instruction_tag(problem_text.strip())
            ea = set(axes_list) if axes_list else set(DEFAULT_AXES_LIST)
            if not model_name or model_name.startswith("("):
                ref_cands = detect_referential_candidates(text)
                html = render_analysis(
                    regex_feats, llm_features=None, discovered_features=disc_feats,
                    ranked_axes=None, enabled_axes=ea,
                    top_k=int(top_k_val), ref_candidates=ref_cands,
                )
                return html, {}, [], "{}", gr.update(label="feature analysis")

            # Parse client/model from dropdown selection
            _, bare_model = parse_model_selection(model_name)
            base_url = c_url or OLLAMA_BASE_URL
            timeout = float(c_tout) if c_tout else 10.0
            llm_feats = enrich_features_with_llm(text, ollama_base_url=base_url, model=bare_model, timeout=timeout)
            merged = dict(regex_feats)
            merged.update(llm_feats)
            if disc_feats:
                merged.update(disc_feats)

            ranked_axes = rank_axes_with_llm(
                text, merged, ollama_base_url=base_url, model=bare_model,
                timeout=timeout, enabled_axes=ea,
            )

            sync = {k: v for k, v in regex_feats.items() if isinstance(v, bool)}
            sync.update(llm_feats)
            if disc_feats:
                sync.update(disc_feats)

            n_feats = sum(1 for v in merged.values() if isinstance(v, bool) and v)
            valid_axes = _get_valid_axes(merged, enabled_axes=ea)
            acc_label = f"feature analysis — {len(valid_axes)} axes, {n_feats} features (LLM enriched)"

            ref_cands = detect_referential_candidates(text)
            html = render_analysis(
                regex_feats, llm_features=llm_feats, discovered_features=disc_feats,
                ranked_axes=ranked_axes, enabled_axes=ea,
                top_k=int(top_k_val), ref_candidates=ref_cands,
            )
            return html, llm_feats, ranked_axes, json.dumps(sync), gr.update(label=acc_label)

        enrich_btn.click(
            fn=enrich_with_llm,
            inputs=[problem_input, features_state, discovered_features_state,
                    gen_model, enabled_axes_cb, top_k_slider,
                    cfg_ollama_url, cfg_ollama_timeout],
            outputs=[analysis_html, llm_features_state, axis_ranking_state,
                     features_sync_box, analysis_accordion],
        )

        # Auto-discover novel features
        def do_auto_discover(problem_text, model_name, regex_feats,
                             llm_feats, disc_feats, ranked_axes,
                             axes_list, top_k_val, c_url, c_tout):
            clean_text, _ = strip_instruction_tag(problem_text.strip()) if problem_text else ("", "")
            all_existing = dict(regex_feats) if regex_feats else {}
            if llm_feats:
                all_existing.update(llm_feats)
            if disc_feats:
                all_existing.update(disc_feats)

            from app.runner import _auto_discover_features
            cfg = {"ollama_base_url": c_url or OLLAMA_BASE_URL,
                   "ollama_timeout": int(c_tout) if c_tout else 120}
            new_disc, _ = _auto_discover_features(clean_text, model_name, all_existing, cfg=cfg)

            merged_disc = dict(disc_feats) if disc_feats else {}
            merged_disc.update(new_disc)

            sync = {k: v for k, v in regex_feats.items() if isinstance(v, bool)}
            if llm_feats:
                sync.update(llm_feats)
            sync.update(merged_disc)

            # Re-rank axes with all merged features
            merged_all = dict(regex_feats) if regex_feats else {}
            if llm_feats:
                merged_all.update(llm_feats)
            merged_all.update(merged_disc)

            ea = set(axes_list) if axes_list else set(DEFAULT_AXES_LIST)
            valid_axes = _get_valid_axes(merged_all, enabled_axes=ea)
            new_ranking = _rank_axes_by_features(merged_all, valid_axes)

            n_feats = sum(1 for v in merged_all.values() if isinstance(v, bool) and v)
            acc_label = f"feature analysis — {len(valid_axes)} axes, {n_feats} features"

            ref_cands = detect_referential_candidates(clean_text) if clean_text else []
            html = render_analysis(
                regex_feats, llm_features=llm_feats, discovered_features=merged_disc,
                ranked_axes=new_ranking,
                enabled_axes=ea, top_k=int(top_k_val),
                ref_candidates=ref_cands,
            )
            return html, merged_disc, json.dumps(sync), new_ranking, gr.update(label=acc_label)

        discover_btn.click(
            fn=do_auto_discover,
            inputs=[problem_input, gen_model, features_state,
                    llm_features_state, discovered_features_state,
                    axis_ranking_state, enabled_axes_cb, top_k_slider,
                    cfg_ollama_url, cfg_ollama_timeout],
            outputs=[analysis_html, discovered_features_state, features_sync_box,
                     axis_ranking_state, analysis_accordion],
        )

        # --- Smart run: detect what changed, pick cheapest path ---
        def _build_cfg(c_backend, c_judge, c_eval, c_validate, c_gtemp, c_gmax, c_gretry,
                       c_stemp, c_smax, c_reasoning, c_think_max,
                       c_url, c_tout, c_hfmax, c_llm_trait_fb=False):
            # c_backend is now a list of checked backends (for dropdown population).
            # The actual backend per call is parsed from the client/model selection.
            # We keep "backend" in cfg for backward compat but it's overridden per call.
            primary = c_backend[0] if (isinstance(c_backend, list) and c_backend) else (c_backend or "ollama")
            # Parse judge model if it has client/ prefix
            judge_backend, judge_model = parse_model_selection(c_judge) if c_judge else ("", "")
            return {
                "backend": primary,
                "judge_model": judge_model or c_judge or "",
                "judge_backend": judge_backend or "",
                "eval_method": c_eval,
                "validate_variations": bool(c_validate),
                "gen_temperature": float(c_gtemp),
                "gen_max_tokens": int(c_gmax),
                "gen_max_retries": int(c_gretry),
                "solver_temperature": float(c_stemp),
                "solver_max_tokens": int(c_smax),
                "enable_reasoning": bool(c_reasoning),
                "think_max_tokens": int(c_think_max),
                "ollama_base_url": c_url or OLLAMA_BASE_URL,
                "ollama_timeout": int(c_tout),
                "hf_max_rows": int(c_hfmax),
                "llm_trait_fallback": bool(c_llm_trait_fb),
            }

        def run_smart(problem, answer, gm, tm, topk, var_mode, axes_list,
                      reg_feats, llm_feats, ax_rank, sync_json, prev_json,
                      c_backend, c_judge, c_eval, c_validate,
                      c_gtemp, c_gmax, c_gretry,
                      c_stemp, c_smax, c_reasoning, c_think_max,
                      c_url, c_tout, c_hfmax, c_llm_trait_fb=False,
                      c_ds_name=""):
            cfg = _build_cfg(c_backend, c_judge, c_eval, c_validate, c_gtemp, c_gmax, c_gretry,
                             c_stemp, c_smax, c_reasoning, c_think_max, c_url, c_tout, c_hfmax,
                             c_llm_trait_fb)
            cfg["variation_mode"] = var_mode or "axes-based"
            cfg["dataset_name"] = c_ds_name or ""

            # Parse client/model from dropdown selections
            gen_backend, gen_model_name = parse_model_selection(gm)
            target_backend, target_model_name = parse_model_selection(tm)
            cfg["gen_backend"] = gen_backend
            cfg["target_backend"] = target_backend
            # Override the primary backend with the generator's backend
            cfg["backend"] = gen_backend

            prev_results, prev_meta = parse_prev(prev_json)
            current_meta = build_meta(problem, gen_model_name, target_model_name, cfg,
                                      strip_instruction_fn=strip_instruction_tag)

            if not prev_results or not prev_meta:
                mode = RunMode.FULL
                prev = None
            elif generation_changed(prev_meta, current_meta):
                mode = RunMode.FULL
                prev = None
            elif testing_changed(prev_meta, current_meta):
                mode = RunMode.RETEST
                prev = prev_results
            elif eval_changed(prev_meta, current_meta):
                mode = RunMode.REEVAL
                prev = prev_results
            else:
                mode = RunMode.INCREMENTAL
                prev = prev_results

            ea = set(axes_list) if axes_list else set(DEFAULT_AXES_LIST)
            gen = run(
                mode, problem, answer, gen_model_name, target_model_name, topk, ea,
                reg_feats, llm_feats, ax_rank, sync_json,
                prev_results=prev, cfg=cfg,
                variation_mode=var_mode or "axes-based",
            )
            for cards, summary, drift, rjson in gen:
                rjson = inject_meta(rjson, current_meta)
                yield (cards, summary, drift, rjson)

        run_event = run_btn.click(
            fn=run_smart,
            inputs=[problem_input, answer_input, gen_model, target_model,
                    top_k_slider, variation_mode, enabled_axes_cb, features_state,
                    llm_features_state, axis_ranking_state, features_sync_box,
                    results_json_state,
                    cfg_backend, cfg_judge_model, cfg_eval_method, cfg_validate,
                    cfg_gen_temp, cfg_gen_max_tokens, cfg_gen_max_retries,
                    cfg_solver_temp, cfg_solver_max_tokens,
                    cfg_enable_reasoning, cfg_think_max_tokens,
                    cfg_ollama_url, cfg_ollama_timeout, cfg_hf_max_rows,
                    cfg_llm_trait_fallback, hf_dataset_name],
            outputs=[cards_html, summary_html, drift_chart_html,
                     results_json_state],
        )

        # Force full re-run (ignores previous results)
        def run_force(problem, answer, gm, tm, topk, var_mode, axes_list,
                      reg_feats, llm_feats, ax_rank, sync_json,
                      c_backend, c_judge, c_eval, c_validate,
                      c_gtemp, c_gmax, c_gretry,
                      c_stemp, c_smax, c_reasoning, c_think_max,
                      c_url, c_tout, c_hfmax, c_llm_trait_fb=False,
                      c_ds_name=""):
            cfg = _build_cfg(c_backend, c_judge, c_eval, c_validate, c_gtemp, c_gmax, c_gretry,
                             c_stemp, c_smax, c_reasoning, c_think_max, c_url, c_tout, c_hfmax,
                             c_llm_trait_fb)
            cfg["variation_mode"] = var_mode or "axes-based"
            cfg["dataset_name"] = c_ds_name or ""

            gen_backend, gen_model_name = parse_model_selection(gm)
            target_backend, target_model_name = parse_model_selection(tm)
            cfg["gen_backend"] = gen_backend
            cfg["target_backend"] = target_backend
            cfg["backend"] = gen_backend

            current_meta = build_meta(problem, gen_model_name, target_model_name, cfg,
                                      strip_instruction_fn=strip_instruction_tag)
            ea = set(axes_list) if axes_list else set(DEFAULT_AXES_LIST)
            gen = run(
                RunMode.FULL, problem, answer, gen_model_name, target_model_name, topk, ea,
                reg_feats, llm_feats, ax_rank, sync_json, cfg=cfg,
                variation_mode=var_mode or "axes-based",
            )
            for cards, summary, drift, rjson in gen:
                rjson = inject_meta(rjson, current_meta)
                yield (cards, summary, drift, rjson)

        force_event = force_rerun_btn.click(
            fn=run_force,
            inputs=[problem_input, answer_input, gen_model, target_model,
                    top_k_slider, variation_mode, enabled_axes_cb, features_state,
                    llm_features_state, axis_ranking_state, features_sync_box,
                    cfg_backend, cfg_judge_model, cfg_eval_method, cfg_validate,
                    cfg_gen_temp, cfg_gen_max_tokens, cfg_gen_max_retries,
                    cfg_solver_temp, cfg_solver_max_tokens,
                    cfg_enable_reasoning, cfg_think_max_tokens,
                    cfg_ollama_url, cfg_ollama_timeout, cfg_hf_max_rows,
                    cfg_llm_trait_fallback, hf_dataset_name],
            outputs=[cards_html, summary_html, drift_chart_html,
                     results_json_state],
        )

        # Stop cancels both
        stop_btn.click(fn=None, cancels=[run_event, force_event])

        # --- HuggingFace wiring ---
        hf_load_btn.click(
            fn=hf_load_dataset_info, inputs=[hf_dataset_name],
            outputs=[hf_config, hf_split, hf_problem_col, hf_answer_col,
                     hf_problem_selector, hf_rows_state, hf_status,
                     hf_dataset_name],
        )
        hf_config.change(
            fn=hf_on_config_change, inputs=[hf_dataset_name, hf_config],
            outputs=[hf_split, hf_problem_col, hf_answer_col,
                     hf_problem_selector, hf_rows_state, hf_status],
        )

        def _hf_fetch(ds, cfg, split, pcol, acol, max_rows,
                      override_p, override_a):
            # Override fields take priority
            eff_pcol = override_p.strip().split("+")[0].strip() if override_p and override_p.strip() else pcol
            eff_acol = override_a.strip() if override_a and override_a.strip() else acol
            return hf_fetch_problems(ds, cfg, split, eff_pcol, eff_acol, "",
                                     n=int(max_rows) if max_rows else 50)

        hf_fetch_btn.click(
            fn=_hf_fetch,
            inputs=[hf_dataset_name, hf_config, hf_split,
                    hf_problem_col, hf_answer_col, cfg_hf_max_rows,
                    hf_override_problem, hf_override_answer],
            outputs=[hf_problem_selector, hf_rows_state,
                     hf_answer_instruction_state, hf_status],
        )

        # JSON upload — populates its own selector
        json_upload.change(
            fn=json_upload_handler, inputs=[json_upload],
            outputs=[json_problem_selector, hf_rows_state,
                     json_problem_col_state, json_answer_col_state,
                     json_upload_status],
        )

        # JSON problem selector → fill inputs
        def _json_select(choice, rows, json_pcol, json_acol):
            if not choice or not rows:
                return "", ""
            return hf_select_problem(choice, rows, json_pcol or "question", json_acol or "answer", "")

        json_problem_selector.change(
            fn=_json_select,
            inputs=[json_problem_selector, hf_rows_state,
                    json_problem_col_state, json_answer_col_state],
            outputs=[problem_input, answer_input],
        )

        # HF Problem selector — uses HF columns only (no JSON state contamination)
        def _select_problem(choice, rows, hf_pcol, hf_acol,
                            ans_instr, override_p, override_a):
            acol = override_a.strip() if override_a and override_a.strip() else hf_acol

            if override_p and override_p.strip():
                parts = [p.strip() for p in override_p.strip().split("+") if p.strip()]
                if len(parts) > 1:
                    import re as _re
                    m = _re.match(r'\[(\d+)\]', choice or "")
                    if not m or not rows:
                        return "", ""
                    idx = int(m.group(1))
                    if idx >= len(rows):
                        return "", ""
                    row = rows[idx]
                    problem = "\n\n".join(str(row.get(p, "")) for p in parts if row.get(p))
                    if ans_instr:
                        problem += f"\n\n[Instruction: {ans_instr}]"
                    from app.hf_loader import hf_parse_answer
                    answer = hf_parse_answer(row, acol) if acol else ""
                    return problem, answer
                else:
                    pcol = parts[0]
            else:
                pcol = hf_pcol

            return hf_select_problem(choice, rows, pcol, acol, ans_instr)

        hf_problem_selector.change(
            fn=_select_problem,
            inputs=[hf_problem_selector, hf_rows_state,
                    hf_problem_col, hf_answer_col, hf_answer_instruction_state,
                    hf_override_problem, hf_override_answer],
            outputs=[problem_input, answer_input],
        )

    return app
