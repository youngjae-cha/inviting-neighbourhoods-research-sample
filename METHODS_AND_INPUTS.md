# Method source and input contracts

## Original simulation and later replay

`archived_simulation.py` preserves the original `Agent`, `make_agents`, `day_state`, `choose_pois`, `p_notice`, `audit_prompt`, `_audit`, `report_prop` and `run_one` bodies, plus the actual non-smoke configuration and constants. The original source's automatic grid launch, Colab mounts, frozen-data loading and individual-case diagnostics are omitted. The new `configure_inputs` binds externally supplied frames and an explicit output directory. It creates no provider client.

The original runner's older audit sees incident wording, visit count and numeric ownership memory. It retains source fallback scores (0.65 for public problems and 0.25 for private problems) when audit values are unavailable. Its reporting propensity is a specified logistic formula. These quantities are not substituted for the later replay results shown in the selected figure.

`replay_judgment.py` and `replay_analysis.py` are complete original files. The later replay withholds scene/place identity, imagery, visual score and numeric ownership memory. It varies recorded history versus a structurally matched first visit, with history comprising visit count and prior noticed problems. Action prompts ask for judged residents out of 100; the separately elicited ownership probe asks for a 0–100 rating. Values are stored on 0–1 scales. Civic action averages clean-up signup and joining a resident group.

## Input schemas

| Input | Original field names (no records supplied) |
|---|---|
| Place frame | `zip`, `poi_id`, `LATITUDE`, `LONGITUDE`, `visual_invitation_actual` |
| Prepared persona pool | `zip`, `respondent_id`, `prosocial`, `attention`, `efficacy`, `busyness`, `attach_prior` |
| Questionnaire frame for mapping | `ZIPCODE` and the source WP/HWB/M19 item columns used by `persona_mapping.py` |
| Archived incidents for replay | Original event fields including area/agent/place keys, day, engine, arm, visual-pull setting, noticed flag, visit count, problem type and state variables |

The mapping code describes transformations only. It includes no survey responses or respondent identities. Runtime caches, incident files and generated outputs belong outside the public repository and are ignored by the candidate `.gitignore`.

## Reproducibility scope

The archived single-area runner uses Python's process-dependent `hash()` in its random-seed expressions. Those source expressions are preserved, so this sample does not promise bitwise regeneration across newly launched processes. The original fallback behavior is also preserved and identified, not silently presented as a fresh model judgment.

The public notebooks import/inspect functions and redraw selected existing summaries. They do not execute the agent runner, create new incidents, issue provider calls, recompute outcome analyses or infer human causal effects. Historical prompt wording and hard-coded price calculations are archival source, not current model or pricing advice. For analysis interpretation, the actual source archive remains distinct from the rules and assumptions that generated it.
