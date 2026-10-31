# Inviting neighbourhoods: selected research code and results

**Selected research sample from an ongoing, unpublished study.** This sample contains selected analysis code and saved results from an ongoing project on neighbourhood environments, felt ownership and civic care. The complete study materials remain private.

## Research question

How do neighbourhood qualities relate to people's sense that a place is theirs to care for, and how might repeated encounters with a place relate to civic responses?

The broader project combines geographic analyses, surveys and archived generative agent-based modelling. This sample illustrates the survey analysis and numerical post-processing of archived model judgments. It does not distribute the simulation engine or its prompts.

## Selected results

![Survey associations with felt ownership](figures/01_saved_survey_associations.png)

The survey panel displays five saved coefficients from one joint model of felt ownership (**2,691 complete cases**), adjusting for demographic covariates and city fixed effects, with city-clustered 95% confidence intervals. Perceived aesthetics has the largest coefficient among these five included neighbourhood qualities. This is an observational association, not an estimated intervention effect.

![Selected archived model judgments](figures/02_saved_archived_replay.png)

The second panel shows saved language-model judgments for archived incidents at places visited at least six times. Each pair compares the **same incidents** judged as a first visit and with the resident's recorded history. Counts are 504 incidents for the first three outcomes and 246 for stewardship. Outcomes are model-judged shares on a 0–100 scale; they are not observed human actions or estimated real-world probabilities.

This descriptive panel pools the original and swapped-scoring arms in the original archived run, following the source summary table. It shows no newly calculated confidence intervals. Calibration and scoring sensitivities in the full study are outside this sample; the panel alone does not establish an aesthetics-to-civic-care causal mechanism.

## Representative code

- `code/area_model_excerpt.py`: unchanged original functions for rank standardization, state demeaning and HC1 estimation in the geographic analysis.
- `code/survey_model_excerpt.py`: the original felt-ownership construction and joint estimation block, wrapped as a function accepting an authorized survey frame. The wrapper supplies original notebook globals explicitly and removes file writing and display statements; the numerical estimation statements are retained.
- `code/archived_replay_excerpt.py`: unchanged original functions for CR1 clustered uncertainty and paired post-processing of archived judgments. It contains no language-model calls or prompt text.
- `code/result_figures.py`: display code for the two selected saved-result panels.

These analysis functions are provided for inspection and require authorized records that are not included. The executed notebook only redraws saved summary results; it does not fit these models or run agents.

## Executed notebook and running it

Open [`notebooks/SELECTED_RESULTS_EXECUTED.ipynb`](notebooks/SELECTED_RESULTS_EXECUTED.ipynb) for completed cells, the included summary tables and embedded result figures.

Tested with Python 3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/execute.py
```

Re-execution redraws the two figures, refreshes notebook outputs and writes an execution receipt under ignored `outputs/`. It needs no API keys, model downloads or external data.

## Included and withheld material

`results/` contains **five model-level survey coefficients and four aggregate pairs of archived judgment means**, with the associated sample sizes. It contains no respondent records, addresses, geographic analysis frames, agent histories, incident-level judgments, screenshots, model weights, private prompts or API credentials. The original research notebooks and full simulation configuration are withheld.

`PUBLIC_FILES.json` is the closed list of proposed public files. This is a selected research sample, not a complete replication package. The source project's copyright and license notice is retained in `LICENSE`; references there to the full upstream package do not imply that its data are included here.
