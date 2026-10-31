"""Draw existing aggregate results. No regressions or agents are run here."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL = "#247A89"
ORANGE = "#C5703B"
GREY = "#AAB4BC"

def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})

def survey_associations(table, path):
    style()
    labels = {"perceived_aesthetics": "Perceived aesthetics", "crime_safety": "Crime safety",
              "access_protection": "Access / protection", "traffic_calm": "Traffic calm",
              "walking_access": "Perceived walking access"}
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for j, row in table.iterrows():
        color = TEAL if row.predictor == "perceived_aesthetics" else GREY
        ax.errorbar(row.estimate, j, xerr=[[row.estimate-row.ci_lower], [row.ci_upper-row.estimate]],
                    fmt="o", capsize=4, color=color, ms=7, lw=2)
        ax.text(.455, j, f"{row.estimate:.3f}", va="center", ha="right", fontsize=10)
    ax.axvline(0, color=GREY, ls="--", lw=1)
    ax.set(yticks=range(len(table)), yticklabels=[labels[x] for x in table.predictor],
           xlabel="Standardized coefficient for felt ownership (stored 95% CI)",
           xlim=(-.015,.465), ylim=(len(table)-.4,-.6))
    fig.suptitle("Neighbourhood qualities and felt ownership", fontsize=16, y=.98)
    ax.set_title("Saved joint survey model · 2,691 complete cases", fontsize=11, pad=14)
    fig.text(.02,.02,"City fixed effects; demographic adjustment; city-clustered uncertainty. Observational associations.", fontsize=9)
    fig.subplots_adjust(left=.27,right=.97,bottom=.17,top=.80)
    fig.savefig(path,dpi=170,facecolor="white")
    plt.close(fig)

def archived_replay(table,path):
    style()
    labels = {"civic_action":"Civic action", "dosomething":"Informal social control",
              "report":"Reporting", "steward":"Stewardship"}
    fig, ax = plt.subplots(figsize=(10.5,5.7))
    y=np.arange(len(table));h=.28
    ax.barh(y-h/2,table.first_visit_mean_pct,height=h,color=GREY,label="Judged as a first visit")
    ax.barh(y+h/2,table.history_mean_pct,height=h,color=TEAL,label="Judged with recorded history")
    for j,r in table.iterrows():
        ax.text(r.first_visit_mean_pct+.45,j-h/2,f"{r.first_visit_mean_pct:.1f}",va="center",fontsize=9)
        ax.text(r.history_mean_pct+.45,j+h/2,f"{r.history_mean_pct:.1f}",va="center",fontsize=9)
    ax.set(yticks=y,yticklabels=[labels[r.outcome]+f"\n(n={int(r.incident_n):,} incidents)" for r in table.itertuples()],
           xlabel="Language-model-judged share (0–100)",xlim=(0,50),ylim=(len(table)-.45,-.55))
    ax.legend(loc="lower right",bbox_to_anchor=(1,1.01),frameon=False,fontsize=9)
    fig.suptitle("Archived model judgments: the same incidents, two contexts",fontsize=15,y=.98)
    fig.text(.02,.065,"Selected repeated-visit subset (≥6 visits); original archived run, pooled scoring arms. Descriptive means.",fontsize=9)
    fig.text(.02,.025,"These are model judgments, not observed human behaviour. No confidence intervals are inferred from these means.",fontsize=9)
    fig.subplots_adjust(left=.28,right=.97,bottom=.20,top=.82)
    fig.savefig(path,dpi=170,facecolor="white")
    plt.close(fig)
