import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path.home() / "projects/14koen_cuypers/final_analysis2"
qc_file = PROJECT_DIR / "results/QC/qc_mean_gm_dvr.tsv"

df = pd.read_csv(qc_file, sep="\t")
print(df.groupby("group")["mean_gm_dvr"].describe())

# Flag outliers (>2.5 SD from group mean)
df["z"] = df.groupby("group")["mean_gm_dvr"].transform(
    lambda x: (x - x.mean()) / x.std()
)
outliers = df[df["z"].abs() > 2.5]
if len(outliers) > 0:
    print("\n  Potential outliers (|z| > 2.5):")
    print(outliers)
else:
    print("\n✓ No outliers beyond 2.5 SD")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="group", y="mean_gm_dvr", ax=axes[0], order=["YOUNG", "OLD"])
sns.stripplot(data=df, x="group", y="mean_gm_dvr", ax=axes[0],
              order=["YOUNG", "OLD"], color="black", alpha=0.6)
axes[0].set_title("Mean GM DVR by group")
axes[0].set_ylabel("Mean GM DVR ([11C]flumazenil)")

# Per-subject bar
df_sorted = df.sort_values(["group", "subject_id"])
colors = ["#d62728" if g == "OLD" else "#1f77b4" for g in df_sorted["group"]]
axes[1].bar(df_sorted["subject_id"], df_sorted["mean_gm_dvr"], color=colors)
axes[1].set_xticklabels(df_sorted["subject_id"], rotation=90)
axes[1].set_title("Per-subject mean GM DVR")
axes[1].set_ylabel("Mean GM DVR")

plt.tight_layout()
plt.savefig(PROJECT_DIR / "results/qc_mean_gm_dvr.png", dpi=150)
plt.show()
print(f"\nPlot saved: {PROJECT_DIR / 'results/qc_mean_gm_dvr.png'}")