"""
Plot autofocus sweep data with shaded depth-of-field region.

Depth of field is defined as where Laplacian Variance >= DOF_THRESHOLD * peak.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH      = Path(__file__).parent / "autofocus_data.csv"
DOF_THRESHOLD = 0.5          # fraction of peak Laplacian Variance for DoF boundary
OUT_PATH      = Path(__file__).parent / "autofocus_plot.png"

# ── Load & sort by position ───────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).sort_values("position_mm")
pos  = df["position_mm"].values
lapv = df["Laplacian Variance"].values

# Remap so peak = 184 mm (distance from camera)
pos = pos - pos[np.argmax(lapv)] + 184.0

# ── DoF boundary ─────────────────────────────────────────────────────────────
threshold = DOF_THRESHOLD * lapv.max()
in_dof    = lapv >= threshold
left      = pos[in_dof][0]
right     = pos[in_dof][-1]
dof_mm    = right - left

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.2))

ax.axvspan(left, right, color="steelblue", alpha=0.15, zorder=0,
           label="Usable DoF = %.1f mm" % dof_mm)
ax.plot(pos, lapv, "k-o", ms=3, lw=1.4, label="Laplacian Variance")

ax.set_xlabel("Distance from camera (mm)")
ax.set_ylabel("Laplacian Variance")
ax.set_title("Autofocus sweep — usable depth of field")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180)
print("Saved %s" % OUT_PATH)
plt.show()
