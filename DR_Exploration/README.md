# Video Game Sales — Data Exploration and Interaction

Team: Anderson Zhu, Krish Thakre, Vasista Vegesana.

## Review

1. Open `Video_Game_Sales_DR_Report.html` for the executed report.
2. Open `Interactive_Explorer.html` in a browser for the offline linked interface. The report also embeds the interface; if notebook trust or browser restrictions block that frame, use the separate HTML.
3. Read the results and try the three exploration cases. Review the contribution evaluation (100% each, as requested) before submitting it yourself.

`Video_Game_Sales_DR.ipynb` is the executable notebook requested by Canvas. The HTML files provide convenient review and interaction. `vgsales.csv` is the existing project dataset, copied unchanged. The prior project files remain untouched in the parent folder.

## Reproduce

Use a Python environment with **NumPy 2.3.5**. An older Anaconda NumPy emitted misleading BLAS matrix-multiplication warnings on this Mac. The corrected notebook stops on unexpected numerical warnings and rejects non-finite results; warnings are not hidden.

In VS Code, select the workspace environment **`.venv-dr-clean/bin/python`** using the notebook's top-right **Select Kernel → Python Environments** control. Restart the kernel, then Run All. Saved outputs are already executed and can be read without rerunning.

For a fresh setup, run from the parent workspace:

```sh
python -m venv .venv-dr-clean
.venv-dr-clean/bin/python -m pip install -r DR_Exploration/requirements.txt
.venv-dr-clean/bin/python DR_Exploration/build_report.py
```

From a configured environment, you can also run from this directory:

```sh
python build_report.py
```

This creates and executes the notebook, fits PCA, four t-SNE configurations and baseline/crafted LAMP maps, computes metrics, exports figures and projected records, and embeds Plotly and all data in the explorer. The notebook can also execute directly if `vgsales.csv`, `lamp.py`, `lamp_core.js`, `lamp_ui.js`, and `explorer_template.html` are in its working directory. Execution requires a Python Jupyter kernel. No API keys, remote data access, CDN or web hosting are required.

## Canvas checklist

- Two DR methods, original-space comparisons, linked exploration, methodology, prompts and analysis are included.
- The current evaluator’s ratings are recorded as 100% for each member, including self. Each student still submits their own evaluation. The form is separate and unsubmitted.
- One student submits the team project. The notebook and companion files are bundled in `Video_Game_Sales_DR_Review_Bundle.zip` in the parent directory.
- LAMP bonus implemented after the user confirmed instructor approval on November 19, 2026. See the notebook bonus section and `lamp_comparison.png`.
- Nothing was uploaded or submitted to Canvas.

## Approved LAMP bonus

The notebook includes the LAMP methodology, mapping source, paper attribution, adaptation scope, metrics, before/after figure and reproduction steps. In the explorer, drag amber controls and release, or change numeric X/Y fields and click Move control. Example move reproduces the notebook’s crafted state. Reset LAMP restores the baseline. Switch to Brush releases to link a LAMP cohort to all existing views. The interface computes locally and works offline.

The user's confirmation of instructor approval was received before implementation. This is a scoped implementation of the paper’s core local orthogonal affine method, with 24 MDS-initialized controls and the same five numeric inputs as the main comparison. It does not claim to reproduce every feature of the HW4 proposal or the original research system.

Run numerical checks after building:

```sh
python check_lamp.py
```

This also runs `check_lamp.js` using Node.js to compare the independent browser solver with NumPy SVD on all 2,000 baseline and crafted coordinates. Node is needed only for this cross-language validation, not for viewing the explorer or executing its notebook.

## Where the AI disclosure and prompts belong

The final notebook contains the complete AI assistance and revised prompt strategy section because prompt strategy is part of the required notebook report. The HTML report contains the same section. Keep this section in the notebook when submitting.

Edit `PROMPT_STRATEGY.md` to change this text. Both Python builders read that file and embed its contents in the notebook; the HTML report is then exported from that notebook. The Markdown file is a convenient readable copy on GitHub, not a replacement for the notebook disclosure. Include it with the other source files when rebuilding.
