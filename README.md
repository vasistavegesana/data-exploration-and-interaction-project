# Video Game Sales — Data Exploration and Interaction

ITCS 4123/5123 course project by  **Krish Thakre, and Vasista Vegesana**.

Explore how game-platform releases compare in their original regional-sales/year features, PCA, t-SNE, and an interactive Local Affine Multidimensional Projection (LAMP) extension.

## Start here

- **[Final executed Jupyter notebook](DR_Exploration/Video_Game_Sales_DR.ipynb)** — methods, outputs, comparisons, prompts, references, and LAMP bonus.
- **[Download the complete project bundle](Video_Game_Sales_DR_With_LAMP_Bonus.zip)** — notebook, code, dataset, report and supporting files.
- **[Combined Python code](DR_Exploration/Video_Game_Sales_DR_Code.py)** — notebook builder and LAMP algorithm in one file.
- **[Interactive explorer](DR_Exploration/Interactive_Explorer.html)** — download and open locally in a browser.
- **[Readable HTML report](DR_Exploration/Video_Game_Sales_DR_Report.html)** — download and open locally.

GitHub previews the notebook. To use the interactive controls, download the ZIP, extract it, and open `DR_Exploration/Interactive_Explorer.html`. It includes Plotly and its data, so it works offline. This repository does not automatically publish a live website.

## Features

- Identical standardized inputs for PCA and t-SNE; four regional log-sales dimensions plus release year.
- Linked brushing, original-feature filters, individual release inspection, distributions and feature profiles.
- LAMP with 24 MDS-initialized control releases, drag or coordinate editing, example movement, baseline reset and live neighborhood diagnostics.
- Reproducible metrics, static before/after figures and independent Python/JavaScript numerical checks.

## Run

Use Python 3.11 or later and the pinned numerical environment. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r DR_Exploration/requirements.txt
python DR_Exploration/Video_Game_Sales_DR_Code.py
```

In Jupyter or VS Code, select that environment before restarting the kernel and running all cells. NumPy 2.3.5 fixes the matrix-warning behavior observed in the original Anaconda environment on the development Mac; unexpected numerical warnings now stop execution, and all maps and metrics are checked for finite values. Saved outputs are already executed.

To run the independent numerical checks (Node.js is required only for the JavaScript comparison):

```sh
python DR_Exploration/check_lamp.py
```

See [project instructions](DR_Exploration/README.md) and [validation notes](DR_Exploration/VALIDATION.md).

## Data and attribution

The local `vgsales.csv` is reused from the earlier course project based on Gregory Smith’s [Video Game Sales dataset](https://www.kaggle.com/datasets/gregorut/videogamesales). Results describe a historical snapshot through 2016, not current market demand. Each row represents a game-platform release.

LAMP follows Joia et al., **Local Affine Multidimensional Projection**, IEEE TVCG 17(12), 2563–2571 (2011), [DOI](https://doi.org/10.1109/TVCG.2011.220). The authors’ [reference implementation](https://github.com/lgnonato/LAMP) informed an independent implementation. AI assistance and prompts are disclosed in the notebook.

Repository upload is separate from Canvas submission.
