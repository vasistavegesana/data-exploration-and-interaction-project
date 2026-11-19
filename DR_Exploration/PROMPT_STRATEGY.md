## AI use and prompt strategy
I directed the project requests and reviewed the generated outputs. ChatGPT/Codex assisted with interpreting requirements, visualization planning, implementation, documentation, and debugging. The prompts below are my own wording, revised after development to describe the implemented workflow.

**Prompt 1: Project structure and reproducibility**
Help implement a reproducible ITCS 4123/5123 project titled “Video Game Sales: Data Exploration and Interaction.” Organize the repository with a root README and .gitignore, plus a DR_Exploration/ directory containing the dataset, requirements, Python source, executed Jupyter notebook, offline interactive explorer, HTML report, figures, metrics, and validation scripts.

Provide a combined Python file containing the notebook builder and LAMP algorithm. Document how to create a virtual environment, install dependencies, select its Jupyter kernel, and regenerate the outputs. Use NumPy 2.3.5 in the corrected numerical environment. Flag inconsistencies between dependency files and the tested environment.

**Prompt 2: Data cleaning and feature representation**
Load the existing vgsales.csv from the Gregory Smith Video Game Sales dataset. Treat each record as one game-platform release, rather than a unique game or franchise.

Convert year and sales fields to numeric values. Retain valid releases from 1980 through 2016, excluding records with missing required metadata, missing numerical values, or negative sales. Preserve the original source-row identifier for cross-view linking.

Take a deterministic uniform sample of up to 2,000 clean records using seed 42. Display the raw, cleaned, and sampled record counts, descriptive statistics, and genre frequencies.

Use five numerical inputs: North American, European, Japanese, and Other regional sales, plus release year. Apply log1p to regional sales, then standardize all five features. Exclude global sales and rank to avoid duplicating sales information. Keep genre and platform as interpretation labels rather than numerical inputs.

Feed exactly the same standardized matrix and release IDs into all projection methods. Explain how preprocessing, sampling, and feature selection affect the interpretation of similarity.

**Prompt 3: PCA and t-SNE**
Apply PCA with two components and the full SVD solver. Display component loadings, explained variance ratios, total retained variance, and reconstruction error in standardized feature space.

Apply two-dimensional t-SNE with Euclidean distance, PCA initialization, automatic learning rate, and 1,000 iterations. Use perplexity 30 and seed 42 for the primary map. Add sensitivity runs using perplexities 10 and 50 with seed 42, and perplexity 30 with seed 7.

Produce side-by-side PCA and t-SNE scatterplots colored by genre, with readable labels and a shared legend. Also show raw and transformed distributions for each input feature and a feature-correlation table.

Explain PCA’s emphasis on linear variance and reconstruction, versus t-SNE’s emphasis on neighborhood relationships. Avoid interpreting t-SNE island distances, areas, or gaps as direct evidence of real market categories.

**Prompt 4: Quantitative projection comparison**
Evaluate projection quality using three complementary measures:

Trustworthiness at k = 10, using scikit-learn.
Exact 10-neighbor overlap between original and projected spaces.
Spearman correlation between original and projected pairwise distances.
Use the complete embedded sample for neighborhood measures. For the distance-correlation calculation, use a fixed, seeded subset of up to 600 releases.

Exclude each record from its own neighbor list and explain the effect of tied observations. Export the results as CSV files and display a rounded comparison table.

Generate numerical statements from the computed results. Distinguish local preservation from global-distance preservation, and describe these as exploratory, in-sample diagnostics rather than predictive accuracy.

**Prompt 5: Independent LAMP implementation**
Implement the core method from Joia et al., “Local Affine Multidimensional Projection,” IEEE TVCG 17(12), 2011. Consult the authors’ reference implementation to check the approach, but write an independent implementation and cite the paper and reference repository.

Select 24 representative controls using deterministic farthest-point sampling, starting near the standardized-data centroid. Initialize their two-dimensional positions using classical metric MDS of the control-point distances.

Expose a function project(X, control_ids, targets) returning one two-dimensional coordinate per input record.

For each release, calculate normalized inverse-Euclidean-distance weights to the controls, weighted input and target centers, and the weighted cross-covariance matrix. Use its SVD to obtain the local orthogonal mapping, then project the centered release and restore the target center.

Use NumPy calculations with a loop over releases. Map exact control matches directly to their targets, validate input shapes and finite values, and handle degenerate configurations without producing NaNs.

Keep the control coordinates in their initial MDS scale. Explain the adaptation choices and distinguish this scoped implementation from a reproduction of the entire research system.

**Prompt 6: LAMP evaluation and reproducible example**
Calculate a baseline LAMP map using the initial MDS control positions. Create a reproducible crafted map by moving the control with the highest Japanese sales by (+2, −2) coordinate units.

Evaluate both maps with the same quality measures used for PCA and t-SNE. Export control identities, baseline and crafted coordinates, and quality metrics.

Produce before-and-after figures with identical axis limits and equal coordinate scaling. Highlight the moved control.

Verify that controls reach their requested targets, other releases move, and restoring the original targets reproduces the baseline.

Explain whether the measured quality improves or worsens. Present steering as an analyst-directed change that needs verification against original features, rather than assuming greater visual separation means better results.

**Prompt 7: Self-contained interactive explorer**
Generate Interactive_Explorer.html from the project’s computed data. Embed Plotly, JavaScript, and the sampled records so it works offline without a server or external CDN.

Include PCA, t-SNE, and an original-feature scatterplot. Support box/lasso selection and individual-point inspection, using shared release IDs to link selections across views.

Add genre selection, a minimum/maximum range selector for an original feature, and controls for changing the original scatterplot’s X and Y dimensions.

Show selected versus full-sample feature distributions, standardized feature profiles, and a release table. Display name, platform, genre, year, regional sales, and source-row ID. Clearly state any display limits on profile lines or table rows.

Preserve the full sample as visual context, distinguish an empty selection from the population, and provide a selection-reset button.

**Prompt 8: Interactive LAMP controls**
Add a LAMP panel to the same explorer. Show the 24 control releases as distinct amber markers and link ordinary releases to the existing selection state.

Provide two interaction modes: “Steer controls” and “Brush releases.” Allow users to drag a control or choose it from a selector and edit its coordinates.

Recompute all LAMP positions when a drag is released or a coordinate edit is applied. Include a button reproducing the notebook’s example movement and a separate button restoring the baseline control positions.

Preserve selected release IDs when resetting the LAMP map. Show whether the view is baseline or crafted.

Display current versus baseline 10-neighbor overlap using 200 fixed probe records, mean displacement of non-controls, and mapping time. Distinguish this probe-based diagnostic from full-sample notebook metrics.

Implement the browser’s local orthogonal mapping independently and compare its coordinates numerically with the NumPy SVD implementation within a stated tolerance.

**Prompt 9: Independent numerical validation**
Write Python and JavaScript validation scripts. Use Node.js for the cross-language comparison; it is required for that check but not for viewing the explorer.

Compare all baseline and crafted LAMP coordinates against the NumPy reference with an absolute tolerance of 1e-8. Test exact control interpolation, baseline restoration, rigid-transform reproduction, target-translation equivariance, and finite outputs for collapsed or collinear control targets.

Record the checks, tolerances, and observed discrepancies in VALIDATION.md. Separate automated numerical checks from browser interaction checks and report only checks actually performed.

**Prompt 10: Numerical-warning diagnosis and output cleanup**
Investigate the matrix-multiplication RuntimeWarnings rather than simply deleting or suppressing their output. Reproduce the behavior on a small example, inspect the numerical environment, and check whether the resulting arrays are finite.

Test NumPy 2.3.5 in an isolated project environment, leaving the main Anaconda installation unchanged. Regenerate the notebook using the verified environment.

Promote unexpected RuntimeWarnings to errors and add explicit finite-value checks for inputs, projections, and metrics. Use the supported IPython iframe display mechanism.

Inspect saved cell outputs for warnings and errors before packaging. Add clear kernel-selection instructions and a brief explanation of the runtime correction.

**Prompt 11: Notebook, interpretation, and final review**
Generate the final executed Jupyter notebook and readable HTML report. Include a reading guide, methodology, preprocessing rationale, figures, computed findings, interactive instructions, LAMP implementation, before-and-after evaluation, references, and this AI-assistance disclosure.

Add reproducible exploration examples: inspect releases with Japanese sales of at least one million units, brush a projection region and compare its original features, and inspect a single release.

Explain limitations involving historical coverage, digital-sales underrepresentation, repeated titles across platforms, sampling, preprocessing, and projection distortion. Avoid claims about current demand, causation, natural clusters, or user-study results that the project does not establish.

Review the documentation against the implementation, identify unsupported claims or dependency inconsistencies, and verify the notebook and supporting files needed to reproduce the project. Package the deliverables for review without submitting anything to Canvas.
