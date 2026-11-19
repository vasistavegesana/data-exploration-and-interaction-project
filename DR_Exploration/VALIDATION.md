# Validation — November 19, 2026

- Notebook executed top to bottom: all eight code cells executed, zero error outputs.
- Compared PCA and four t-SNE configurations on identical input records; metrics exported.
- Inspected both static figure files for readable axes, legends, and layout.
- Verified local offline explorer renders all five plots without browser console errors.
- Verified original-feature range JP_Sales >= 1 selects 28 of 2,000 releases across views.
- Verified clicking Super Mario Kart selects one source record (row 284).
- Verified t-SNE box brushing selects 333 records and updates linked views/table.
- Verified JP_Sales >= 1000 produces an explicit empty cohort, and reset restores 2,000.
- Verified Role-Playing genre cohort selects 171 records; changing original X updates its axis.
- Corrected profile y range to include all standardized values, including early years.
- Retained all three team names and recorded user-requested 100% ratings for each.
- Nothing uploaded or submitted to Canvas.

Browser screenshot: explorer_verified.jpg.

## LAMP bonus validation

The user confirmed instructor approval before implementation.

- Independent Python SVD and browser orthogonal-factor solvers agree on all 2,000 records in baseline and crafted maps (maximum absolute discrepancy 6.22e-15).
- Verified exact target coordinates at all 24 controls, deterministic baseline restoration, rigid-transform reproduction, target-translation equivariance, and finite results for collapsed/collinear targets.
- Executed final notebook top to bottom: eight code cells, zero errors.
- Inspected before/after static figure, with shared coordinate limits and equal axis scaling.
- Browser: Example move recomputes the map, changing 200-probe overlap from 30.9% to 30.8% and moving non-controls by 0.066 MDS units on average.
- Browser: dragged a control; the crafted view and diagnostics changed.
- Browser: brushed a LAMP cohort (1,925 records), verified shared selection count and table, then reset LAMP. Selection identities persisted while baseline quality and zero displacement returned.
- Browser: numeric Control X changed to -3 and Move control recomputed the map; reset again restored baseline.
- Final browser build has zero observed console errors. Screenshot: lamp_verified.jpg.
- No upload, email, or Canvas submission occurred.

## Numerical warning cleanup

- Reproduced original NumPy 1.26.4/OpenBLAS configuration with 33 RuntimeWarnings on a small PCA/t-SNE test; its outputs were finite.
- The same test with project-local NumPy 2.3.5 emitted zero warnings. Main Anaconda packages were not changed.
- Regenerated the complete notebook with NumPy 2.3.5: all eight code cells executed, zero stderr outputs, zero warnings and zero errors.
- Unexpected RuntimeWarnings are promoted to errors, not filtered out. Explicit finite-value checks cover transformed inputs, all PCA/t-SNE/LAMP maps and the main metric table.
- Replaced HTML iframe display with the supported IPython IFrame object, removing the iframe display warning.
- Added a reading guide and environment instructions. Saved notebook can be read without rerunning. For reruns, choose .venv-dr-clean/bin/python and restart the kernel.
- Rechecked the independent LAMP solvers and numerical invariants successfully after runtime update.
- Exported the updated HTML report and refreshed both review ZIP archives.

## Repository consistency repair — November 19, 2026

- Preserved every Markdown cell from Krish’s `322b0c2` notebook exactly (8 of 8 cells). SHA256 of the JSON-serialized Markdown list: `f21143d3627bac55a9ecfaacf920520d7baf5d454a116d1e363a68740a9d27f5`. No editorial changes were made.
- Installed the pinned direct dependencies into a fresh virtual environment without system site-packages. `pip check` passed. Tested Python 3.13.9 and NumPy 2.3.5.
- Executed the documented combined-builder command from the repository root: all 8 code cells executed, with zero warning/error outputs. The builder uses the selected Python interpreter and cleans up its kernel.
- Corrected a Python escape literal in the explorer-generation code using a raw string, avoiding a SyntaxWarning without changing the generated replacement text.
- LAMP Python/JavaScript checks passed; maximum absolute discrepancy was 6.22e-15. Control interpolation, baseline restoration, rigid transforms, translation and degenerate cases passed.
- LAMP metrics matched the prior results exactly; the largest PCA/t-SNE diagnostic difference was 8.86e-11, below reporting precision.
- Both documented builder entry points resolve to the same build function. The final notebook is the build input, and its Markdown is checked for preservation. The combined source holds the LAMP implementation and inserts its code into the notebook.
- Both READMEs have no missing local Markdown-link targets. The ZIP passed its integrity check, contains `Video_Game_Sales_FINAL_DR.ipynb`, preserves Krish’s wording, and excludes the deleted notebook filename.
- The HTML report is exported from the final notebook. The separate prompt document is exported from its existing prompt section. The removed contribution-evaluation section is not reinserted; the separate evaluation file remains.
- The offline explorer contains embedded scripts/data and has no external script dependency or unresolved generation placeholders.
- No Canvas submission was made.
