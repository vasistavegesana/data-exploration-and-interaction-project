"""Local orthogonal affine fitting, following Joia et al. (TVCG 2011).

Independent NumPy implementation; inverse-Euclidean weighting follows the
authors' reference implementation at https://github.com/lgnonato/LAMP.
No optional automatic rescaling: controls stay in the initial metric-MDS units.
"""
import numpy as np
from scipy.spatial.distance import cdist


def choose_controls(X, count=24):
    """Deterministic farthest-point coverage, starting near the data centroid."""
    X = np.asarray(X, dtype=float)
    chosen = [int(np.argmin(np.sum((X-X.mean(axis=0))**2, axis=1)))]
    nearest = np.full(len(X), np.inf)
    while len(chosen) < min(count, len(X)):
        nearest = np.minimum(nearest, np.sum((X-X[chosen[-1]])**2, axis=1))
        nearest[chosen] = -1
        nxt = int(np.argmax(nearest))
        if nearest[nxt] <= 1e-16:
            break
        chosen.append(nxt)
    return np.array(chosen)


def classical_mds(X):
    """Classical metric MDS of Euclidean control distances into two dimensions."""
    squared = cdist(X, X, 'sqeuclidean')
    centered = -.5*(squared-squared.mean(axis=0)-squared.mean(axis=1)[:,None]+squared.mean())
    vals, vecs = np.linalg.eigh(centered)
    order = np.argsort(vals)[::-1][:2]
    return vecs[:,order]*np.sqrt(np.maximum(vals[order],0))


def project(X, control_ids, targets):
    """Fit M per input record with M.T @ M = I and map about weighted centers.

    Given C= X[control_ids], each local cross covariance is
    H = (C-xbar).T @ diag(w) @ (targets-ybar). For H=U S V.T,
    the orthogonal least-squares solution is M=U @ V.T.
    Exact control matches interpolate the supplied target without division by 0.
    """
    X = np.asarray(X, dtype=float)
    control_ids = np.asarray(control_ids, dtype=int)
    targets = np.asarray(targets, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2 or targets.shape != (len(control_ids),2):
        raise ValueError('Expected N x d input (d>=2) and m x 2 control targets')
    if len(control_ids) < 3 or len(set(control_ids)) != len(control_ids):
        raise ValueError('Use at least three distinct control IDs')
    if not np.isfinite(X).all() or not np.isfinite(targets).all():
        raise ValueError('Inputs and controls must be finite')
    C = X[control_ids]
    distances = cdist(X,C)
    result = np.empty((len(X),2))
    for i,x in enumerate(X):
        close = np.flatnonzero(distances[i] <= 1e-10)
        if len(close):
            result[i] = targets[close[0]]
            continue
        weights = 1 / distances[i]
        weights /= weights.sum()
        xbar = weights @ C
        ybar = weights @ targets
        H = (C-xbar).T @ (weights[:,None]*(targets-ybar))
        U, _, Vt = np.linalg.svd(H, full_matrices=False)
        result[i] = (x-xbar) @ (U@Vt)+ybar
    return result


# === Notebook and report build runtime ===
import hashlib
import inspect
import json
import sys
import zipfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'Video_Game_Sales_FINAL_DR.ipynb'


def markdown_digest(notebook):
    """Protect the final notebook's editorial content during regeneration."""
    text = json.dumps([cell.source for cell in notebook.cells if cell.cell_type == 'markdown'], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def sync_prompt_copy(notebook):
    """Export the notebook's existing wording; never replace it with older text."""
    for cell in notebook.cells:
        if cell.cell_type != 'markdown':
            continue
        for heading in ('## AI use and prompt strategy', '## AI assistance and revised prompt strategy'):
            if heading in cell.source:
                section = cell.source.split(heading, 1)[1]
                for boundary in ('### Numerical validation and runtime', '## References'):
                    section = section.split(boundary, 1)[0]
                (ROOT / 'PROMPT_STRATEGY.md').write_text(heading + section.rstrip() + '\n')
                return
    raise RuntimeError('The final notebook is missing its AI disclosure section.')


def package_project():
    """Package exactly the final notebook and its current supporting files."""
    output = ROOT.parent / 'Video_Game_Sales_DR_With_LAMP_Bonus.zip'
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.iterdir()):
            if path.is_file() and path.name != '.DS_Store':
                archive.write(path, 'DR_Exploration/' + path.name)
        for filename in ('README.md', '.gitignore'):
            path = ROOT.parent / filename
            if path.is_file():
                archive.write(path, filename)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Project ZIP integrity check failed.')
    return output


def build():
    """Execute the canonical notebook without altering any Markdown cell."""
    if np.lib.NumpyVersion(np.__version__) < '2.3.5':
        raise RuntimeError('Install requirements.txt in the selected Python environment. NumPy >= 2.3.5 is required.')
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f'Keep {NOTEBOOK.name} beside this builder; it contains the final source and wording.')
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    before = markdown_digest(notebook)
    # Keep the visible notebook algorithm synchronized with this combined source.
    lamp_source = 'import numpy as np\nfrom scipy.spatial.distance import cdist\n\n' + '\n\n'.join(
        inspect.getsource(function) for function in (choose_controls, classical_mds, project))
    matches = [cell for cell in notebook.cells if cell.cell_type == 'code' and
               'def choose_controls(' in cell.source and 'def project(' in cell.source]
    if len(matches) != 1:
        raise RuntimeError('Expected one LAMP implementation cell in the final notebook.')
    matches[0].source = lamp_source
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv[0] = sys.executable
    try:
        NotebookClient(notebook, km=manager, timeout=600, kernel_name='python3',
                       resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    issues = [(i, output) for i, cell in enumerate(notebook.cells) if cell.cell_type == 'code'
              for output in cell.outputs if output.output_type == 'error' or
              (output.output_type == 'stream' and
               (output.name == 'stderr' or 'Warning' in output.text or 'Traceback' in output.text))]
    if issues:
        raise RuntimeError(f'Notebook has warning/error output in cells {[i for i, _ in issues]}; outputs were not published.\n' + '\n'.join(str(output.get('text', output))[:2000] for _, output in issues))
    if before != markdown_digest(notebook):
        raise RuntimeError('Notebook Markdown changed during execution; outputs were not published.')
    template_dirs = [str(Path(prefix) / 'share/jupyter/nbconvert/templates')
                     for prefix in dict.fromkeys((sys.prefix, sys.base_prefix))
                     if (Path(prefix) / 'share/jupyter/nbconvert/templates').is_dir()]
    report, _ = HTMLExporter(extra_template_basedirs=template_dirs,
                             extra_template_paths=template_dirs).from_notebook_node(notebook)
    nbformat.write(notebook, NOTEBOOK)
    (ROOT / 'Video_Game_Sales_DR_Report.html').write_text(report)
    sync_prompt_copy(notebook)
    archive = package_project()
    print(f'Built {NOTEBOOK.name}, HTML report, offline explorer and {archive.name}.')
    print('Verified: all Markdown preserved; no warning/error outputs.')


if __name__ == '__main__':
    build()
