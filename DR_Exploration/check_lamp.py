"""Numerical checks for the projection method; run after build_report.py."""
from pathlib import Path
import json
import subprocess
import numpy as np
from lamp import project

rng = np.random.default_rng(5)
X = rng.normal(size=(100,2))
controls = np.array([0,5,13,30,45,70])
angle = .7
rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
expected = X@rotation+[2,-3]
actual = project(X,controls,expected[controls])
assert np.allclose(actual,expected,atol=1e-10), 'Rigid 2D transformation must reproduce exactly'
translated = project(X,controls,expected[controls]+[7,4])
assert np.allclose(translated,actual+[7,4],atol=1e-10), 'Target translation must translate the map'
zero = project(X,controls,np.zeros((len(controls),2)))
assert np.isfinite(zero).all(), 'Collapsed targets must not produce NaNs'
print('Python: rigid transform, translation equivariance, collapsed-target finiteness passed')

root=Path(__file__).resolve().parent
result=subprocess.run(['node',str(root/'check_lamp.js')],cwd=root,text=True,capture_output=True,check=True)
print(result.stdout)
