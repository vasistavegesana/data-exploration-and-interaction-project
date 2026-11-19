const fs=require('node:fs'),assert=require('node:assert/strict');
const {projectLamp}=require('./lamp_core.js');
const D=JSON.parse(fs.readFileSync('lamp_validation.json','utf8'));
let errors=[];
for(const [targets,expected] of [[D.targets,D.expected],[D.steered,D.expectedSteered]]){
 const out=projectLamp(D.x,D.ids,targets);
 const error=Math.max(...out.flatMap((p,i)=>p.map((v,j)=>Math.abs(v-expected[i][j]))));
 assert.ok(error<1e-8,'Browser solver differs from NumPy SVD: '+error);errors.push(error);
 D.ids.forEach((id,k)=>assert.deepEqual(out[id],targets[k]));
}
const reset=projectLamp(D.x,D.ids,D.targets);
assert.ok(Math.max(...reset.flatMap((p,i)=>p.map((v,j)=>Math.abs(v-D.expected[i][j]))))<1e-8);
const collapsed=projectLamp(D.x,D.ids,D.ids.map(()=>[0,0]));
assert.ok(collapsed.flat().every(Number.isFinite));
const line=projectLamp(D.x,D.ids,D.targets.map(p=>[p[0],0]));
assert.ok(line.flat().every(Number.isFinite));
console.log('JS vs NumPy: baseline and steered maps passed; max errors',errors);
console.log('Control interpolation, reset, rank-zero and rank-one finiteness passed');
