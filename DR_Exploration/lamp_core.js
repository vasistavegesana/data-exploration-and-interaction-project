/* Independent browser LAMP solver. Joia et al., TVCG 2011, doi:10.1109/TVCG.2011.220.
 * Thin polar/SVD factor of a d x 2 cross-covariance via its 2 x 2 Gram matrix.
 * Positive inverse-Euclidean weights match the Python reference configuration.
 */
(function(root){
function projectLamp(X, controls, Y){
  const C=controls.map(i=>X[i]), d=X[0].length;
  return X.map(x=>{
    const distances=C.map(c=>Math.hypot(...c.map((v,j)=>v-x[j])));
    const match=distances.findIndex(v=>v<=1e-10);
    if(match>=0)return Y[match].slice();
    let w=distances.map(v=>1/v), sum=w.reduce((a,b)=>a+b,0);w=w.map(v=>v/sum);
    const xb=Array(d).fill(0),yb=[0,0];
    C.forEach((c,k)=>{c.forEach((v,j)=>xb[j]+=w[k]*v);yb[0]+=w[k]*Y[k][0];yb[1]+=w[k]*Y[k][1]});
    const H=Array.from({length:d},()=>[0,0]);
    C.forEach((c,k)=>c.forEach((v,j)=>{H[j][0]+=w[k]*(v-xb[j])*(Y[k][0]-yb[0]);H[j][1]+=w[k]*(v-xb[j])*(Y[k][1]-yb[1])}));
    let a=0,b=0,c=0;H.forEach(h=>{a+=h[0]**2;b+=h[0]*h[1];c+=h[1]**2});
    const angle=.5*Math.atan2(2*b,a-c),v1=[Math.cos(angle),Math.sin(angle)],v2=[-v1[1],v1[0]];
    let u1=H.map(h=>h[0]*v1[0]+h[1]*v1[1]),u2=H.map(h=>h[0]*v2[0]+h[1]*v2[1]);
    const norm=v=>Math.hypot(...v),n1=norm(u1);
    u1=n1>1e-12?u1.map(v=>v/n1):Array.from({length:d},(_,j)=>j===0?1:0);
    const dot=u2.reduce((s,v,j)=>s+v*u1[j],0);u2=u2.map((v,j)=>v-dot*u1[j]);
    let n2=norm(u2);
    if(n2<=Math.max(1e-12,n1*1e-12)){
      let axis=0;for(let j=1;j<d;j++)if(Math.abs(u1[j])<Math.abs(u1[axis]))axis=j;
      u2=u1.map((v,j)=>(j===axis?1:0)-u1[axis]*v);n2=norm(u2);
    }
    u2=u2.map(v=>v/n2);
    const q1=x.reduce((s,v,j)=>s+(v-xb[j])*u1[j],0),q2=x.reduce((s,v,j)=>s+(v-xb[j])*u2[j],0);
    return [q1*v1[0]+q2*v2[0]+yb[0],q1*v1[1]+q2*v2[1]+yb[1]];
  });
}
if(typeof module!=='undefined'&&module.exports)module.exports={projectLamp};else root.projectLamp=projectLamp;
})(typeof globalThis!=='undefined'?globalThis:this);
