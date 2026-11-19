// Canvas interaction layer; computation is isolated in lamp_core.js.
let lampPositions=D.lamp.map(p=>p.slice()),lampTargets=D.targets.map(p=>p.slice()),activeControl=0;
const lampCanvas=$('lampCanvas'),ctx=lampCanvas.getContext('2d');
let lampDrag=null,lampBrush=null,lampDirty=false;
const allXs=D.lamp.map(p=>p[0]),allYs=D.lamp.map(p=>p[1]);
let view={xmin:Math.min(...allXs)-3,xmax:Math.max(...allXs)+3,ymin:Math.min(...allYs)-3,ymax:Math.max(...allYs)+3};
D.controls.forEach((i,k)=>{const o=document.createElement('option');o.value=k;o.textContent=(k+1)+'. '+rec[i].Name+' · '+rec[i].Platform;$('lampControl').append(o)});
function canvasPoint(event){let r=lampCanvas.getBoundingClientRect();return [(event.clientX-r.left)*lampCanvas.width/r.width,(event.clientY-r.top)*lampCanvas.height/r.height]}
function lampScale(){return Math.min((lampCanvas.width-120)/(view.xmax-view.xmin),(lampCanvas.height-80)/(view.ymax-view.ymin))}
function screen(p){const s=lampScale();return [lampCanvas.width/2+(p[0]-(view.xmin+view.xmax)/2)*s,lampCanvas.height/2-(p[1]-(view.ymin+view.ymax)/2)*s]}
function fromScreen(p){const s=lampScale();return [(p[0]-lampCanvas.width/2)/s+(view.xmin+view.xmax)/2,(lampCanvas.height/2-p[1])/s+(view.ymin+view.ymax)/2]}
function drawLamp(){
 ctx.clearRect(0,0,lampCanvas.width,lampCanvas.height);ctx.fillStyle='#fafbfd';ctx.fillRect(0,0,lampCanvas.width,lampCanvas.height);
 ctx.strokeStyle='#dce3eb';ctx.lineWidth=1;ctx.font='11px system-ui';ctx.fillStyle='#566b80';
 const low=fromScreen([40,lampCanvas.height-30]),hi=fromScreen([lampCanvas.width-20,20]);
 let step=Math.pow(10,Math.floor(Math.log10((hi[0]-low[0])/8)));if((hi[0]-low[0])/step>15)step*=2;
 for(let x=Math.ceil(low[0]/step)*step;x<=hi[0];x+=step){let p=screen([x,0]);ctx.beginPath();ctx.moveTo(p[0],20);ctx.lineTo(p[0],lampCanvas.height-30);ctx.stroke();ctx.fillText(x.toFixed(step<1?1:0),p[0]+3,lampCanvas.height-12)}
 for(let y=Math.ceil(low[1]/step)*step;y<=hi[1];y+=step){let p=screen([0,y]);ctx.beginPath();ctx.moveTo(40,p[1]);ctx.lineTo(lampCanvas.width-20,p[1]);ctx.stroke();ctx.fillText(y.toFixed(step<1?1:0),7,p[1]+3)}
 // Ghost baseline stays visible when the analyst crafts a view.
 if(lampDirty){ctx.fillStyle='rgba(105,120,140,.12)';D.lamp.forEach(p=>{let q=screen(p);ctx.beginPath();ctx.arc(q[0],q[1],2,0,Math.PI*2);ctx.fill()})}
 [false,true].forEach(on=>{ctx.fillStyle=on?'rgba(22,121,139,.72)':'rgba(170,184,198,.25)';lampPositions.forEach((p,i)=>{if(selected.has(i)!==on)return;let q=screen(p);ctx.beginPath();ctx.arc(q[0],q[1],on?3:2,0,Math.PI*2);ctx.fill()})});
 lampTargets.forEach((p,k)=>{let q=screen(p);ctx.fillStyle=k===activeControl?'#eeaa3a':'#d58623';ctx.strokeStyle=k===activeControl?'#b3294b':'#172d44';ctx.lineWidth=k===activeControl?3:1;ctx.beginPath();ctx.arc(q[0],q[1],k===activeControl?8:6,0,Math.PI*2);ctx.fill();ctx.stroke();if(k===activeControl){ctx.fillStyle='#172d44';ctx.fillText('Control '+(k+1),q[0]+12,q[1]-10)}});
 if(lampBrush){ctx.fillStyle='rgba(22,121,139,.12)';ctx.strokeStyle='#16798b';let [a,b]=lampBrush;ctx.fillRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1])}
 ctx.fillStyle='#172d44';ctx.fillText('LAMP 1 →',lampCanvas.width-90,lampCanvas.height-12);ctx.fillText('LAMP 2 ↑',7,14);
}
function controlFields(){ $('lampControl').value=activeControl;$('lampX').value=lampTargets[activeControl][0].toFixed(4);$('lampY').value=lampTargets[activeControl][1].toFixed(4);$('lampDetail').textContent='Active control '+(activeControl+1)+': '+text(rec[D.controls[activeControl]])+' · Amber circles are controls; teal points are selected releases.';}
function overlap(Z){let total=0;D.probeIds.forEach((id,k)=>{const p=Z[id];let near=Z.map((q,j)=>[j,(q[0]-p[0])**2+(q[1]-p[1])**2]).filter(a=>a[0]!==id).sort((a,b)=>a[1]-b[1]||a[0]-b[0]).slice(0,10);const original=new Set(D.probeNeighbors[k]);total+=near.filter(a=>original.has(a[0])).length});return total/(10*D.probeIds.length)}
function quality(ms){let score=overlap(lampPositions),controls=new Set(D.controls),moved=0,count=0;lampPositions.forEach((p,i)=>{if(!controls.has(i)){moved+=Math.hypot(p[0]-D.lamp[i][0],p[1]-D.lamp[i][1]);count++}});$('lampQuality').textContent='10-neighbor overlap · Baseline '+(100*D.probeBaseline).toFixed(1)+'% → Current '+(100*score).toFixed(1)+'% · 200 fixed probes / 2,000 releases · Mean non-control displacement '+(moved/count).toFixed(3)+' MDS units · Last mapping '+ms.toFixed(1)+' ms';}
function updateLamp(name){const start=performance.now();lampPositions=projectLamp(D.z,D.controls,lampTargets);const elapsed=performance.now()-start;lampDirty=lampTargets.some((p,i)=>p.some((v,j)=>Math.abs(v-D.targets[i][j])>1e-9));
 // Expand bounds only when needed. Identical axis scaling avoids visual stretching.
 const xs=lampPositions.map(p=>p[0]),ys=lampPositions.map(p=>p[1]);view.xmin=Math.min(view.xmin,Math.min(...xs)-1);view.xmax=Math.max(view.xmax,Math.max(...xs)+1);view.ymin=Math.min(view.ymin,Math.min(...ys)-1);view.ymax=Math.max(view.ymax,Math.max(...ys)+1);
 $('lampState').textContent=lampDirty?'Crafted view · '+name:'Baseline · MDS controls';quality(elapsed);controlFields();drawLamp();}
function inspectLamp(p){let best=-1,distance=100;lampPositions.forEach((v,i)=>{let q=screen(v),dd=(q[0]-p[0])**2+(q[1]-p[1])**2;if(dd<distance){distance=dd;best=i}});if(best>=0){selected=new Set([best]);source='Individual LAMP release';sync()}}
lampCanvas.addEventListener('pointerdown',event=>{event.preventDefault();let p=canvasPoint(event);lampCanvas.setPointerCapture(event.pointerId);if($('lampMode').value==='steer'){let nearest=-1,distance=225;lampTargets.forEach((v,k)=>{let q=screen(v),dd=(q[0]-p[0])**2+(q[1]-p[1])**2;if(dd<distance){nearest=k;distance=dd}});if(nearest>=0){activeControl=nearest;lampDrag={start:p,slot:nearest};controlFields();drawLamp()}else inspectLamp(p)}else{lampBrush=[p,p];drawLamp()}});
lampCanvas.addEventListener('pointermove',event=>{let p=canvasPoint(event);if(lampDrag){lampTargets[lampDrag.slot]=fromScreen(p);controlFields();drawLamp()}else if(lampBrush){lampBrush[1]=p;drawLamp()}});
lampCanvas.addEventListener('pointerup',event=>{if(lampDrag){lampTargets[lampDrag.slot]=fromScreen(canvasPoint(event));lampDrag=null;updateLamp('control moved by analyst')}else if(lampBrush){const [a,b]=lampBrush;lampBrush=null;if(Math.hypot(a[0]-b[0],a[1]-b[1])<5){inspectLamp(b);drawLamp();return}selected=new Set(ids.filter(i=>{const q=screen(lampPositions[i]);return q[0]>=Math.min(a[0],b[0])&&q[0]<=Math.max(a[0],b[0])&&q[1]>=Math.min(a[1],b[1])&&q[1]<=Math.max(a[1],b[1])}));source='Brushed LAMP cohort';sync()}});
lampCanvas.addEventListener('pointercancel',()=>{lampDrag=null;lampBrush=null;updateLamp('drag interrupted')});
$('lampControl').onchange=()=>{activeControl=Number($('lampControl').value);controlFields();drawLamp()};
$('moveControl').onclick=()=>{const a=$('lampX').value,b=$('lampY').value,p=[Number(a),Number(b)];if(a===''||b===''||p.some(v=>!Number.isFinite(v)||Math.abs(v)>1000)){$('lampState').textContent='Enter finite X/Y coordinates between −1000 and 1000.';return}lampTargets[activeControl]=p;updateLamp('control coordinates changed')};
$('lampExample').onclick=()=>{lampTargets=D.exampleTargets.map(p=>p.slice());activeControl=D.targets.findIndex((p,i)=>p.some((v,j)=>v!==D.exampleTargets[i][j]));updateLamp('example control moved +2, −2')};
$('lampReset').onclick=()=>{lampTargets=D.targets.map(p=>p.slice());view={xmin:Math.min(...allXs)-3,xmax:Math.max(...allXs)+3,ymin:Math.min(...allYs)-3,ymax:Math.max(...allYs)+3};updateLamp('reset')};
new ResizeObserver(()=>{const r=lampCanvas.getBoundingClientRect();lampCanvas.width=Math.round(r.width);lampCanvas.height=Math.round(r.height);drawLamp()}).observe(lampCanvas);
controlFields();quality(0);drawLamp();
