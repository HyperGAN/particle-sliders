'use strict';
const $=s=>document.querySelector(s), el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
if(matchMedia('(max-width:820px)').matches)$('#atmosphere-panel').open=false;
let names=[],retiredNames=new Set();let images=[],referenceImages=[],selected=null,controls={},historyLimit=48,refreshTimer=null;const blindOrder=new Map(),checkpointCatalog=new Map(),latestCheckpoints=new Map(),modelCatalog=new Map();
async function api(path,body){const r=await fetch('/api/'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
function notice(text){$('#notice').textContent=text;}
function drawMixers(variations){
 if(JSON.stringify(names)===JSON.stringify(variations))return;
 const previous=Object.fromEntries(names.map(n=>[n,$('#'+n).value]));
 names=[...variations];$('#mixers').replaceChildren();
 for(const name of names){const row=el('div',undefined,'mix-row'),head=el('div',undefined,'section-label'),label=el('label',name[0].toUpperCase()+name.slice(1)),value=el('output','0.00');label.htmlFor=name;value.id=name+'-value';const maxButton=el('button','Max','mix-max');maxButton.type='button';maxButton.setAttribute('aria-label','Set '+label.textContent+' to maximum');maxButton.onclick=()=>{$('#'+name).value='1';updateMix();};head.append(label,value,maxButton);const slider=el('input');Object.assign(slider,{type:'range',min:0,max:1,step:.01,value:previous[name]||0,id:name});slider.addEventListener('input',updateMix);const availability=el('small',undefined,'mix-availability');availability.id=name+'-availability';row.append(head,slider,availability);$('#mixers').append(row);}
 updateMix();
}
function updateMix(){
 const total=names.reduce((s,n)=>s+Number($('#'+n).value),0),energy=Number($('#energy').value);
 $('#energy-value').textContent=energy.toFixed(2);$('#energy').style.setProperty('--fill',(energy/Number($('#energy').max)*100)+'%');
 for(const n of names){const slider=$('#'+n),value=Number(slider.value);$('#'+n+'-value').textContent=value.toFixed(2);slider.style.setProperty('--fill',(value*100)+'%');slider.closest('.mix-row').classList.toggle('is-active',value>0);}
 $('#effective').textContent=total?'Energy '+energy.toFixed(2)+' shared as: '+names.map(n=>titleCase(n)+' '+(energy*Number($('#'+n).value)/total).toFixed(2)).join(' · '):'All faders at zero: adapters are off.';
}
$('#energy').addEventListener('input',updateMix);updateMix();
function updateResolutions(){
 const model=modelCatalog.get($('#model').value),select=$('#resolution'),current=select.value;
 const presets=model?.resolution_presets?.length?model.resolution_presets:[{label:'Square',width:model?.default_size||768,height:model?.default_size||768,icon:'□'}];
 select.replaceChildren(...presets.map(p=>new Option(p.icon+' '+p.label+' · '+p.width+' × '+p.height,p.width+'x'+p.height)));
 const preferred=model?.default_size+'x'+model?.default_size;
 select.value=presets.some(p=>p.width+'x'+p.height===current)?current:presets.some(p=>p.width+'x'+p.height===preferred)?preferred:select.options[0].value;
}
$('#model').addEventListener('change',updateResolutions);
function settings(){
 const mix={},checkpoints={};for(const n of names){mix[n]=Number($('#'+n).value);const sha=latestCheckpoints.get(n)?.sha256;if(sha)checkpoints[n]=sha;}
 const [width,height]=$('#resolution').value.split('x').map(Number);
 return {model:$('#model').value,prompt:$('#prompt').value,seed:$('#seed').value===''?null:Number($('#seed').value),width,height,steps:Number($('#steps').value),energy:Number($('#energy').value),mix,checkpoints};
}
async function enqueue(mode){try{if(!$('#generate').reportValidity())return;const req=settings();if(mode)req.mode=mode;const result=await api(mode?'comparisons':'generate',req);notice(result.ids.length+' image'+(result.ids.length===1?'':'s')+' queued.');await refresh();}catch(e){notice(e.message);}}
$('#generate').addEventListener('submit',e=>{e.preventDefault();enqueue();});$('#off-on').onclick=()=>enqueue('off_on');$('#sweep').onclick=()=>enqueue('sweep');
async function showTab(){const tab=['gallery','queue','training','definitions'].includes(location.hash.slice(1))?location.hash.slice(1):'gallery';for(const b of document.querySelectorAll('nav button'))b.classList.toggle('active',b.dataset.tab===tab);for(const v of document.querySelectorAll('.view'))v.hidden=v.id!==tab;if(tab==='training'){drawRuns();await refreshTraining();}if(location.hash&&matchMedia('(max-width:820px)').matches)document.querySelector('nav').scrollIntoView({block:'start'});}
for(const button of document.querySelectorAll('nav button'))button.onclick=()=>{if(location.hash==='#'+button.dataset.tab)showTab();else location.hash=button.dataset.tab;};
window.addEventListener('hashchange',showTab);
function imageLabel(m){if($('#blind').checked)return 'Image '+m.job_id.slice(0,6);if(m.reference_side)return (m.candidate_label?m.candidate_label+' lighting preview · ':'')+(m.reference_side==='neutral'?'Neutral':'Positive')+' reference · '+m.definition;const active=Object.entries(m.strengths).filter(([n,v])=>v>0);const label=active.length?'Energy '+m.energy.toFixed(2)+' · '+active.map(([n,v])=>n[0].toUpperCase()+n.slice(1)+' '+v.toFixed(2)).join(' + '):'Off · original model';const archived=Object.values(m.checkpoints||{}).some(sha=>checkpointCatalog.get(sha)?.path?.includes('/archive/'));return label+(active.some(([n])=>retiredNames.has(n))?' · retired atmosphere':archived?' · archived run':'');}
function titleCase(value){return value?value[0].toUpperCase()+value.slice(1):'';}
function stableJSON(value){return JSON.stringify(value,(_,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);}
function imageState(m){
 if(m.reference_side)return {kind:m.reference_side==='neutral'?'neutral':'target',label:m.reference_side==='neutral'?'Neutral':'Target'};
 const on=Object.values(m.strengths||{}).some(v=>v>0);
 return {kind:on?'on':'off',label:on?'On · Energy '+Number(m.energy).toFixed(2):'Off'};
}
function imageCard(item,blind){
 const m=item.metadata,state=imageState(m),card=el('article',undefined,'image-card'),preview=el('div',undefined,'image-preview'),img=el('img');
 card.dataset.imageId=item.id;img.src='/api/images/'+item.id;img.alt=blind?'Blind review image':imageLabel(m);img.loading='lazy';img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','Open '+img.alt);img.onclick=()=>zoom(item);img.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();zoom(item);}};
 preview.append(img);
 const caption=el('div',undefined,'caption');
 if(!blind){const status=el('div',undefined,'image-card-status');status.append(el('span',state.kind==='on'?'On':state.label,'image-state state-'+state.kind));if(state.kind==='on')status.append(el('span','Energy '+Number(m.energy).toFixed(2),'image-energy'));caption.append(status);}
 const description=blind?imageLabel(m):m.reference_side?(m.reference_side==='neutral'?'Original lighting':'Lighting target'):imageLabel(m).replace(/^Energy [^·]+ · |^Off · /,'');
 caption.append(el('strong',description));
 if(blind)caption.append(el('span','Seed '+m.seed+' · '+m.width+' × '+m.height));
 card.append(preview,caption);return card;
}
function comparisonKey(item){
 const m=item.metadata;
 return stableJSON([m.group_id||item.id,m.seed,m.width,m.height,m.steps,m.cfg,m.model_identity,m.case,m.family,m.character,
  m.reference_side?['reference',m.definition,m.candidate]:['render',m.prompt,m.checkpoints]]);
}
function imageGroups(items){
 const groups=new Map();
 for(const item of items){const key=comparisonKey(item);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item);}
 return [...groups.values()].sort((a,b)=>Math.max(...b.map(i=>i.created))-Math.max(...a.map(i=>i.created))).map(group=>group.sort((a,b)=>{
  const order=m=>m.reference_side?(m.reference_side==='neutral'?0:1):Number(m.energy);
  return order(a.metadata)-order(b.metadata)||a.created-b.created;
 }));
}
function groupHeading(group){
 const m=group[0].metadata,variations=Object.entries(m.mix||{}).filter(([,v])=>v>0).map(([n])=>titleCase(n));
 if(!variations.length)for(const item of group)for(const [n,v] of Object.entries(item.metadata.strengths||{}))if(v>0&&!variations.includes(titleCase(n)))variations.push(titleCase(n));
 const name=m.candidate_label||variations.join(' + ')||titleCase(m.definition?.split('-')[0]);
 const energies=new Set(group.map(i=>i.metadata.energy));
 const comparison=m.reference_side?'Lighting preview':energies.size>2?'Energy sweep':group.some(i=>imageState(i.metadata).kind==='off')&&group.some(i=>imageState(i.metadata).kind==='on')?'Off / On':'Render';
 let origin=m.purpose==='development'?'Training sample':m.purpose==='audit'?'Evaluation':comparison;
 if(m.purpose==='development'&&comparison!=='Render')origin+=' · '+comparison;
 return [name,origin,m.sampling_step!==undefined?'Update '+m.sampling_step:null].filter(Boolean).join(' · ');
}
function drawImageListing(root,items){
 const blind=$('#blind').checked;root.replaceChildren();root.classList.toggle('blind-listing',blind);
 if(blind){
  for(const item of items)if(!blindOrder.has(item.id))blindOrder.set(item.id,crypto.getRandomValues(new Uint32Array(1))[0]);
  for(const item of [...items].sort((a,b)=>blindOrder.get(a.id)-blindOrder.get(b.id)))root.append(imageCard(item,true));
  return;
 }
 for(const group of imageGroups(items)){
  const m=group[0].metadata,section=el('section',undefined,'image-group'),header=el('div',undefined,'image-group-header'),heading=groupHeading(group),cards=el('div',undefined,'comparison-images');
  section.setAttribute('aria-label',heading+' · Seed '+m.seed);section.dataset.groupId=m.group_id||group[0].id;
  header.append(el('h3',heading),el('p',[m.definition,'Seed '+m.seed,m.width+' × '+m.height,m.steps+' steps'].filter(Boolean).join(' · ')));
  const prompt=el('p',m.prompt,'group-prompt');prompt.title=m.prompt;header.append(prompt);section.append(header);
  cards.classList.toggle('single-image',group.length===1);for(const item of group)cards.append(imageCard(item,false));section.append(cards);root.append(section);
 }
}
function drawImages(){drawImageListing($('#images'),images);$('#empty').hidden=images.length>0;$('#more').hidden=images.length<historyLimit;}
$('#blind').onchange=()=>{drawImages();drawTrainingImages();};$('#more').onclick=()=>{historyLimit+=48;refresh();};
let comparison=null,selectedSide='A';
const preservation=['character','outfit','pose','composition','medium','quality'],leakage=['unwanted_objects','recurring_features'];
for(const name of [...preservation,...leakage]){const label=el('label',undefined,'toggle'),box=el('input');box.type='checkbox';box.id='review-'+name;label.append(box,document.createTextNode(name.replaceAll('_',' ')+(preservation.includes(name)?' preserved':'')));$('#review-checks').append(label);}
function setZoomComparison(enabled){
 $('#zoom').classList.toggle('comparing',enabled);
 $('#zoom-compare').setAttribute('aria-pressed',String(enabled));
 $('#zoom-compare').textContent=enabled?'Single image':'Compare';
 $('#reference-figure').hidden=!enabled||!comparison;
}
$('#zoom-compare').onclick=()=>setZoomComparison(!$('#zoom').classList.contains('comparing'));
async function zoom(item){
 selected=item;comparison=null;setZoomComparison(false);$('#zoom-compare').hidden=true;
 const blind=$('#blind').checked;selectedSide=blind&&crypto.getRandomValues(new Uint8Array(1))[0]%2?'B':'A';
 $('#selected-figure').style.order=selectedSide==='A'?0:1;$('#reference-figure').style.order=selectedSide==='A'?1:0;
 $('#selected-label').textContent=blind?'Image '+selectedSide:imageState(item.metadata).label;
 $('#reference-label').textContent=blind?'Image '+(selectedSide==='A'?'B':'A'):'Matched reference';
 $('#atmosphere-strength').value='';for(const k of preservation)$('#review-'+k).checked=true;for(const k of leakage)$('#review-'+k).checked=false;
 $('#atmosphere').replaceChildren(...[['unrated','Unrated'],['win',blind?'Selected image ('+selectedSide+') has more atmosphere':'Win'],['tie','Tie'],['loss',blind?'Other image has more atmosphere':'Loss']].map(([v,t])=>new Option(t,v)));
 $('#zoom-image').src='/api/images/'+item.id;$('#zoom-meta').textContent=imageLabel(item.metadata)+' · Seed '+item.metadata.seed;
 $('#zoom-prompt').textContent=blind?'':item.metadata.prompt;$('#download').href='/api/images/'+item.id+'?download=true';
 $('#review-notes').value='';$('#atmosphere').value='unrated';$('#regression').checked=false;$('#review-status').textContent='';
 $('#zoom').showModal();$('#zoom').scrollTop=0;
 try{
  const matched=await api('images/'+item.id+'/comparison');
  if(selected.id!==item.id||!$('#zoom').open)return;
  comparison=matched;$('#zoom-compare').hidden=!comparison;
  if(comparison){if(!blind)$('#reference-label').textContent=imageState(comparison.metadata).label;$('#reference-image').src='/api/images/'+comparison.id;}
 }catch(e){if(selected.id===item.id&&$('#zoom').open)$('#review-status').textContent=e.message;}
}
let backdropPressed=false;
function outsideZoom(e){const r=$('#zoom').getBoundingClientRect();return e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom;}
$('#zoom').addEventListener('pointerdown',e=>{backdropPressed=e.target===$('#zoom')&&outsideZoom(e);});
$('#zoom').addEventListener('pointercancel',()=>{backdropPressed=false;});
$('#zoom').addEventListener('click',e=>{if(backdropPressed&&e.target===$('#zoom')&&outsideZoom(e))$('#zoom').close();backdropPressed=false;});
$('#zoom').addEventListener('close',()=>{backdropPressed=false;});
$('#close-zoom').onclick=()=>$('#zoom').close();$('#rerender').onclick=async()=>{try{await api('images/'+selected.id+'/rerender',{});$('#zoom').close();notice('Rerender queued with the original settings and checkpoint hashes.');refresh();}catch(e){$('#review-status').textContent=e.message;}};
$('#review').onsubmit=async e=>{e.preventDefault();try{await api('images/'+selected.id+'/review',{notes:$('#review-notes').value,atmosphere:$('#atmosphere').value,major_regression:$('#regression').checked,blind:$('#blind').checked,atmosphere_strength:$('#atmosphere-strength').value===''?null:Number($('#atmosphere-strength').value),...Object.fromEntries([...preservation,...leakage].map(k=>[k,$('#review-'+k).checked]))});$('#review-status').textContent='Review saved';}catch(err){$('#review-status').textContent=err.message;}};
async function action(path){try{await api(path,{});refresh();}catch(e){notice(e.message);}}
function drawJobs(jobs){$('#queue-count').textContent=jobs.filter(j=>['queued','running','cancelling'].includes(j.status)).length;const root=$('#jobs');root.replaceChildren();const queued=jobs.filter(j=>j.status==='queued');for(const job of jobs.slice(0,80)){const row=el('div',undefined,'job'),body=el('div'),actions=el('div',undefined,'actions');body.append(el('strong',job.status+' · seed '+job.payload.seed),el('p',job.payload.prompt.slice(0,160)));if(job.error)body.append(el('p',job.error));if(job.status==='running'){const p=el('progress');p.max=1;p.value=job.progress;body.append(p);}if(['queued','running'].includes(job.status)){const b=el('button','Cancel');b.onclick=()=>action('jobs/'+job.id+'/cancel');actions.append(b);}if(['failed','cancelled'].includes(job.status)){const b=el('button','Retry');b.onclick=()=>action('jobs/'+job.id+'/retry');actions.append(b);}if(job.status==='queued'){const b=el('button','Move first');b.onclick=async()=>{try{await api('queue/reorder',{ids:[job.id,...queued.filter(j=>j.id!==job.id).map(j=>j.id)]});refresh();}catch(e){notice(e.message);}};actions.append(b);}row.append(body,actions);root.append(row);}}
$('#pause').onclick=async()=>{try{await api('training/pause',{paused:!controls.paused});refresh();}catch(e){notice(e.message);}};
let runs=[],trainingImages=[],trainingProgress=null,trainingRefresh=null;
const probeViews=new Map();
function drawProbeDetails(card,run){
 const metric=run.metrics.at(-1);if(!metric?.breakdown)return;
 const view=probeViews.get(run.id)||{open:false,facet:'timestep'};probeViews.set(run.id,view);
 const details=el('details'),summary=el('summary','Latest probe details · update '+metric.step);
 details.open=view.open;details.addEventListener('toggle',()=>{if(details.isConnected)view.open=details.open;});
 details.append(summary,el('p','Fixed development fixtures at Energy 1. Cosine measures edit alignment; gain 1 matches the teacher magnitude.'));
 const table=el('table');table.setAttribute('aria-label','Latest development metrics');
 const metrics=[['Residual p95',metric.residual_p95],['Edit cosine',metric.edit_cosine],['Edit gain',metric.edit_gain],['Orthogonal error RMS',metric.orthogonal_rms],['Teacher SWD · 256 projections',metric.teacher_swd_256],...Object.entries(metric.game_swd||{}).map(([noise,value])=>['Game SWD · noise '+noise,value])];
 for(const [label,value] of metrics){if(!Number.isFinite(value))continue;const row=el('tr');row.append(el('th',label),el('td',value.toFixed(4)));table.append(row);}
 details.append(table);
 const label=el('label','Residual breakdown'),select=el('select');select.setAttribute('aria-label','Residual breakdown for '+run.id);
 for(const [value,text] of [['timestep','Denoising position'],['definition','Lighting definition'],['character','Character'],['framing','Framing'],['bare','Prompt wording']])select.append(new Option(text,value));
 select.value=view.facet;label.append(select);details.append(label);
 const breakdown=el('table');breakdown.setAttribute('aria-label','Residual RMS by '+view.facet);details.append(breakdown);
 function draw(){
  view.facet=select.value;breakdown.setAttribute('aria-label','Residual RMS by '+view.facet);breakdown.replaceChildren();
  const head=el('tr');head.append(el('th',select.selectedOptions[0].text),el('th','Raw RMS'));breakdown.append(head);
  const entries=Object.entries(metric.breakdown).filter(([key])=>key.startsWith(view.facet+':'));
  entries.sort(([a],[b])=>a.localeCompare(b,undefined,{numeric:true}));
  for(const [key,value] of entries){let name=key.slice(key.indexOf(':')+1);if(view.facet==='timestep')name='Position '+(Number(name)+1)+' of 10';if(view.facet==='bare')name=name==='True'?'Bare prompt':'Captioned prompt';const row=el('tr');row.append(el('td',name),el('td',value.toFixed(4)));breakdown.append(row);}
 }
 select.addEventListener('change',draw);draw();card.append(details);
}
function drawRuns(){const root=$('#runs'),retiredRoot=$('#retired-run-list');root.replaceChildren();retiredRoot.replaceChildren();$('#retired-runs').hidden=!runs.some(r=>retiredNames.has(r.config.variation));const priority=run=>run.config.directory?.includes('/archive/')?5:run.status==='running'?0:run.status==='queued'?(run.config.until===200?1:2):run.config.until===1600?3:4;const ordered=[...runs].sort((a,b)=>priority(a)-priority(b)||a.id.localeCompare(b.id));if(!runs.length)root.append(el('p','Slider training has not started. After target preparation and benchmarking, each variation starts with a 200-update pilot. The full campaign is 1,600 updates per variation, subject to pilot review.'));for(const run of ordered){const card=el('div',undefined,'run'),target=run.config.until||200,variation=run.config.variation,title=variation[0].toUpperCase()+variation.slice(1),retired=retiredNames.has(variation),container=retired?retiredRoot:root;card.append(el('h2',title+(target===200?' · pilot':' · full campaign')+(retired?' · retired':run.config.directory?.includes('/archive/')?' · archived':'')),el('p',(target===200&&run.pilot_qualification?(run.pilot_qualification.passed?'pilot qualified':'pilot review failed'):run.status.replaceAll('_',' '))+' · update '+run.step+' / '+target));card.append(el('small',run.id+' · seed '+run.config.seed));const progress=el('progress');progress.max=target;progress.value=run.step;progress.setAttribute('aria-label',title+' training updates');card.append(progress);if(run.error)card.append(el('p',run.error));
if(target===200&&run.pilot_qualification?.passed===false){const labels={finite_gradients:'finite gradients',improved_held_out_residual:'held-out residual improvement',exact_resume_and_studio_parity:'resume and renderer agreement',usable_character_renders:'character preservation in sample images'};card.append(el('p','Failed pilot checks: '+Object.entries(run.pilot_qualification.checks).filter(([k,v])=>!v).map(([k])=>labels[k]||k).join(', ')+'. Full training is gated.'));}
if(run.log_error)card.append(el('p',run.log_error));
if(run.timing){const t=run.timing;card.append(el('p',t.seconds_per_update.toFixed(2)+' sec/update · approximately '+Math.ceil(t.training_seconds_remaining/60)+' min of updates remaining. Previews and evaluations add time.'));}
if(run.recent_updates?.length){const latest=run.recent_updates.at(-1),chart=el('canvas');chart.setAttribute('aria-label','Recent discriminator and generator losses');card.append(el('p','Recent game losses: discriminator in coral, generator in mint. These fluctuate; sample quality and held-out residual determine progress.'),chart,el('p','D '+latest.d_adv.toFixed(4)+' · G '+latest.g_adv.toFixed(4)+' · particle VIC '+latest.vic.toFixed(4)+' · gradient cap '+latest.d_penalty.toFixed(4)));requestAnimationFrame(()=>plotGame(chart,run.recent_updates));}
if(run.metrics.length){card.append(el('p','Full-strength residual: raw in coral, best-so-far in mint. Lower is better. Quality and character preservation are checked in the samples below.'));const canvas=el('canvas');canvas.setAttribute('aria-label','Raw full-strength residual and best-so-far residual');card.append(canvas);const table=el('table'),head=el('tr');for(const t of ['Update','Raw R','Best-so-far R'])head.append(el('th',t));table.append(head);for(const m of run.metrics){const row=el('tr');for(const v of [m.step,m.full_strength_raw_R?.toFixed(4),m.best_so_far_R?.toFixed(4)])row.append(el('td',String(v)));table.append(row);}card.append(table);drawProbeDetails(card,run);container.append(card);requestAnimationFrame(()=>plot(canvas,run.metrics));}else container.append(card);}}
function drawPreparation(){if(!trainingProgress)return;const data=trainingProgress,root=$('#preparation');root.replaceChildren();const card=el('div',undefined,'run'),percent=Math.floor(100*data.done/Math.max(1,data.total));card.append(el('h2',data.complete?'Teaching examples ready':'Preparing teaching examples'),el('p',data.done.toLocaleString()+' / '+data.total.toLocaleString()+' paired trajectories · '+percent+'%'));const progress=el('progress');progress.max=data.total;progress.value=data.done;progress.setAttribute('aria-label','Target preparation');card.append(progress);const table=el('table'),head=el('tr');for(const name of ['Variation','Training pairs','Dev pairs'])head.append(el('th',name));table.append(head);for(const row of data.targets){const tr=el('tr');tr.append(el('td',row.variation[0].toUpperCase()+row.variation.slice(1)+(retiredNames.has(row.variation)?' · retired':'')));for(const split of ['train','dev'])tr.append(el('td',row[split].done+' / '+row[split].total+(row[split].complete?' ✓':'')));table.append(tr);}card.append(table);const owner=controls.owner;card.append(el('p',controls.paused?'Training paused for rendering.':owner?(typeof owner==='string'?owner:owner.state):'Worker offline · saved progress is preserved.'));if(data.updated_at)card.append(el('small','Preparation last saved '+new Date(data.updated_at*1000).toLocaleTimeString()));root.append(card);drawBenchmark(root,data);}
function drawBenchmark(root,data){const b=data.benchmark;if(b){const card=el('div',undefined,'run'),index=Math.min((b.candidates?.length||0)+1,b.candidate_count||6);card.append(el('h2','Training speed benchmark'),el('p',b.status==='running'?'Configuration '+index+' / '+b.candidate_count+' · '+b.phase:b.status+(b.accepted?' · selected '+b.accepted:'')));if(b.status==='running'&&b.phase_total){const progress=el('progress');progress.max=b.phase_total;progress.value=b.phase_done;progress.setAttribute('aria-label','Benchmark '+b.phase);card.append(el('p',b.phase_done+' / '+b.phase_total+' updates'),progress);}if(b.error)card.append(el('p',b.error));if(b.candidates?.length){const table=el('table'),head=el('tr');for(const value of ['Config','Status','Seconds/update'])head.append(el('th',value));table.append(head);b.candidates.forEach((candidate,i)=>{const row=el('tr');for(const value of [String(i+1),candidate.status,candidate.mean_seconds?.toFixed(2)||'—'])row.append(el('td',value));table.append(row);});card.append(table);}if(b.status==='completed')card.append(el('p','Benchmark complete. Training and review results appear below.'));root.append(card);}if(data.hardware_benchmark?.status==='completed'&&data.hardware_benchmark.parity_passed){const h=data.hardware_benchmark,card=el('div',undefined,'run');card.append(el('h2','Current GPU benchmark'),el('p',h.mean_seconds.toFixed(2)+' sec/update · '+h.peak_vram_gib.toFixed(2)+' GiB peak GPU memory'),el('p','The replacement GPU reproduced the same training state and passed rendering checks. Timing uses 20 warmup updates and 50 measured updates.'));root.append(card);}if(data.campaign?.stage?.startsWith('verifying')){const card=el('div',undefined,'run');card.append(el('h2','GPU verification'),el('p',data.campaign.stage));root.append(card);}if(data.campaign?.error){const card=el('div',undefined,'run');card.append(el('h2','Campaign '+data.campaign.stage),el('p',data.campaign.error));root.append(card);}}
function drawTrainingImages(){drawImageListing($('#training-images'),trainingImages);}
function drawCheckpointReviews(reports){
 const root=$('#checkpoint-reviews');root.replaceChildren(el('h2','Checkpoint quality reviews'),el('p','Each selection checkpoint needs at least 75% atmosphere wins and 90% without major preservation or quality regression. With eight cases, all eight must avoid major regression.'));
 for(const name of names){
  const report=reports[name];if(!report)continue;
  const card=el('div',undefined,'run');card.append(el('h2',name[0].toUpperCase()+name.slice(1)));
  const rows=[...report.candidates.map(c=>({...c,complete:true})),...report.incomplete].sort((a,b)=>a.step-b.step);
  if(!rows.length)card.append(el('p','Selection reviews begin at update 400.'));
  for(const row of rows){
   if(!row.complete){card.append(el('p','Update '+row.step+' · '+row.rated+' / '+row.required+' cases reviewed · waiting for images or reviews.'));continue;}
   const qualifies=row.atmosphere_wins/row.cases>=.75&&row.major_regressions/row.cases<=.1;
   card.append(el('p','Update '+row.step+' · '+row.atmosphere_wins+' / '+row.cases+' atmosphere wins · '+row.major_regressions+' / '+row.cases+' major regressions · '+(qualifies?'meets review thresholds.':'does not qualify.')));
  }
  card.append(el('p',report.selected?'Current development choice: update '+report.selected.step+'. Final selection waits for all four checkpoints to be reviewed.':'No qualifying checkpoint selected.'));
  root.append(card);
 }
}
function objectiveNumber(value){return value!==0&&Math.abs(value)<.001?value.toExponential(2):value.toFixed(4);}
function drawConvergence(reports){
 const root=$('#convergence');root.replaceChildren(el('h2','Has the slider settled?'),el('p','Fixed-input measurements compare the saved checkpoints. Smaller prediction changes mean less movement. Frozen-opponent checks look for remaining objective improvement. These checks do not certify image quality or global convergence.'));
 for(const name of names){
  const report=reports[name];if(!report)continue;
  const card=el('div',undefined,'run');card.append(el('h3',name[0].toUpperCase()+name.slice(1)));
  if(report.status==='not_started'){card.append(el('p',name==='theatrical'?'A qualified full run is needed before this late-checkpoint assessment.':'Convergence measurements have not started.'));root.append(card);continue;}
  card.append(el('p',report.status==='completed'?'Assessment complete':report.status==='running'?'Measuring · '+report.stage:report.status));
  if(report.error)card.append(el('p',report.error));
  if(report.conclusion)card.append(el('p',report.conclusion));
  if(report.status==='completed'&&report.responses?.length){
   const finalStep=Math.max(...report.responses.map(r=>r.step));
   for(const [side,label] of [['d','Critic'],['g','Slider']]){
    const trials=report.responses.filter(r=>r.step===finalStep&&r.side===side),improved=trials.filter(r=>r.improvement.ci95[0]>0).length,worsened=trials.filter(r=>r.improvement.ci95[1]<0).length;
    if(trials.length)card.append(el('p',label+' at update '+finalStep+': '+improved+'/'+trials.length+' trials with a positive improvement interval'+(worsened?'; '+worsened+' with a negative interval.':'.')));
   }
  }
  if(report.fixture)card.append(el('small',report.fixture.prompt_rows+' development prompt rows · '+report.fixture.velocity_fields+' fixed velocity fields · all 10 timesteps · Energy 1'));
  const movement=report.movements||[];
  if(movement.length){
   card.append(el('h4','Prediction change'),el('p','Change in denoising predictions as a percentage of the previous checkpoint’s edit. Live weights show unsmoothed movement; EMA is the version used for rendering.'));
   const canvas=el('canvas');canvas.setAttribute('aria-label',name+' prediction change across checkpoints');card.append(canvas);
   const table=el('table'),head=el('tr');for(const value of ['Updates','Weights','Change / prior edit','Change / teacher edit'])head.append(el('th',value));table.append(head);
   for(const m of [...movement].sort((a,b)=>(a.end-a.start)-(b.end-b.start)||a.start-b.start||Number(a.weights==='EMA')-Number(b.weights==='EMA'))){const row=el('tr');for(const v of [m.start+' → '+m.end,m.weights,m.relative_to_previous_edit==null?'Undefined (zero prior edit)':(100*m.relative_to_previous_edit).toFixed(2)+'%',m.relative_to_teacher_edit==null?'—':(100*m.relative_to_teacher_edit).toFixed(2)+'%'])row.append(el('td',v));table.append(row);}card.append(table);
   requestAnimationFrame(()=>plotMovement(canvas,movement));
  }
  if(report.responses?.length){
   if(report.response_protocol?.response_updates)card.append(el('p',report.response_protocol.response_updates+' updates per trial · '+report.response_protocol.trial_seeds.length+' sampler seeds · each trial starts from the saved checkpoint.'));
   card.append(el('h4','Remaining improvement against a frozen opponent'),el('p','Positive improvement means the tested side found a lower objective. Intervals resample whole development prompt rows. These are short local searches on disposable copies.'));
   const table=el('table'),head=el('tr');for(const value of ['Checkpoint','Side / trial','Before → after','Improvement · 95% interval'])head.append(el('th',value));table.append(head);
   for(const r of report.responses){const m=r.improvement,row=el('tr');for(const v of [String(r.step),(r.side==='g'?'Slider':'Critic')+' / '+r.seed,objectiveNumber(m.before)+' → '+objectiveNumber(m.after),objectiveNumber(m.improvement)+' ['+m.ci95.map(objectiveNumber).join(', ')+']'])row.append(el('td',v));table.append(row);}card.append(table);
  }
  if(report.source_files_unchanged)card.append(el('small','Original checkpoint files verified unchanged.'));
  root.append(card);
 }
 const link=el('a','Open full numerical report');link.href='/api/training/convergence';link.target='_blank';link.rel='noopener';root.append(link);
}
function plotMovement(canvas,rows){
 const ratio=devicePixelRatio||1,w=canvas.clientWidth||700,h=200,c=canvas.getContext('2d');canvas.width=w*ratio;canvas.height=h*ratio;c.scale(ratio,ratio);
 const data=rows.filter(r=>r.end-r.start===100&&r.relative_to_previous_edit!=null);if(!data.length)return;
 const max=Math.max(.001,...data.map(r=>100*r.relative_to_previous_edit))*1.15,first=Math.min(...data.map(r=>r.end)),last=Math.max(...data.map(r=>r.end));
 for(const [weights,color] of [['live','#f49991'],['EMA','#9bd5bc']]){c.beginPath();c.strokeStyle=color;c.lineWidth=2;data.filter(r=>r.weights===weights).sort((a,b)=>a.end-b.end).forEach((r,i)=>{const x=40+(w-56)*(r.end-first)/Math.max(1,last-first),y=h-24-(h-44)*(100*r.relative_to_previous_edit)/max;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();}
 c.fillStyle='#a0a9ae';c.font='11px system-ui';c.fillText(max.toFixed(1)+'%',2,14);c.fillText('0%',2,h-24);c.fillText('Update '+first,40,h-5);c.textAlign='right';c.fillText('Update '+last,w-16,h-5);c.textAlign='left';c.fillText('Live: coral · EMA: mint',45,14);
}
function refreshTraining(){if(trainingRefresh)return trainingRefresh;trainingRefresh=Promise.all([api('training/progress'),api('training/images'),api('audits/selection'),api('training/convergence')]).then(([progress,samples,selections,convergence])=>{trainingProgress=progress;trainingImages=samples;drawConvergence(convergence);drawPreparation();drawTrainingImages();drawCheckpointReviews(selections);}).catch(e=>notice(e.message)).finally(()=>{trainingRefresh=null;});return trainingRefresh;}
function plot(canvas,data){const ratio=devicePixelRatio||1,w=canvas.clientWidth||700,h=200;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);const max=Math.max(...data.map(d=>d.full_strength_raw_R),.001)*1.12;for(const [field,color] of [['full_strength_raw_R','#f49991'],['best_so_far_R','#9bd5bc']]){c.beginPath();c.strokeStyle=color;c.lineWidth=2;data.forEach((d,i)=>{const x=16+(w-32)*d.step/Math.max(400,data.at(-1).step),y=h-16-(h-32)*d[field]/max;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();}}
function plotGame(canvas,data){
 const ratio=devicePixelRatio||1,w=canvas.clientWidth||700,h=200,c=canvas.getContext('2d');canvas.width=w*ratio;canvas.height=h*ratio;c.scale(ratio,ratio);
 const max=Math.max(.001,...data.flatMap(d=>[d.d_adv,d.g_adv]))*1.12,first=data[0].step,last=data.at(-1).step;
 for(const [field,color] of [['d_adv','#f49991'],['g_adv','#9bd5bc']]){c.beginPath();c.strokeStyle=color;c.lineWidth=2;data.forEach((d,i)=>{const x=16+(w-32)*(d.step-first)/Math.max(1,last-first),y=h-26-(h-42)*d[field]/max;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();}
 c.fillStyle='#a0a9ae';c.font='11px system-ui';c.fillText('Update '+first,16,h-6);c.textAlign='right';c.fillText('Update '+last,w-16,h-6);
}
async function loadCatalog(){
 const c=await api('catalog');retiredNames=new Set(Object.keys(c.retired_variations||{}));checkpointCatalog.clear();latestCheckpoints.clear();
 for(const checkpoint of c.checkpoints)checkpointCatalog.set(checkpoint.sha256,checkpoint);
 const offered=c.mixer_variations||['candlelit','moonlit'];
 for(const n of offered){
  const current=c.checkpoints.filter(row=>row.variation===n&&!row.path?.includes('/archive/')&&row.metadata?.weights!=='live'&&stableJSON(row.metadata?.model_identity)===stableJSON(c.model));
  current.sort((a,b)=>(b.metadata.step||0)-(a.metadata.step||0)||b.created-a.created||a.sha256.localeCompare(b.sha256));
  if(current.length)latestCheckpoints.set(n,current[0]);
 }
 drawMixers(offered);
 for(const n of names){const available=latestCheckpoints.has(n),slider=$('#'+n);slider.disabled=!available;slider.closest('.mix-row').querySelector('.mix-max').disabled=!available;$('#'+n+'-availability').textContent=available?'':'Not available yet';if(!available)slider.value=0;}
 updateMix();
 $('#draft-atmospheres').replaceChildren(...Object.values(c.draft_atmospheres||{}).map(d=>{const note=el('p',undefined,'mix-note');note.append(el('strong',d.label+' · In development'),el('br'),document.createTextNode(d.description));return note;}));
 drawImages();drawTrainingImages();modelCatalog.clear();for(const entry of c.models)modelCatalog.set(entry.id,entry);const model=$('#model'),selectedModel=model.value;model.replaceChildren(...c.models.map(m=>new Option(m.label,m.id)));model.value=modelCatalog.has(selectedModel)?selectedModel:c.default_model;updateResolutions();
}
async function loadDefinitions(){const data=await api('definitions'),root=$('#definition-list'),retiredRoot=$('#retired-definition-list');root.replaceChildren();retiredRoot.replaceChildren();$('#retired-definitions').hidden=!data.definitions.some(d=>retiredNames.has(d.variation));for(const d of data.definitions){const card=el('article',undefined,'definition');card.append(el('small',d.variation.toUpperCase()+' · '+d.split),el('h2',d.id),el('p',d.lighting));const details=el('details'),summary=el('summary','Show paired example');details.append(summary);const row=[...data.train.rows,...data.dev.rows].find(r=>r.definition===d.id);if(row){details.append(el('p','Neutral: '+row.neutral),el('p','Positive: '+row.positive));}card.append(details);const thumbs=el('div',undefined,'definition-images');thumbs.dataset.definition=d.id;card.append(thumbs);(retiredNames.has(d.variation)?retiredRoot:root).append(card);}drawDefinitionImages();}
async function getHistory(){const pages=[];for(let offset=0;offset<historyLimit;offset+=500)pages.push(api("history?limit="+Math.min(500,historyLimit-offset)+"&offset="+offset));return (await Promise.all(pages)).flat();}
async function refresh(){try{const [state,jobs,history,training,references]=await Promise.all([api('status'),api('jobs'),getHistory(),api('training/runs'),api('definitions/images')]);controls=state.controls;const owner=controls.owner;$('#status').textContent=owner?(typeof owner==='string'?owner:owner.state+' · GPU '+owner.gpu):'Worker offline · queue persists';$('#status-dot').style.background=owner?'var(--mint)':'var(--warm)';$('#pause').textContent=controls.paused?'Resume training':'Pause training';images=history;referenceImages=references;runs=training;drawJobs(jobs);drawImages();drawDefinitionImages();if(!$('#training').hidden){drawRuns();await refreshTraining();}}catch(e){$('#status').textContent='Connection interrupted';notice(e.message);}}
function drawDefinitionImages(){for(const holder of document.querySelectorAll('.definition-images')){holder.replaceChildren();for(const item of referenceImages.filter(i=>i.metadata.definition===holder.dataset.definition&&i.metadata.reference_side).slice(0,4)){const figure=el('figure'),img=el('img');img.src='/api/images/'+item.id;img.alt=item.metadata.reference_side+' character reference';img.loading='lazy';img.onclick=()=>zoom(item);figure.append(img,el('figcaption',item.metadata.reference_side));holder.append(figure);}}}
const events=new EventSource('/api/events');events.addEventListener('update',e=>{const event=JSON.parse(e.data);if(event.kind==='catalog')loadCatalog();if(!refreshTimer)refreshTimer=setTimeout(()=>{refreshTimer=null;refresh();},400);});events.onerror=()=>{$('#status').textContent='Reconnecting…';};
showTab();setInterval(()=>{if(!document.hidden&&!$('#training').hidden)refresh();},5000);
loadCatalog().then(()=>Promise.all([refresh(),loadDefinitions()])).catch(e=>notice(e.message));
