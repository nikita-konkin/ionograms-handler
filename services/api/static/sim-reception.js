/* Same quarter-dB live columns and colour scale as /ui/live. */
(() => {
  const decode = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
  function picture(head) {
    return {head, columns: new Map()};
  }
  function column(pic, event) {
    if (!pic || event.t0 !== pic.head.t0 || event.index < 0 || event.index >= pic.head.columns) return;
    const bytes = decode(event.db);
    if (bytes.length === pic.head.rows) pic.columns.set(event.index, bytes);
  }
  // Export the stream reducer for tests without a browser or a server.
  if (typeof module !== 'undefined') module.exports = {picture, column};
  if (typeof document === 'undefined') return;
  const panel = document.getElementById('simReception');
  if (!panel) return;
  const tx = panel.dataset.tx, rx = panel.dataset.rx;
  const choose = document.getElementById('simStation');
  const leftInfo = document.getElementById('simPredictionInfo');
  const rightInfo = document.getElementById('simReceivingInfo');
  const connection = document.getElementById('simConnection');
  let current = null, prediction = null, station = '', controller = null, version = 0;
  let drawing = false, polling = false;
  const statusText = {
    choose_station: T('sim_pair.js.choose'), waiting: T('sim_pair.js.waiting'),
    no_forecast: T('sim_pair.js.no_forecast'), building: T('sim_pair.js.building'),
    busy: T('sim_pair.js.building'), failed: T('sim_pair.js.failed'),
  };
  const headers = () => token() ? {Authorization: 'Bearer ' + token()} : {};
  const colours = [[0,0,255], [0,255,255], [173,255,47], [255,255,0], [255,159,0], [255,0,0]];
  function draw(id, pic) {
    const cv = document.getElementById(id), dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.max(320, cv.clientWidth) * dpr; cv.height = 360 * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr);
    const w = cv.width / dpr, h = 360, m = {l:58,r:12,t:12,b:45};
    const pw = w-m.l-m.r, ph = h-m.t-m.b;
    g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--plot').trim() || '#14161a';
    g.fillRect(m.l,m.t,pw,ph);
    if (!pic) return;
    const head=pic.head, nx=Math.min(800,head.columns), ny=Math.min(256,head.rows);
    const pixels=new Float32Array(nx*ny).fill(-1);
    for (const [i, values] of pic.columns) {
      const x=Math.floor(i*nx/head.columns);
      for (let r=0;r<head.rows;r++) {
        const y=ny-1-Math.floor(r*ny/head.rows), k=y*nx+x;
        pixels[k]=Math.max(pixels[k],values[r]/4);
      }
    }
    const raster=document.createElement('canvas'); raster.width=nx; raster.height=ny;
    const ctx=raster.getContext('2d'), im=ctx.createImageData(nx,ny);
    pixels.forEach((db,i)=>{
      if(db<6)return;
      const v=Math.max(0,Math.min(5,(db-6)/30*6)), j=Math.min(4,Math.floor(v)), u=v-j;
      for(let k=0;k<3;k++)im.data[i*4+k]=colours[j][k]*(1-u)+colours[j+1][k]*u;
      im.data[i*4+3]=255;
    });
    ctx.putImageData(im,0,0); g.imageSmoothingEnabled=false;
    g.drawImage(raster,m.l,m.t,pw,ph);
    g.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
    g.font='12px sans-serif'; g.textAlign='center';
    for(let i=0;i<=4;i++) {
      const f=head.f_lo_mhz+i/4*head.columns*head.f_step_mhz;
      g.fillText(f.toFixed(1),m.l+pw*i/4,h-25);
      const delay=(head.range_lo_km+i/4*head.rows*head.range_step_km)/head.km_per_ms;
      g.fillText(delay.toFixed(1),m.l-28,m.t+ph*(1-i/4)+4);
    }
    g.fillText(T('sim_pair.js.frequency'),m.l+pw/2,h-5);
    g.save();g.translate(12,m.t+ph/2);g.rotate(-Math.PI/2);
    g.fillText(T('sim_pair.js.delay'),0,0);g.restore();
  }
  function redraw() {
    if(drawing)return; drawing=true;
    requestAnimationFrame(()=>{
      drawing=false; draw('simReceiving',current); draw('simPrediction',prediction);
      const progress=document.getElementById('simProgress');
      progress.max=current ? current.head.columns : 1;
      progress.value=current ? current.columns.size : 0;
      rightInfo.textContent=current ? T('sim_pair.js.progress', {
        at:T.msk(current.head.t0,true), n:current.columns.size, total:current.head.columns,
      }) : T('sim_pair.js.waiting');
    });
  }
  function start(head) {
    if(current && current.head.t0===head.t0)return;
    current=picture(head); prediction=null;
    leftInfo.textContent=T('sim_pair.js.no_forecast');
  }
  function apply(kind, ev) {
    if(kind==='snapshot') {
      const view=ev.current[tx] || ev.last[tx];
      if(!view){current=null;prediction=null;leftInfo.textContent=T('sim_pair.js.waiting');redraw();return;}
      start(view); current=picture(view);
      for(const [index,db] of Object.entries(view.data))column(current,{t0:view.t0,index:Number(index),db});
    } else if(ev.tx===tx) {
      if(kind==='sounding')start(ev);
      if(kind==='column')column(current,ev);
    }
    redraw();
  }
  async function listen(selected, generation, signal) {
    while(!signal.aborted) {
      try {
        connection.textContent=T('sim_pair.js.connecting');
        const r=await fetch('/stations/'+encodeURIComponent(selected)+'/live/events',{headers:headers(),signal});
        if(!r.ok || !r.body)throw new Error('HTTP '+r.status);
        connection.textContent=T('sim_pair.js.connected');
        const reader=r.body.pipeThrough(new TextDecoderStream()).getReader();let buffer='';
        try {
          for(;;) {
            const {value,done}=await reader.read();if(done)break;
            buffer+=value;let at;
            while((at=buffer.indexOf('\n\n'))>=0) {
              const block=buffer.slice(0,at);buffer=buffer.slice(at+2);
              let kind='',data='';
              for(const line of block.split('\n')) {
                if(line.startsWith('event: '))kind=line.slice(7);
                if(line.startsWith('data: '))data+=line.slice(6);
              }
              if(data && generation===version)apply(kind,JSON.parse(data));
            }
          }
        } finally {await reader.cancel().catch(()=>{});}
      } catch(_) { /* reconnect from a fresh snapshot */ }
      if(signal.aborted)return;
      connection.textContent=T('sim_pair.js.disconnected');
      await new Promise(resolve=>setTimeout(resolve,2000));
    }
  }
  function switchStation(value) {
    if(station===value)return;
    station=value;version++;if(controller)controller.abort();
    current=null;prediction=null;leftInfo.textContent=T('sim_pair.js.waiting');redraw();
    if(value){controller=new AbortController();listen(value,version,controller.signal);}
  }
  async function poll() {
    if(polling)return;polling=true;const generation=version;
    try {
      const url='/sim/live/'+encodeURIComponent(tx)+'/'+encodeURIComponent(rx)+'/reception';
      const r=await fetch(url+(station?'?station='+encodeURIComponent(station):''),{headers:headers()});
      if(!r.ok)throw new Error('HTTP '+r.status);
      const data=await r.json();if(generation!==version)return;
      choose.replaceChildren(new Option(T('sim_pair.js.choose'),''),...data.stations.map(s=>new Option(s,s)));
      choose.value=station || data.station || '';
      if(!station && data.station){switchStation(data.station);return;}
      if(current && data.t0===current.head.t0 && data.status==='ready') {
        prediction=picture(data.head);const bytes=decode(data.cells);
        for(let i=0;i<data.head.columns;i++)prediction.columns.set(i,bytes.subarray(i*data.head.rows,(i+1)*data.head.rows));
        leftInfo.textContent=T('sim_pair.js.issued',{at:T.msk(data.issued_at,true),due:T.msk(data.valid_at,true)});
      } else if(!prediction)leftInfo.textContent=statusText[data.status] || T('sim_pair.js.waiting');
      redraw();
    } catch(_) {leftInfo.textContent=T('sim_pair.js.disconnected');}
    finally {polling=false;}
  }
  choose.addEventListener('change',()=>{switchStation(choose.value);poll();});
  window.addEventListener('resize',redraw);window.addEventListener('themechange',redraw);
  window.addEventListener('pagehide',()=>{if(controller)controller.abort();});
  redraw();poll();setInterval(poll,5000);
})();
