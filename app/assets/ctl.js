/* Выпадающий список вместо системного <select>. Исходный select остаётся в DOM
   (скрыт): страница читает его value и ловит change как раньше. Список перестраивается
   при замене option и при программной записи value. */
(function(){
  'use strict';
  const GAP=6, MAXH=320;
  let openDd=null;

  function build(sel){
    if(sel._dd||sel.hasAttribute('data-native')) return;
    const wrap=document.createElement('div'); wrap.className='dd';
    const btn=document.createElement('button'); btn.type='button'; btn.className='dd-btn';
    btn.setAttribute('aria-haspopup','listbox'); btn.setAttribute('aria-expanded','false');
    const lab=sel.getAttribute('aria-label'); if(lab) btn.setAttribute('aria-label',lab);
    btn.innerHTML='<span class="dd-label"></span><span class="dd-chev"></span>';
    const pop=document.createElement('div'); pop.className='dd-pop'; pop.setAttribute('role','listbox');
    const uid='dd'+Math.random().toString(36).slice(2,8); pop.id=uid; btn.setAttribute('aria-controls',uid);
    sel.parentNode.insertBefore(wrap,sel); wrap.appendChild(btn); wrap.appendChild(sel); sel.classList.add('dd-native');
    sel.tabIndex=-1; sel.setAttribute('aria-hidden','true');
    document.body.appendChild(pop);
    const st={sel,wrap,btn,pop,act:-1,buf:'',bufT:0,opts:[]};
    sel._dd=st;

    function labelText(){const o=sel.options[sel.selectedIndex];return o?o.text:'';}
    function sync(){
      btn.querySelector('.dd-label').textContent=labelText();
      btn.disabled=sel.disabled;
      pop.querySelectorAll('.dd-opt').forEach((el,i)=>el.setAttribute('aria-selected',String(i===sel.selectedIndex)));
    }
    function rebuild(){
      pop.innerHTML=''; st.opts=[];
      [...sel.options].forEach((o,i)=>{
        const el=document.createElement('div'); el.className='dd-opt'; el.setAttribute('role','option'); el.id=uid+'-'+i;
        el.innerHTML='<span class="t"></span>'; el.firstChild.textContent=o.text;
        if(o.disabled) el.setAttribute('aria-disabled','true');
        el.dataset.i=i; pop.appendChild(el); st.opts.push(el);
      });
      sync();
    }
    function setAct(i,scroll){
      if(st.act>=0&&st.opts[st.act]) st.opts[st.act].classList.remove('act');
      st.act=i;
      const el=st.opts[i];
      if(el){el.classList.add('act');btn.setAttribute('aria-activedescendant',el.id);if(scroll!==false)el.scrollIntoView({block:'nearest'});}
      else btn.removeAttribute('aria-activedescendant');
    }
    function step(from,dir){
      let i=from;
      for(let n=0;n<st.opts.length;n++){i+=dir;if(i<0||i>=st.opts.length)return from;if(!sel.options[i].disabled)return i;}
      return from;
    }
    function place(){
      const r=btn.getBoundingClientRect(), vh=innerHeight, vw=innerWidth;
      pop.style.minWidth=r.width+'px'; pop.style.maxWidth=Math.min(vw-16,Math.max(r.width,420))+'px';
      pop.style.maxHeight=MAXH+'px';
      const h=Math.min(pop.scrollHeight+2,MAXH);
      const below=vh-r.bottom-GAP-8, above=r.top-GAP-8;
      const up=h>below&&above>below;
      pop.classList.toggle('up',up);
      const room=Math.max(120,up?above:below); pop.style.maxHeight=Math.min(MAXH,room)+'px';
      const hh=Math.min(h,room);
      pop.style.top=(up?r.top-GAP-hh:r.bottom+GAP)+'px';
      const w=Math.max(r.width,pop.offsetWidth);
      let left=r.left; if(left+w>vw-8) left=Math.max(8,vw-8-w);
      pop.style.left=left+'px';
      pop.style.setProperty('--ox',(r.left+r.width/2-left)+'px');
      pop.style.setProperty('--oy',up?'100%':'0');
    }
    function open(){
      if(sel.disabled||wrap.classList.contains('open')) return;
      if(openDd&&openDd!==st) openDd.close();
      rebuild(); place();
      wrap.classList.add('open'); pop.classList.add('open'); btn.setAttribute('aria-expanded','true');
      setAct(sel.selectedIndex<0?0:sel.selectedIndex);
      openDd=st;
    }
    function close(refocus){
      if(!wrap.classList.contains('open')) return;
      wrap.classList.remove('open'); pop.classList.remove('open'); btn.setAttribute('aria-expanded','false');
      btn.removeAttribute('aria-activedescendant'); if(openDd===st) openDd=null;
      if(refocus) btn.focus({preventScroll:true});
    }
    function pick(i){
      const o=sel.options[i]; if(!o||o.disabled) return;
      const changed=sel.selectedIndex!==i;
      sel.selectedIndex=i; sync(); close(true);
      if(changed){sel.dispatchEvent(new Event('input',{bubbles:true}));sel.dispatchEvent(new Event('change',{bubbles:true}));}
    }
    st.open=open; st.close=close; st.place=place;

    btn.addEventListener('click',()=>wrap.classList.contains('open')?close(true):open());
    btn.addEventListener('keydown',e=>{
      const isOpen=wrap.classList.contains('open'), k=e.key;
      if(!isOpen){
        if(k==='ArrowDown'||k==='ArrowUp'||k==='Enter'||k===' '){e.preventDefault();open();}
        return;
      }
      if(k==='Escape'){e.preventDefault();e.stopPropagation();close(true);}
      else if(k==='Tab'){close(false);}
      else if(k==='ArrowDown'){e.preventDefault();setAct(step(st.act,1));}
      else if(k==='ArrowUp'){e.preventDefault();setAct(step(st.act,-1));}
      else if(k==='Home'){e.preventDefault();setAct(step(-1,1));}
      else if(k==='End'){e.preventDefault();setAct(step(st.opts.length,-1));}
      else if(k==='PageDown'){e.preventDefault();setAct(Math.min(st.opts.length-1,st.act+7));}
      else if(k==='PageUp'){e.preventDefault();setAct(Math.max(0,st.act-7));}
      else if(k==='Enter'||k===' '){e.preventDefault();pick(st.act);}
      else if(k.length===1&&!e.ctrlKey&&!e.metaKey&&!e.altKey){
        clearTimeout(st.bufT); st.buf+=k.toLowerCase(); st.bufT=setTimeout(()=>{st.buf='';},700);
        const n=st.opts.length;
        for(let d=1;d<=n;d++){const i=(st.act+(st.buf.length>1?0:d))%n;
          if(sel.options[i].text.toLowerCase().startsWith(st.buf)){setAct(i);break;}}
      }
    });
    pop.addEventListener('mousemove',e=>{const el=e.target.closest('.dd-opt');if(el&&!el.hasAttribute('aria-disabled')&&+el.dataset.i!==st.act)setAct(+el.dataset.i,false);});
    pop.addEventListener('mousedown',e=>e.preventDefault());               // фокус остаётся на кнопке
    pop.addEventListener('click',e=>{const el=e.target.closest('.dd-opt');if(el)pick(+el.dataset.i);});
    sel.addEventListener('change',sync);

    // программная запись value / selectedIndex
    for(const p of ['value','selectedIndex']){
      const d=Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,p);
      Object.defineProperty(sel,p,{configurable:true,get(){return d.get.call(this);},set(v){d.set.call(this,v);sync();}});
    }
    new MutationObserver(()=>{if(!wrap.classList.contains('open'))rebuild();else{const a=st.act;rebuild();setAct(Math.min(a,st.opts.length-1),false);}})
      .observe(sel,{childList:true,subtree:true,attributes:true,attributeFilter:['disabled','selected']});
    rebuild();
  }

  document.addEventListener('mousedown',e=>{
    if(openDd&&!openDd.wrap.contains(e.target)&&!openDd.pop.contains(e.target)) openDd.close(false);
  },true);
  addEventListener('resize',()=>{if(openDd)openDd.close(false);});
  addEventListener('blur',()=>{if(openDd)openDd.close(false);});
  document.addEventListener('scroll',e=>{if(openDd&&!openDd.pop.contains(e.target))openDd.close(false);},true);

  function scan(root){(root||document).querySelectorAll('select').forEach(build);}
  function init(){
    scan();
    new MutationObserver(ms=>{for(const m of ms)for(const n of m.addedNodes){
      if(n.nodeType!==1)continue; if(n.tagName==='SELECT')build(n); else if(n.querySelector)scan(n);}})
      .observe(document.body,{childList:true,subtree:true});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init); else init();
  window.ctlSelect={scan};
})();
