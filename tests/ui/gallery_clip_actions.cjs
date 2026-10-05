// Real gallery cards inside a clipped feed. All requests are mocked; no jobs
// or user media are touched. Build ui first, then run this file with Node.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    const files=[0,1].map(index=>({name:'clip'+index+'.mp4',workspace:'A',type:'video',mode:'video',
      url:'/api/v1/file/clip'+index+'.mp4?workspace=A',size:100,created_at:1,metadata_ready:true}));
    useStore.setState({models:[],servicesConfig:{},outputs:files,activeWorkspace:'A',browsingUploads:false,
      workspaces:Array.from({length:12},(_,i)=>({name:'Workspace '+i})),jobs:[],isGenerating:false});
    createRoot(document.getElementById('root')).render(<React.StrictMode>
      <header style={{height:100,flexShrink:0,position:'relative',zIndex:40,background:'#151515'}}>Gallery toolbar</header>
      <main style={{overflow:'hidden',flex:1,display:'flex',minHeight:0}}>
        <div id="feed" style={{overflowY:'auto',flex:1,padding:12}}>
          {files.map((file,index)=><MediaFeedItem key={file.name} file={file} index={index} isActive={false}
            onActivate={()=>{}} onPlaybackStart={()=>{}} onMeasured={()=>{}} onOpenViewer={()=>{}}
            style={{position:'relative',marginBottom:16}}/>)}
        </div>
      </main>
    </React.StrictMode>);`, resolveDir: path.join(root, 'ui'), loader: 'tsx'},
    bundle: true, write: false, jsx: 'automatic', define: {'process.env.NODE_ENV':'"development"'}, logLevel:'silent'});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try {
    for(const viewport of [{width:390,height:660},{width:844,height:390},{width:1280,height:900}]) {
      const page=await browser.newPage({viewport,hasTouch:viewport.width<900,isMobile:viewport.width<900});
      page.setDefaultTimeout(7000);
      const errors=[];
      page.on('pageerror',error=>errors.push(error.message));
      await page.route('**/*',route=>{
        const url=new URL(route.request().url());
        if(url.pathname==='/') return route.fulfill({contentType:'text/html',body:
          '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="display:flex;flex-direction:column;height:100dvh"></div>'});
        if(url.pathname.endsWith('/metadata')) return route.fulfill({json:{source:'sidecar',params:{
          prompt:'A silent scene.',seed:42,model_type:'minimax_h3_fused_turbo'}}});
        if(url.pathname.startsWith('/api/v1/thumbnail/')) return route.fulfill({contentType:'image/svg+xml',body:
          '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="green"/></svg>'});
        if(url.pathname.startsWith('/api/v1/file/')) return route.fulfill({contentType:'video/mp4',body:''});
        throw new Error('Unexpected request: '+route.request().method()+' '+url.pathname);
      });
      await page.goto('http://clip-actions.test');
      await page.addStyleTag({content:css});
      await page.addScriptTag({content:bundle.outputFiles[0].text});
      const menu=page.getByRole('menu',{name:'Clip actions'});
      const settle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const inBounds=async()=>{
        await settle();
        const result=await menu.evaluate(el=>{
          const r=el.getBoundingClientRect(),v=window.visualViewport;
          return {left:r.left,top:r.top,right:r.right,bottom:r.bottom,
            minX:v?.offsetLeft || 0,minY:v?.offsetTop || 0,
            maxX:(v?.offsetLeft || 0)+(v?.width || innerWidth),maxY:(v?.offsetTop || 0)+(v?.height || innerHeight),
            overflow:getComputedStyle(el).overflowY,overscroll:getComputedStyle(el).overscrollBehaviorY};
        });
        assert.ok(result.left>=result.minX+11 && result.right<=result.maxX-11,JSON.stringify(result));
        assert.ok(result.top>=result.minY+11 && result.bottom<=result.maxY-11,JSON.stringify(result));
        assert.equal(result.overflow,'auto');
        assert.equal(result.overscroll,'contain');
      };
      const canHit=async locator=>{
        await locator.scrollIntoViewIfNeeded();
        await settle();
        assert.equal(await locator.evaluate(el=>{
          const r=el.getBoundingClientRect();
          return el.contains(document.elementFromPoint(r.left+r.width/2,r.top+r.height/2));
        }),true,'Menu action must be reachable, including over the toolbar');
      };
      for(const index of [0,1]) {
        const trigger=page.locator(`[data-feed-index="${index}"]`).getByRole('button',{name:'More clip actions'});
        await trigger.click();
        await menu.getByRole('menuitem',{name:'Regenerate with same settings'}).waitFor();
        await inBounds();
        await canHit(menu.getByRole('menuitem',{name:'Open full-screen gallery'}));
        await menu.getByRole('menuitem',{name:'Move to workspace'}).click();
        await canHit(menu.getByRole('menuitem',{name:'Workspace 11',exact:true}));
        await inBounds();
        const feedScroll=await page.locator('#feed').evaluate(el=>el.scrollTop);
        await canHit(menu.getByRole('menuitem',{name:'Delete output',exact:true}));
        assert.equal(await page.locator('#feed').evaluate(el=>el.scrollTop),feedScroll,'Menu scroll stays inside menu');
        await canHit(menu.getByRole('menuitem',{name:'Open full-screen gallery'}));
        await page.keyboard.press('Escape');
        await menu.waitFor({state:'hidden'});
        assert.equal(await trigger.evaluate(el=>el===document.activeElement),true);
      }
      // Simulate Safari chrome reducing/offsetting the visible viewport while
      // leaving the layout viewport larger. Exercise both resize and scroll.
      await page.setViewportSize({width:390,height:720});
      await page.locator('[data-feed-index="0"]').getByRole('button',{name:'More clip actions'}).click();
      await page.evaluate(()=>{
        const viewport=new EventTarget();
        Object.assign(viewport,{offsetLeft:0,offsetTop:42,width:390,height:500,scale:1});
        Object.defineProperty(window,'visualViewport',{configurable:true,value:viewport});
      });
      // Reopen to subscribe to the replacement viewport.
      await page.keyboard.press('Escape');
      await page.locator('[data-feed-index="0"]').getByRole('button',{name:'More clip actions'}).click();
      await inBounds();
      await page.evaluate(()=>{window.visualViewport.height=350;window.visualViewport.dispatchEvent(new Event('resize'));});
      await inBounds();
      await page.evaluate(()=>{window.visualViewport.offsetTop=70;window.visualViewport.dispatchEvent(new Event('scroll'));});
      await inBounds();
      await page.mouse.click(2,2);
      await menu.waitFor({state:'hidden'});
      assert.deepEqual(errors,[]);
      console.log(`PASS clip actions ${viewport.width}x${viewport.height}: first/last actions, workspace list, viewport changes, dismissal`);
      await page.close();
    }
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
