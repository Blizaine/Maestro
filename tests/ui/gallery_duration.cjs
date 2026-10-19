// Isolated measured-duration badges for real gallery cards and thumbnails.
// Every request is intercepted; no Maestro process or user media is touched.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const playwright = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const screenshotDir=path.join(root,'.codex-tmp/gallery-duration-20261014');
fs.mkdirSync(screenshotDir,{recursive:true});

const files = [
  {name:'clip.mp4',workspace:'Workspace A'},
  {name:'clip.mp4',workspace:'Workspace B'},
  {name:'boolean.mp4',workspace:'Bad'},
  {name:'missing-seed.mp4',workspace:'Fallback'},
  {name:'unknown.mp4',workspace:'Unknown'},
  ...Array.from({length:20},(_,index)=>({name:`lazy-${index+5}.mp4`,workspace:'Lazy'})),
].map((file,index)=>({
  ...file,id:`${file.workspace}/${file.name}`,type:'video',mode:'video',favorite:false,
  size:100+index,created_at:1000+index,metadata_ready:true,
  url:`/api/v1/file/${encodeURIComponent(file.name)}?workspace=${encodeURIComponent(file.workspace)}`,
}));

(async () => {
  const bundle = await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React, {useState} from 'react';
    import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    import {ThumbnailGallery} from './src/components/MainContent/ThumbnailGallery';
    import {formatMediaDuration, getMediaInfoUrl} from './src/lib/mediaDuration';
    window.store=useStore;
    window.formatMediaDuration=formatMediaDuration;
    window.getMediaInfoUrl=getMediaInfoUrl;
    window.setup=initialFiles=>{
      useStore.setState({models:[],servicesConfig:{},outputs:initialFiles,outputsTotal:initialFiles.length,
        activeWorkspace:'Workspace A',browsingAllFolders:true,browsingUploads:false,mediaFilter:'all',jobs:[]});
      function Harness(){
        const [items,setItems]=useState(initialFiles);
        window.replaceFirst=patch=>{
          const next=items.map((file,index)=>index===0?{...file,...patch}:file);
          setItems(next);
          useStore.setState({outputs:next,outputsTotal:next.length});
        };
        return <div style={{height:'100%',display:'flex',minWidth:0}}>
          <main id="feed" style={{height:'100%',flex:'1 1 auto',minWidth:0,overflowY:'auto'}}>
            <div style={{maxWidth:620,padding:10}}>{items.map((file,index)=><MediaFeedItem key={file.workspace+'/'+file.name}
              file={file} index={index} isActive={index===0} onActivate={()=>{}} onPlaybackStart={()=>{}}
              onMeasured={()=>{}} onOpenViewer={()=>{window.fullscreenOpens=(window.fullscreenOpens||0)+1}}
              style={{position:'relative',marginBottom:12}}/>)}</div>
          </main>
          <ThumbnailGallery activeIndex={0} onThumbnailClick={()=>{}}/>
        </div>;
      }
      window.fullscreenOpens=0;
      createRoot(document.getElementById('root')).render(<Harness/>);
    };
  `},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent',plugins:[{
    name:'resolve-from-ui',
    setup(build){
      const uiRoot=path.join(root,'ui');
      const extensions=['.tsx','.ts','.jsx','.js','.mjs','.cjs','.json','.css'];
      build.onResolve({filter:/.*/},args=>{
        if(args.path.startsWith('.')||path.isAbsolute(args.path)) {
          const base=path.resolve(args.resolveDir||path.dirname(args.importer),args.path);
          const candidates=[base,...extensions.map(extension=>`${base}${extension}`),
            ...extensions.map(extension=>path.join(base,`index${extension}`))];
          const resolved=candidates.find(candidate=>{
            try{return fs.statSync(candidate).isFile()}catch{return false}
          });
          if(resolved)return {path:resolved};
        } else {
          try{return {path:require.resolve(args.path,{paths:[uiRoot]})}}catch{}
        }
        throw new Error(`Unable to resolve UI bundle import: ${args.path}`);
      });
    },
  }]});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser=await playwright.chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try {
    for(const mobile of [false,true]) {
      const viewport=mobile?{width:390,height:844}:{width:1280,height:900};
      const page=await browser.newPage({viewport,hasTouch:mobile,isMobile:mobile});
      page.setDefaultTimeout(8000);
      const errors=[],mediaInfoRequests=[],videoRequests=[];
      const aCalls=new Map();
      let releaseFirstA;
      const firstAResponse=new Promise(resolve=>{releaseFirstA=resolve});
      page.on('pageerror',error=>errors.push(error.message));
      await page.route('**/*',async route=>{
        const request=route.request(),url=new URL(request.url()),key=url.pathname;
        const json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
        if(key==='/') return route.fulfill({contentType:'text/html',body:
          '<meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#root{height:100%;margin:0}#root{overflow:hidden}</style><div id="root"></div>'});
        if(key.startsWith('/api/v1/media-info/')) {
          const workspace=url.searchParams.get('workspace')||'';
          const filename=key.split('/').pop();
          mediaInfoRequests.push({url:url.pathname+url.search,workspace,filename});
          if(workspace==='Workspace A') {
            const count=(aCalls.get(workspace)||0)+1;
            aCalls.set(workspace,count);
            if(count===1 && !url.searchParams.has('v')) await firstAResponse;
            return json({duration_seconds:count===1?65.4:126.2});
          }
          if(filename==='clip.mp4'&&workspace==='Workspace B') return json({duration_seconds:3670});
          if(filename==='boolean.mp4') return json({duration_seconds:true});
          if(filename==='missing-seed.mp4') return route.fulfill({status:404,body:'Missing fixture facts'});
          if(filename==='unknown.mp4') return json({duration_seconds:0});
          return json({duration_seconds:13.5});
        }
        if(key.endsWith('/metadata')) {
          const workspace=url.searchParams.get('workspace')||'';
          const filename=decodeURIComponent(url.pathname.split('/').slice(-2,-1)[0]||'');
          const seed=workspace==='Workspace A'&&filename==='clip.mp4'?{duration_seconds:90,size_bytes:100}:
            workspace==='Workspace B'&&filename==='clip.mp4'?{duration_seconds:90,size_bytes:101}:
            filename==='missing-seed.mp4'?{duration_seconds:31.2,size_bytes:103}:undefined;
          return json({source:'sidecar',params:{prompt:'Duration fixture'},
            ...(seed===undefined?{}:{media_info:seed})});
        }
        if(key.startsWith('/api/v1/thumbnail/')) return route.fulfill({contentType:'image/svg+xml',body:
          '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="90"><rect width="160" height="90" fill="#334155"/></svg>'});
        if(key.startsWith('/api/v1/file/')&&key.endsWith('.mp4')) {
          videoRequests.push(url.pathname+url.search);
          return route.fulfill({status:404,body:'A video request is unexpected before playback'});
        }
        return json({checks:[],downloads:[],jobs:[],items:[],entries:[],queue:[],pipelines:[],recipes:[],
          loras:[],models:[],families:[],presets:[],capabilities:{},workspaces:[],characters:[]});
      });
      try {
        await page.goto('http://gallery-duration.test');
        await page.addStyleTag({content:css});
        await page.addScriptTag({content:bundle.outputFiles[0].text});
        await page.evaluate(files=>window.setup(files),files);

        const cardA=page.locator('[data-feed-index="0"]');
        const video=cardA.locator('video[data-gallery-media]');
        await video.waitFor();
        assert.equal(await video.getAttribute('preload'),'none','Feed video retains preload=none');
        assert.equal(await video.evaluate(element=>element.controls),true,'Native video controls remain enabled');

        await page.waitForFunction(()=>document.querySelector('[data-feed-index="0"] [data-media-duration-badge]')?.textContent==='1:30',null,{timeout:8000});
        await pause(120);
        assert.equal(mediaInfoRequests.filter(item=>item.workspace==='Workspace A').length,1,
          'Visible feed card and thumbnail share the same pending request');
        assert.equal(mediaInfoRequests.filter(item=>item.workspace==='Workspace B').length,1,
          'The same filename in a different workspace gets its own request');

        const mediaOverlay=await cardA.locator('[data-media-duration-badge]').evaluate(element=>{
          const badge=element.getBoundingClientRect(), parent=element.parentElement.getBoundingClientRect();
          const fullscreen=element.parentElement.querySelector('[aria-label="Open full-screen gallery"]')?.getBoundingClientRect();
          return {position:getComputedStyle(element).position,pointerEvents:getComputedStyle(element).pointerEvents,
            inside:badge.left>=parent.left&&badge.top>=parent.top&&badge.right<=parent.right&&badge.bottom<=parent.bottom,
            clearOfFullscreen:!fullscreen||badge.right<=fullscreen.left||badge.left>=fullscreen.right||badge.bottom<=fullscreen.top||badge.top>=fullscreen.bottom,
            clearOfNativeControls:badge.bottom<parent.bottom-30};
        });
        assert.deepEqual(mediaOverlay,{position:'absolute',pointerEvents:'none',inside:true,clearOfFullscreen:true,clearOfNativeControls:true},
          `${mobile?'mobile':'desktop'} feed badge overlays the poster without covering controls`);
        const fullscreen=cardA.getByRole('button',{name:'Open full-screen gallery'});
        await fullscreen.click();
        assert.equal(await page.evaluate(()=>window.fullscreenOpens),1,'Full-screen gallery button remains reachable');
        if(mobile) await page.getByTitle('Show thumbnails').click();

        releaseFirstA();
        await page.waitForFunction(()=>document.querySelector('[data-feed-index="0"] [data-media-duration-badge]')?.textContent==='1:05');
        await page.waitForFunction(()=>document.querySelector('[data-thumb-index="0"] [data-media-duration-badge]')?.textContent==='1:05');
        await page.waitForFunction(()=>document.querySelector('[data-thumb-index="1"] [data-media-duration-badge]')?.textContent==='1:01:10');
        assert.equal(await page.locator('[data-thumb-index="2"] [data-media-duration-badge]').count(),0,
          'A boolean duration is rejected instead of appearing as 0:00');
        assert.equal(await page.locator('[data-thumb-index="4"] [data-media-duration-badge]').count(),0,
          'Unknown or nonpositive duration has no badge');
        await page.screenshot({path:path.join(screenshotDir,mobile?'phone.png':'desktop.png')});

        const lazyUrl='/api/v1/media-info/lazy-20.mp4?workspace=Lazy';
        assert.equal(mediaInfoRequests.some(item=>item.url===lazyUrl),false,
          'A card and thumbnail outside their visible regions do not request duration');
        await page.locator('[data-feed-index="20"]').scrollIntoViewIfNeeded();
        const lazyDeadline=Date.now()+5000;
        while(!mediaInfoRequests.some(item=>item.url===lazyUrl)&&Date.now()<lazyDeadline) await pause(25);
        assert.ok(mediaInfoRequests.some(item=>item.url===lazyUrl),'Scrolling a card into view requests its duration');

        await page.locator('[data-feed-index="3"]').scrollIntoViewIfNeeded();
        await page.waitForFunction(()=>document.querySelector('[data-feed-index="3"] [data-media-duration-badge]')?.textContent==='0:31',
          null,{timeout:5000});
        assert.ok(mediaInfoRequests.some(item=>item.filename==='missing-seed.mp4'),
          'The isolated 404 was requested and a valid measured metadata seed remained available');

        await page.locator('[data-feed-index="0"]').scrollIntoViewIfNeeded();
        await page.evaluate(()=>window.replaceFirst({size:999,created_at:9999}));
        await page.waitForFunction(()=>document.querySelector('[data-feed-index="0"] [data-media-duration-badge]')?.textContent==='2:06',
          null,{timeout:5000});
        assert.equal(mediaInfoRequests.filter(item=>item.workspace==='Workspace A').length,2,
          'Changing only file size and created_at invalidates the previous duration');
        assert.equal(mediaInfoRequests.filter(item=>item.url==='/api/v1/media-info/clip.mp4?workspace=Workspace%20A').length,2,
          'The stat-only replacement rechecks the same managed URL');
        await page.evaluate(()=>window.replaceFirst({url:'/api/v1/file/clip.mp4?workspace=Workspace%20A&v=2'}));
        const replacementDeadline=Date.now()+5000;
        while(!mediaInfoRequests.some(item=>item.url==='/api/v1/media-info/clip.mp4?workspace=Workspace%20A&v=2')
          &&Date.now()<replacementDeadline) await pause(25);
        await page.waitForFunction(()=>document.querySelector('[data-feed-index="0"] [data-media-duration-badge]')?.textContent==='2:06',
          null,{timeout:5000});
        assert.ok(mediaInfoRequests.some(item=>item.url==='/api/v1/media-info/clip.mp4?workspace=Workspace%20A&v=2'),
          'Replacing a clip URL also invalidates the duration entry and keeps cache-busting query data');

        const helpers=await page.evaluate(()=>({
          formatted:[65.4,3670,0,-1,Infinity,true,'12'].map(window.formatMediaDuration),
          mediaInfoUrl:window.getMediaInfoUrl('/api/v1/file/a%20%23b.mp4?workspace=A%20B&v=2#t=0.1'),
          externalUrl:window.getMediaInfoUrl('https://elsewhere.test/api/v1/file/clip.mp4'),
        }));
        assert.deepEqual(helpers.formatted,['1:05','1:01:10',null,null,null,null,null]);
        assert.equal(helpers.mediaInfoUrl,'/api/v1/media-info/a%20%23b.mp4?workspace=A%20B&v=2',
          'Managed URL conversion preserves encoded filenames, workspace, and cache-busting query');
        assert.equal(helpers.externalUrl,null,'External media URLs are not sent to the managed endpoint');
        assert.deepEqual(videoRequests,[],'Browsing and duration probing never request or preload video bytes');
        assert.deepEqual(errors,[],`${mobile?'mobile':'desktop'} fixture has no browser errors`);
        console.log(`PASS gallery duration badges: ${mobile?'mobile':'desktop'}`);
      } finally {
        releaseFirstA();
        await page.close();
      }
    }
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
