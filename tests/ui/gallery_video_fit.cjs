// Render real gallery cards in Chromium. Videos are tiny CPU-generated color
// clips and all application requests are intercepted; no user media is read.
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

const cases = [
  {name:'near-832x480.mp4', width:832, height:480, color:'red', metadata:{width:1344,height:768}},
  {name:'near-1344x768.mp4', width:1344, height:768, color:'green'},
  {name:'near-1280x704.mp4', width:1280, height:704, color:'blue'},
  {name:'portrait-480x800.mp4', width:480, height:800, color:'yellow'},
  {name:'square-640x640.mp4', width:640, height:640, color:'purple'},
  {name:'ultrawide-1920x800.mp4', width:1920, height:800, color:'cyan'},
  {name:'missing-meta-832x480.mp4', width:832, height:480, color:'orange', metadata:null},
  {name:'invalid-meta-832x480.mp4', width:832, height:480, color:'pink', metadata:{width:0,height:480}},
  {name:'stale-size-1280x704.mp4', width:1280, height:704, color:'white', staleSize:true},
  {name:'recycled-before.mp4', width:832, height:480, color:'lime', metadata:{width:1344,height:768}},
  {name:'recycled-after.mp4', width:1280, height:704, color:'navy', metadata:{width:1280,height:704}},
];

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'maestro-gallery-video-fit-'));
let browser;

(async () => {
  const media = new Map();
  for (const item of cases) {
    const output = path.join(tempDir, item.name);
    const encoded = spawnSync(process.env.MAESTRO_FFMPEG || 'ffmpeg', [
      '-hide_banner','-loglevel','error','-y',
      '-f','lavfi','-i',`color=c=${item.color}:s=${item.width}x${item.height}:r=5:d=1`,
      '-an','-c:v','libx264','-threads','1','-preset','ultrafast','-pix_fmt','yuv420p',
      '-movflags','+faststart',output,
    ], {encoding:'utf8'});
    assert.equal(encoded.status, 0, `ffmpeg failed for ${item.name}: ${encoded.stderr || encoded.error || ''}`);
    const buffer = fs.readFileSync(output);
    media.set(item.name, buffer);
    item.byteLength = buffer.length;
  }

  const files = cases.map((item, index) => ({
    name:item.name, workspace:'A', type:'video', mode:'video',
    url:`/api/v1/file/${item.name}?workspace=A`,
    size:item.byteLength, created_at:index + 1, metadata_ready:true,
  }));
  const recycledBefore = files.find(file => file.name === 'recycled-before.mp4');
  const recycledAfter = {...files.find(file => file.name === 'recycled-after.mp4'), size:recycledBefore.size};
  const imageFile = {name:'viewer-image.svg',workspace:'A',type:'image',mode:'image',
    url:'/api/v1/file/viewer-image.svg?workspace=A',size:200,created_at:50,metadata_ready:true};
  const allFiles = [...files, recycledAfter, imageFile];
  const fileByName = new Map(allFiles.map(file => [file.name, file]));
  const itemByName = new Map(cases.map(item => [item.name, item]));
  const requests = {video:[],metadata:[],mediaInfo:[],thumbnail:[],image:[]};

  const bundle = await esbuild.build({stdin:{contents:`
    import React, {useState} from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    import {getGalleryVideoFrameAspect,getTrustedGalleryVideoDimensions,getGalleryVideoIdentity} from './src/lib/galleryVideoFit';
    const files=window.__galleryVideoFitFiles;
    const recyclingFile=files.find(file=>file.name==='recycled-before.mp4');
    const imageFile=files.find(file=>file.name==='viewer-image.svg');
    useStore.setState({models:[],servicesConfig:{},outputs:files,activeWorkspace:'A',browsingUploads:false,
      workspaces:[{name:'Workspace A'}],jobs:[],isGenerating:false});
    window.galleryVideoFitPolicy={getGalleryVideoFrameAspect,getTrustedGalleryVideoDimensions,getGalleryVideoIdentity};
    window.galleryVideoFitMeasurements={};
    function App(){
      const [recycled,setRecycled]=useState(recyclingFile);
      window.replaceGalleryVideo=()=>setRecycled(files.find(file=>file.name==='recycled-after.mp4'));
      return <main style={{overflow:'hidden',display:'flex',minHeight:0}}>
        <div id="feed" style={{overflowY:'auto',flex:1,padding:12}}>
          {files.filter(file=>file.type==='video'&&file.name!=='recycled-after.mp4').map((file,index)=><MediaFeedItem
            key={file.name} file={file} index={index} isActive={index===0} onActivate={()=>{}}
            onPlaybackStart={()=>{}} onMeasured={(i,height)=>window.galleryVideoFitMeasurements[i]=height}
            onOpenViewer={file=>window.galleryViewerOpened=file.name} style={{position:'relative',marginBottom:12,width:'100%'}}/>)}
          <MediaFeedItem key="recycled-row" file={recycled} index={100} isActive={false}
            onActivate={()=>{}} onPlaybackStart={()=>{}}
            onMeasured={(i,height)=>window.galleryVideoFitMeasurements[i]=height}
            onOpenViewer={()=>{}} style={{position:'relative',marginBottom:12,width:'100%'}}/>
          <MediaFeedItem key={imageFile.name} file={imageFile} index={101} isActive={false}
            onActivate={()=>{}} onPlaybackStart={()=>{}}
            onMeasured={(i,height)=>window.galleryVideoFitMeasurements[i]=height}
            onOpenViewer={file=>window.galleryViewerOpened=file.name}
            style={{position:'relative',marginBottom:12,width:'100%'}}/>
        </div>
      </main>;
    }
    createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
  `,resolveDir:path.join(root,'ui'),loader:'tsx'},bundle:true,write:false,jsx:'automatic',
    define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  browser=await chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});

  const page=await browser.newPage({viewport:{width:1600,height:8000}});
  page.setDefaultTimeout(10000);
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(value=>{window.__galleryVideoFitFiles=value;},allFiles);
  await page.route('**/*',async route=>{
    const url=new URL(route.request().url());
    const pathName=url.pathname;
    if(pathName==='/') return route.fulfill({contentType:'text/html',body:
      '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100dvh"></div>'});
    const metadataMatch=pathName.match(/^\/api\/v1\/outputs\/([^/]+)\/metadata$/);
    if(metadataMatch){
      const name=decodeURIComponent(metadataMatch[1]);
      requests.metadata.push(name);
      if(name==='recycled-after.mp4') await new Promise(resolve=>setTimeout(resolve,350));
      const item=itemByName.get(name);
      const file=fileByName.get(name);
      const meta=item?(item.metadata===null?{}:item.metadata||{width:item.width,height:item.height}):null;
      const mediaInfo=item&&meta?{
        width:meta.width,height:meta.height,
        size_bytes:item.staleSize?file.size+1:(name==='recycled-after.mp4'?file.size+1:file.size),
        duration_seconds:1,
      }:undefined;
      return route.fulfill({json:{source:'sidecar',params:{},...(mediaInfo?{media_info:mediaInfo}:{})}});
    }
    if(pathName.startsWith('/api/v1/media-info/')){
      requests.mediaInfo.push(decodeURIComponent(pathName.split('/').pop()));
      return route.fulfill({json:{duration_seconds:1}});
    }
    if(pathName.startsWith('/api/v1/thumbnail/')){
      requests.thumbnail.push(pathName);
      return route.fulfill({contentType:'image/svg+xml',body:
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="#315"/></svg>'});
    }
    if(pathName.startsWith('/api/v1/file/')){
      const name=decodeURIComponent(pathName.split('/').pop());
      if(name==='viewer-image.svg'){
        requests.image.push(name);
        return route.fulfill({contentType:'image/svg+xml',body:
          '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40"><rect width="80" height="40" fill="tomato"/></svg>'});
      }
      requests.video.push(name);
      const buffer=media.get(name);
      assert.ok(buffer,`unexpected video file request: ${name}`);
      return route.fulfill({status:200,contentType:'video/mp4',body:buffer});
    }
    throw new Error('Unexpected request: '+route.request().method()+' '+pathName);
  });

  await page.goto('http://gallery-video-fit.test');
  await page.addStyleTag({content:css});
  await page.addScriptTag({content:bundle.outputFiles[0].text});
  const card=fileName=>page.locator(`[data-output-name="${fileName}"]`);
  const frameRatio=async fileName=>card(fileName).locator('video').evaluate(video=>{
    const rect=video.parentElement.getBoundingClientRect();
    return rect.width/rect.height;
  });
  const ratioFor=(width,height)=>width/height;
  const waitForRatio=async(fileName,expected)=>page.waitForFunction(({fileName,expected})=>{
    const card=document.querySelector(`[data-output-name="${fileName}"]`);
    const video=card?.querySelector('video');
    if(!video)return false;
    const rect=video.parentElement.getBoundingClientRect();
    return Math.abs(rect.width/rect.height-expected)<0.004;
  },{fileName,expected});

  try {
    await page.waitForFunction(expected=>document.querySelectorAll('[data-gallery-media="true"]').length===expected,cases.length);
    await page.waitForFunction(()=>window.galleryVideoFitMeasurements[0]>0);
    // Matching-size metadata is a temporary seed, while the browser's native
    // dimensions must replace it as soon as the video metadata is available.
    await waitForRatio('near-832x480.mp4',ratioFor(1344,768));
    assert.ok(Math.abs(await frameRatio('near-832x480.mp4')-ratioFor(1344,768))<0.004);
    for(const item of cases.filter(item=>{
      const near=item.width/item.height>=16/9*0.95&&item.width/item.height<=16/9*1.05;
      const validSeed=!item.staleSize&&item.metadata!==null
        &&(!item.metadata||(item.metadata.width>0&&item.metadata.height>0));
      return near&&validSeed&&item.name!=='recycled-before.mp4'&&item.name!=='recycled-after.mp4';
    })){
      if(item.name==='near-832x480.mp4'||item.name==='recycled-before.mp4') continue;
      await waitForRatio(item.name,ratioFor(item.width,item.height));
    }
    for(const item of cases.filter(item=>['portrait-480x800.mp4','square-640x640.mp4','ultrawide-1920x800.mp4',
      'missing-meta-832x480.mp4','invalid-meta-832x480.mp4','stale-size-1280x704.mp4'].includes(item.name))){
      await waitForRatio(item.name,16/9);
    }

    const policy=await page.evaluate(()=>{
      const fit=window.galleryVideoFitPolicy;
      const identity={id:'asset',workspace:'A',name:'clip.mp4',url:'/file/clip.mp4',size:100,created_at:1};
      return {
        near:fit.getGalleryVideoFrameAspect({width:832,height:480}),
        portrait:fit.getGalleryVideoFrameAspect({width:480,height:800}),
        invalid:fit.getGalleryVideoFrameAspect({width:0,height:480}),
        trusted:fit.getTrustedGalleryVideoDimensions({width:832,height:480,size_bytes:100},100),
        mismatched:fit.getTrustedGalleryVideoDimensions({width:832,height:480,size_bytes:101},100),
        sameContentIdentity:fit.getGalleryVideoIdentity(identity)===fit.getGalleryVideoIdentity({...identity,metadata_updated_at:2}),
      };
    });
    assert.equal(policy.near,'832 / 480');
    assert.equal(policy.portrait,undefined);
    assert.equal(policy.invalid,undefined);
    assert.deepEqual(policy.trusted,{width:832,height:480});
    assert.equal(policy.mismatched,null);
    assert.equal(policy.sameContentIdentity,true,'sidecar updates do not change content identity');

    const imageCard=card('viewer-image.svg');
    const imageAspect=await imageCard.locator(':scope > div').first().evaluate(el=>{
      const r=el.getBoundingClientRect();return r.width/r.height;
    });
    assert.ok(Math.abs(imageAspect-16/9)<0.004,'image cards retain their fixed 16:9 frame');
    await imageCard.scrollIntoViewIfNeeded();
    await imageCard.getByRole('button',{name:'Enlarge viewer-image.svg'}).click();
    assert.equal(await page.evaluate(()=>window.galleryViewerOpened),'viewer-image.svg','image viewer action remains connected');

    const videoStates=await page.locator('video').evaluateAll(videos=>videos.map(video=>({
      preload:video.preload,controls:video.controls,playsInline:video.playsInline,
      objectFit:getComputedStyle(video).objectFit,
    })));
    assert.ok(videoStates.every(video=>video.preload==='none'&&video.controls&&video.playsInline&&video.objectFit==='contain'));
    assert.equal(requests.video.length,0,'video files are not fetched before an explicit load');

    await page.evaluate(async()=>Promise.all(Array.from(document.querySelectorAll('video')).map(video=>new Promise((resolve,reject)=>{
      video.muted=true;
      if(video.readyState>=1){video.pause();resolve();return;}
      video.addEventListener('loadedmetadata',()=>{video.pause();resolve();},{once:true});
      video.addEventListener('error',()=>reject(new Error('Synthetic video failed to load')),{once:true});
      video.play().catch(reject);
    }))));
    await page.waitForFunction(()=>Array.from(document.querySelectorAll('video')).every(video=>video.videoWidth>0&&video.videoHeight>0));
    const initialVideoNames=new Set(requests.video);
    for(const item of cases.filter(item=>item.name!=='recycled-after.mp4')){
      assert.ok(initialVideoNames.has(item.name),`${item.name} is loaded only after playback is requested`);
    }
    for(const item of cases.filter(item=>item.name!=='recycled-after.mp4')){
      const dimensions=await card(item.name).locator('video').first().evaluate(video=>({width:video.videoWidth,height:video.videoHeight}));
      assert.deepEqual(dimensions,{width:item.width,height:item.height},`${item.name} exposes native dimensions`);
      const near=item.width/item.height>=16/9*0.95&&item.width/item.height<=16/9*1.05;
      await waitForRatio(item.name,near?ratioFor(item.width,item.height):16/9);
    }
    await waitForRatio('recycled-before.mp4',ratioFor(832,480));

    const initialCardHeight=await page.evaluate(()=>window.galleryVideoFitMeasurements[0]);
    await page.setViewportSize({width:900,height:1200});
    await page.waitForFunction(previous=>{
      const next=window.galleryVideoFitMeasurements[0];
      return next>0&&Math.abs(next-previous)>50;
    },initialCardHeight);
    await waitForRatio('near-832x480.mp4',ratioFor(832,480));

    await page.locator('[data-feed-index="100"]').scrollIntoViewIfNeeded();
    await page.evaluate(()=>window.replaceGalleryVideo());
    await page.waitForFunction(()=>document.querySelector('[data-feed-index="100"] video')?.currentSrc.includes('recycled-after.mp4'));
    await waitForRatio('recycled-after.mp4',16/9);
    assert.ok(!requests.video.includes('recycled-after.mp4'),'replacing a virtualized row does not preload its new video');
    const targetVideo=page.locator('[data-feed-index="100"] video');
    await targetVideo.evaluate(video=>{video.muted=true;void video.play().catch(()=>{});});
    await page.waitForFunction(()=>{
      const video=document.querySelector('[data-feed-index="100"] video');
      return video?.videoWidth===1280&&video.videoHeight===704;
    });
    await waitForRatio('recycled-after.mp4',ratioFor(1280,704));
    assert.equal(requests.mediaInfo.filter(name=>name==='recycled-after.mp4').length,1,
      'the existing duration lookup runs once for the recycled file; fitting adds no probes');
    assert.equal(requests.metadata.filter(name=>name==='recycled-after.mp4').length,1,
      'the recycled file gets its existing lazy metadata request once');

    const screenshotDir=process.env.MAESTRO_TEST_ARTIFACT_DIR
      ?path.resolve(process.env.MAESTRO_TEST_ARTIFACT_DIR):tempDir;
    fs.mkdirSync(screenshotDir,{recursive:true});
    await card('near-832x480.mp4').screenshot({path:path.join(screenshotDir,'near-832x480.png')});
    await card('near-832x480.mp4').getByRole('button',{name:'More clip actions'}).click();
    await page.getByRole('menu',{name:'Clip actions'}).getByRole('menuitem',{name:'Open full-screen gallery'}).click();
    assert.equal(await page.evaluate(()=>window.galleryViewerOpened),'near-832x480.mp4','video viewer action remains connected');
    assert.deepEqual(errors,[]);
    console.log('PASS gallery video fit: near ratios, fallback ratios, metadata validity, native load, recycled row, resize, image/video viewer behavior');
    console.log(`Screenshot: ${path.join(screenshotDir,'near-832x480.png')}`);
  } finally {
    if(browser) await browser.close();
    for(const item of cases) fs.rmSync(path.join(tempDir,item.name),{force:true});
    fs.rmdirSync(tempDir);
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
