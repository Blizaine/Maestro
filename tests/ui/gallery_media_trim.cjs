// Isolated gallery + trim-dialog interactions. Every HTTP request is mocked;
// the only media created is tiny synthetic CPU media in .codex-tmp.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');

(async () => {
  const fixtures = path.join(root, '.codex-tmp/gallery-audio-trim-20260926');
  fs.mkdirSync(fixtures, {recursive:true});
  const ffmpeg = process.env.MAESTRO_FFMPEG || 'ffmpeg';
  const audioPath = path.join(fixtures, 'fixture.wav'), videoPath = path.join(fixtures, 'fixture.mp4');
  execFileSync(ffmpeg, ['-y','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=32000:duration=8','-ac','2',audioPath]);
  execFileSync(ffmpeg, ['-y','-v','error','-f','lavfi','-i','testsrc2=s=160x90:r=10:d=8','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',videoPath]);
  const audio = fs.readFileSync(audioPath), video = fs.readFileSync(videoPath);
  const bundle = await esbuild.build({stdin:{resolveDir:path.join(root,'ui'),loader:'tsx',contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    import {useStore} from './src/stores/useStore';
    import {useGalleryInputs} from './src/lib/galleryInputs';
    window.store=useStore; window.inputs=useGalleryInputs; window.received=[];
    useStore.setState({activeWorkspace:'Destination',browsingAllFolders:true,models:[],jobs:[]});
    window.register=()=>useGalleryInputs.setState({targets:['audio','video','image'].map(kind=>({
      id:kind,label:kind==='audio'?'soundtrack':kind==='video'?'control video':'start frame',kind,
      receive:async file=>{window.received.push({name:file.name,type:file.type,size:file.size});}
    }))});
    window.register();
    const files=['audio','video'].map((type,index)=>({name:'fixture.'+(type==='audio'?'wav':'mp4'),
      workspace:'Origin A',type,mode:type,url:'/api/v1/file/fixture.'+(type==='audio'?'wav':'mp4')+'?workspace=Origin%20A',
      size:100,created_at:index,favorite:false,metadata_ready:true}));
    createRoot(document.getElementById('root')).render(<React.StrictMode>{files.map((file,index)=><MediaFeedItem
      key={file.name} file={file} index={index} isActive={false} onActivate={()=>{}} onPlaybackStart={()=>{}} onMeasured={()=>{}}/>)}</React.StrictMode>);
  `},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const assets = path.join(root,'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser = await chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:process.env.MAESTRO_CHROME||'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  try {
    const page = await browser.newPage({viewport:{width:1000,height:950}});
    const errors=[], trims=[], probes=[];
    let failTrim=false, deferTrim=null, probeDuration=8;
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/*',async route=>{
      const request=route.request(), url=new URL(request.url()), key=url.pathname;
      const json=value=>route.fulfill({json:value});
      if(key==='/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="max-width:700px;margin:auto"></div>'});
      if(key==='/api/v1/editor/media/probe') {
        const body=request.postDataJSON();probes.push(body);
        return json({name:body.asset.name,type:body.asset.type,duration:probeDuration,width:160,height:90,fps:10,has_audio:true,audio_channels:2,audio_sample_rate:32000,size:100});
      }
      if(key==='/api/v1/editor/media/preview') return json({preview_id:'fixture',waveform:Array.from({length:512},(_,i)=>0.1+Math.abs(Math.sin(i/7))*0.8)});
      if(key==='/api/v1/media/trim') {
        const body=request.postDataJSON();trims.push(body);
        if(deferTrim) await deferTrim;
        if(failTrim) return route.fulfill({status:400,json:{detail:'Could not trim the selected range.'}});
        const isVideo=body.asset.name.endsWith('.mp4');
        return json({filename:isVideo?'excerpt.mp4':'excerpt.wav',media_type:isVideo?'video':'audio',mime_type:isVideo?'video/mp4':'audio/wav',duration:body.end_time-body.start_time});
      }
      if(key.includes('/file/')||key.includes('/uploads/')) {
        const isVideo=key.endsWith('.mp4'), bytes=isVideo?video:audio;
        const range=/bytes=(\d+)-(\d*)/.exec(request.headers().range||'');
        if(range) {
          const start=Number(range[1]),end=range[2]?Math.min(Number(range[2]),bytes.length-1):bytes.length-1;
          return route.fulfill({status:206,contentType:isVideo?'video/mp4':'audio/wav',headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${bytes.length}`},body:bytes.subarray(start,end+1)});
        }
        return route.fulfill({contentType:isVideo?'video/mp4':'audio/wav',body:bytes});
      }
      if(key.endsWith('/metadata')) return json({params:{prompt:'Fixture'}});
      if(key==='/api/v1/recipes') return json({recipes:[]});
      if(key.includes('/thumbnail/')) return route.fulfill({status:404});
      throw new Error('Unexpected request: '+request.method()+' '+key);
    });
    await page.goto('http://gallery-trim.test');
    await page.addStyleTag({content:css});
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    const open=async(type='audio')=>{
      const card=page.locator(`[data-feed-index="${type==='audio'?0:1}"]`);
      const toggle=card.getByTitle('More clip actions');
      if(await toggle.getAttribute('aria-expanded')!=='true') await toggle.click();
      const menu=page.getByRole('menu', {name:'Clip actions'});
      if(type==='audio') assert.equal(await menu.getByRole('menuitem',{name:/Use.*(control video|start frame)/}).count(),0,'Audio must never target image/video inputs');
      await menu.getByRole('menuitem',{name:type==='audio'?'Use audio as soundtrack':'Use video as control video',exact:true}).click();
      await page.getByRole('dialog').waitFor();
      await page.getByLabel('End (seconds)').waitFor();
    };
    const close=()=>page.getByRole('button',{name:'Close trim preview',exact:true}).click();
    const receiveCount=()=>page.evaluate(()=>window.received.length);

    await open();
    assert.equal(trims.length,0,'Opening a picker must not trim or dispatch');
    assert.equal(await receiveCount(),0);
    await close();
    assert.equal(await receiveCount(),0,'Cancel does not send media');
    await open();
    await page.getByRole('button',{name:'Use full clip',exact:true}).click();
    await page.getByRole('dialog').waitFor({state:'hidden'});
    assert.equal(trims.length,0,'Use full clip skips transcoding');
    assert.deepEqual(await page.evaluate(()=>window.received.at(-1)),{name:'fixture.wav',type:'audio/wav',size:audio.length});
    assert.equal(probes.at(-1).asset.workspace,'Origin A','Keep source folder independent of destination workspace');

    await open();
    await page.getByLabel('Start (seconds)').fill('1.25');
    await page.getByRole('button',{name:'3s',exact:true}).click();
    assert.equal(await page.getByLabel('End (seconds)').inputValue(),'4.25');
    await page.getByRole('button',{name:'Zoom to selection',exact:true}).click();
    await page.getByRole('slider',{name:'Trim start',exact:true}).focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.getByLabel('Start (seconds)').inputValue(),'1.35');
    await page.getByLabel('Start (seconds)').fill('1.25');
    await page.getByRole('button',{name:'Use selection',exact:true}).dblclick();
    await page.getByRole('dialog').waitFor({state:'hidden'});
    assert.equal(trims.length,1,'Double clicks submit only one trim');
    assert.deepEqual(trims[0],{asset:{name:'fixture.wav',origin:'output',workspace:'Origin A'},start_time:1.25,end_time:4.25});
    assert.equal(await page.evaluate(()=>window.received.at(-1).type),'audio/wav');

    await open('video');
    await page.getByLabel('Start (seconds)').fill('2');
    await page.getByLabel('End (seconds)').fill('1');
    assert.equal(await page.getByRole('button',{name:'Use selection',exact:true}).isDisabled(),true,'Invalid range cannot submit');
    await page.getByLabel('End (seconds)').fill('2.4');
    await page.getByRole('button',{name:'Preview selection',exact:true}).click();
    await page.waitForFunction(()=>{const video=document.querySelector('[role="dialog"] video');return video&&!video.paused&&video.currentTime>=2;});
    await page.waitForFunction(()=>{const video=document.querySelector('[role="dialog"] video');return video&&video.paused&&video.currentTime<=2.05;});
    await page.screenshot({path:path.join(fixtures,'trim-video-desktop.png')});
    failTrim=true;
    const beforeFailure=await receiveCount();
    await page.getByRole('button',{name:'Use selection',exact:true}).click();
    await page.getByRole('dialog').getByRole('alert').filter({hasText:'Could not trim'}).waitFor();
    assert.equal(await receiveCount(),beforeFailure,'Failed trim never sends media');
    failTrim=false;
    await page.getByRole('button',{name:'Use selection',exact:true}).click();
    await page.getByRole('dialog').waitFor({state:'hidden'});
    assert.equal(await page.evaluate(()=>window.received.at(-1).type),'video/mp4');

    await open();
    await page.evaluate(()=>window.inputs.setState(state=>({targets:state.targets.map(target=>({...target,label:'a different input'}))})));
    assert.equal(await page.getByRole('button',{name:'Use full clip',exact:true}).isDisabled(),true);
    await page.getByRole('alert').filter({hasText:'no longer available'}).waitFor();
    await close();
    await page.evaluate(()=>window.register());
    await open();
    await page.getByRole('button',{name:'3s',exact:true}).click();
    let release;
    deferTrim=new Promise(resolve=>{release=resolve;});
    const beforeStale=await receiveCount(), beforeTrims=trims.length;
    await page.getByRole('button',{name:'Use selection',exact:true}).click();
    while(trims.length===beforeTrims) await new Promise(resolve=>setTimeout(resolve,20));
    await page.evaluate(()=>window.inputs.setState({targets:[]}));
    release();deferTrim=null;
    await page.getByRole('button',{name:'Close trim preview',exact:true}).waitFor();
    await page.waitForFunction(()=>!document.querySelector('[aria-label="Close trim preview"]').disabled);
    assert.equal(await receiveCount(),beforeStale,'Removed input cannot receive a late excerpt');
    await close();

    await page.evaluate(()=>window.register());
    probeDuration=480;
    await page.setViewportSize({width:390,height:844});
    await open();
    await page.getByLabel('Start (seconds)').fill('300');
    await page.getByRole('button',{name:'5s',exact:true}).click();
    await page.getByRole('button',{name:'Zoom to selection',exact:true}).click();
    const startHandle=page.getByRole('slider',{name:'Trim start',exact:true});
    const bounds=await startHandle.boundingBox();
    await page.mouse.move(bounds.x+bounds.width-2,bounds.y+bounds.height/2);
    await page.mouse.down();await page.mouse.move(bounds.x+bounds.width+12,bounds.y+bounds.height/2);await page.mouse.up();
    assert.ok(Number(await page.getByLabel('Start (seconds)').inputValue())>300,'Zoomed handles adjust a short range within a long clip');
    const dialogBounds=await page.getByRole('dialog').boundingBox();
    assert.ok(dialogBounds.x>=0&&dialogBounds.x+dialogBounds.width<=390&&dialogBounds.height<=844,'Picker fits a phone');
    await page.screenshot({path:path.join(fixtures,'trim-audio-mobile.png')});
    await page.keyboard.press('Escape');
    await page.getByRole('dialog').waitFor({state:'hidden'});
    assert.equal(errors.length,0,errors.join('\n'));
    console.log('Gallery trim passed: audio routing, source workspace, cancel/full/range, keyboard/drag/zoom/mobile, video range playback, invalid range, server failure/retry, double submit and stale targets.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
