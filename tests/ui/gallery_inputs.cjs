// Real gallery menus and sidecar inputs, with all uploads isolated from the app.
// node tests/ui/gallery_inputs.cjs http://127.0.0.1:<Maestro port>
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const base = process.argv[2];
if (!base) throw new Error('Pass the running Maestro URL (read-only model catalogue).');
const read = async endpoint => {
  const response = await fetch(base + endpoint);
  assert.ok(response.ok, endpoint);
  return response.json();
};

(async () => {
  const catalogue = await read('/api/v1/models');
  const ids = ['minimax_h3_ref2va_fused_turbo', 'minimax_h3_fused_turbo', 'viggle_animate', 'qwen_image_21_7B', 'ltx2_22B_distilled_1_1'];
  const options = Object.fromEntries(await Promise.all(ids.map(async id => [id, await read('/api/v1/model-options/' + id)])));
  const fixtures = path.join(root, '.codex-tmp/gallery-inputs');
  fs.mkdirSync(fixtures, {recursive:true});
  const videoPath = path.join(fixtures, 'clip.mp4');
  execFileSync(process.env.MAESTRO_FFMPEG || 'ffmpeg', ['-y', '-v', 'error', '-f', 'lavfi', '-i',
    'color=c=red:s=64x64:r=10:d=2', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', videoPath]);
  const video = fs.readFileSync(videoPath);
  const picture = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=', 'base64');
  const bundle = await esbuild.build({stdin: {contents: `
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import {useStore} from './src/stores/useStore';
    import {Sidebar} from './src/components/Sidebar/Sidebar';
    import {MediaFeedItem} from './src/components/MainContent/MediaFeedItem';
    import {useGalleryInputs, sendToGalleryInput} from './src/lib/galleryInputs';
    window.store=useStore; window.galleryInputs=useGalleryInputs; window.sendToInput=sendToGalleryInput;
    window.baseState=useStore.getState();
    const files=['image','video'].map((type,index)=>({name:type==='image'?'same.png':'clip.mp4',
      workspace:'Origin A',type,mode:type,url:'/api/v1/file/'+(type==='image'?'same.png':'clip.mp4')+'?workspace=Origin%20A',
      size:100,created_at:index,favorite:false,metadata_ready:true}));
    window.mount=()=>createRoot(document.getElementById('root')).render(<React.StrictMode><Sidebar/>
      <main style={{flex:1,minWidth:0,overflow:'auto',padding:16}}>{files.map((file,index)=><MediaFeedItem key={file.name}
        file={file} index={index} isActive={false} onActivate={()=>{}} onPlaybackStart={()=>{}} onMeasured={()=>{}}/>)}</main>
      </React.StrictMode>);`, resolveDir: path.join(root, 'ui'), loader: 'tsx'}, bundle: true, write: false,
    jsx: 'automatic', define: {'process.env.NODE_ENV': '"development"'}, logLevel: 'silent'});
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await chromium.launch({headless:true, ...(process.platform === 'win32' ? {
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1100,height:900}});
    const errors=[], downloads=[], uploads=[];
    let failUpload=false;
    page.on('pageerror',error=>errors.push(error.message));
    await page.route('**/*', async route => {
      const url=new URL(route.request().url()), key=url.pathname;
      const json=body=>route.fulfill({json:body});
      if(key==='/') return route.fulfill({contentType:'text/html',body:'<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="display:flex;height:100dvh"></div>'});
      if(key==='/api/v1/models') return json(catalogue);
      if(key.includes('/model-options/')) return json(options[key.split('/').pop()] || {});
      if(key.includes('/defaults/')) return json({});
      if(key==='/api/v1/upload') {
        if(failUpload) return route.fulfill({status:500,body:'Upload unavailable'});
        const body=route.request().postDataBuffer();
        const name=/filename="([^"]+)"/.exec(body.toString('latin1'))?.[1];
        uploads.push(name);
        return json({filename:name,path:'/uploads/'+name,url:'/api/v1/file/'+name,duration_seconds:2,has_audio:true,fps:10});
      }
      if(key.includes('/api/v1/file/')) {
        downloads.push(url.pathname+url.search);
        const range=/bytes=(\d+)-(\d*)/.exec(route.request().headers().range || '');
        if(key.endsWith('.mp4') && range) {
          const start=Number(range[1]), end=range[2]?Math.min(Number(range[2]),video.length-1):video.length-1;
          return route.fulfill({status:206,contentType:'video/mp4',headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${video.length}`},body:video.subarray(start,end+1)});
        }
        return route.fulfill({contentType:key.endsWith('.mp4')?'video/mp4':'image/png',body:key.endsWith('.mp4')?video:picture});
      }
      if(key.endsWith('/metadata')) return json({source:'sidecar',params:{prompt:'Test media',seed:42}});
      if(key==='/api/v1/characters') return json({characters:[]});
      if(key.includes('/media-flow/capabilities')) return json({neural_rendering:{available:false},frame_generation:{available:false,factors:[]}});
      if(key==='/api/v1/system-stats') return json({gpu:{available:false},ram:{used_gb:1,total_gb:32,percent:3},cpu:{percent:1}});
      if(key==='/api/v1/generate') throw new Error('This test must never generate media.');
      return json({items:[],models:[],loras:[],presets:[],recipes:[],jobs:[],downloads:[],status:'available',configured:true});
    });
    await page.goto('http://gallery-inputs.test');
    await page.addStyleTag({content:css});
    await page.addScriptTag({content:bundle.outputFiles[0].text});
    await page.evaluate(({catalogue,options})=>{
      window.configure=(mode='video',workflow='references',id='minimax_h3_ref2va_fused_turbo',sidebarMode='studio')=>{
        window.store.setState({models:catalogue.models,families:catalogue.families,enabledModels:new Set(catalogue.models.map(m=>m.model_type)),
          modelOptions:options[id],generationMode:mode,sidebarMode,sidebarOpen:false,studioVideoWorkflow:workflow,
          studioImageWorkflow:mode==='image'?workflow:'generate',editSubMode:workflow,
          selectedModelPerMode:{video:id,image:'qwen_image_21_7B'},activeWorkspace:'Destination',browsingAllFolders:true,
          startImage:null,endImage:null,imageRefs:[],imageRefType:'',directorH3References:[],directorStep:'upload',
          directorSkill:'music_video',directorMusicSource:'upload',
          directorLoading:false,directorAudioFile:null,directorReferenceImage:null,directorReferenceImageUrl:null,
          toolsTool:'upscale',toolsUpscaleMedia:mode==='tools'?workflow:'video',toolsSourcePath:'',toolsSourceUrl:'',
          editVideoFile:null,editVideoPath:'',editVideoUrl:'',editVideoDuration:0,editVideoResolution:'',
          durationSeconds:14.4,slidingWindowSeconds:14.4,slidingWindowOverlap:18,slidingWindowLocked:false,
          jobs:[],isGenerating:false,isEnhancing:false,promptEnhanceError:null,
          // Do not download automatic editing LoRAs in a UI test.
          ensureEditAnythingLora:async()=>{},ensureTransitionLoraForBlend:async()=>{},
          params:{...window.baseState.params,model_type:id,prompt:'A scene.',image_mode:workflow==='extend'?3:workflow==='blend'?4:0,
            video_guide:undefined,image_start:undefined,image_end:undefined,minimax_h3_references:[],_viggle_edited_frame:undefined,
            viggle_character:undefined,_duration_planning_mode:'duration'}});
      };
      window.configure();window.mount();
    },{catalogue,options});
    const settle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const menu=async (index,label)=>{
      const card=page.locator(`[data-feed-index="${index}"]`);
      const toggle=card.getByTitle('More clip actions');
      if(await toggle.getAttribute('aria-expanded')!=='true') {
        await page.keyboard.press('Escape');
        await toggle.click();
      }
      return card.getByRole('menuitem',{name:label,exact:true});
    };
    const send=async(index,label)=>{
      const button=await menu(index,label);
      await button.click();
      try {await button.locator('svg.text-accent-green').waitFor({timeout:10000});}
      catch(error) {
        console.error('Failed destination',label,await page.evaluate(()=>({
          targets:window.galleryInputs.getState().targets.map(({id,label,disabledReason})=>({id,label,disabledReason})),
          alerts:Array.from(document.querySelectorAll('[role="alert"]')).map(e=>e.textContent),
          mode:window.store.getState().generationMode,workflow:window.store.getState().studioVideoWorkflow,
          model:window.store.getState().params.model_type,video:window.store.getState().params.video_guide,
        })));
        throw error;
      }
    };
    const configure=async(...args)=>{await page.evaluate(args=>window.configure(...args),args);await settle();};
    const state=fn=>page.evaluate(fn);

    await send(0,'Use as reference image');
    assert.deepEqual(await state(()=>({refs:window.store.getState().params.minimax_h3_references.map(r=>r.type),
      start:window.store.getState().startImage,workflow:window.store.getState().studioVideoWorkflow,workspace:window.store.getState().activeWorkspace})),
      {refs:['image'],start:null,workflow:'references',workspace:'Destination'});
    assert.ok(downloads.some(url=>url.includes('same.png?workspace=Origin%20A')),'Fetch the gallery item’s own workspace');
    await send(1,'Use video as reference video');
    assert.deepEqual(await state(()=>window.store.getState().params.minimax_h3_references.map(r=>[r.type,r.include_audio])),[['image',undefined],['video',true]]);
    // Gallery posters deliberately avoid decoding every offscreen video.
    // Start playback before selecting a nonzero frame, as a user would.
    await page.locator('[data-feed-index="1"] video').evaluate(video=>video.play());
    await page.waitForFunction(()=>document.querySelector('[data-feed-index="1"] video').readyState>=2);
    await page.evaluate(()=>{
      const video=document.querySelector('[data-feed-index="1"] video');
      video.pause();
      return new Promise(resolve=>{video.addEventListener('seeked',()=>resolve(),{once:true});video.currentTime=1.1;});
    });
    await send(1,'Use current frame as reference image');
    assert.equal(uploads.at(-1),'clip_t1.10s.png','Capture the selected frame, not frame zero');
    // Exercise offscreen decoding while the preview is still completing a seek.
    await page.evaluate(()=>Object.defineProperty(document.querySelector('[data-feed-index="1"] video'),'seeking',{configurable:true,value:true}));
    await send(1,'Use current frame as reference image');
    assert.equal(uploads.at(-1),'clip_t1.10s.png','A pending seek must keep the selected timestamp');
    await page.evaluate(()=>delete document.querySelector('[data-feed-index="1"] video').seeking);
    await page.screenshot({path:path.join(fixtures,'menu-desktop.png')});
    await page.evaluate(()=>window.store.setState(state=>({modelOptions:{...state.modelOptions,omni_reference_limits:{image:3,video:1,audio:1,total:5}}})));
    assert.equal(await (await menu(0,'Use as reference image')).isDisabled(),true,'Respect reference limits');

    await configure('video','references','minimax_h3_ref2va_fused_turbo','director');
    await send(0,'Use as Director reference image');
    await send(1,'Use video as Director reference video');
    assert.deepEqual(await state(()=>({director:window.store.getState().directorH3References.map(r=>r.type),studio:window.store.getState().params.minimax_h3_references})),
      {director:['image','video'],studio:[]});
    await configure('video','frames','minimax_h3_fused_turbo','director');
    await send(0,'Use as Director reference image');
    assert.equal(await state(()=>window.store.getState().directorReferenceImage.name),'same.png');
    await page.getByRole('button',{name:'Additional references',exact:true}).click();
    await send(0,'Use as Director character reference');
    assert.equal(await state(()=>window.store.getState().directorCharacterRefs[0].name),'same.png');

    await configure('video','frames','minimax_h3_fused_turbo');
    await send(0,'Use as start frame');
    await send(0,'Use as end frame');
    assert.deepEqual(await state(()=>[window.store.getState().startImage?.name,window.store.getState().endImage?.name]),['same.png','same.png']);
    await send(1,'Use video as control video');
    assert.equal(await state(()=>window.store.getState().params.video_guide),'/uploads/clip.mp4');
    assert.equal(await state(()=>window.store.getState().params.audio_prompt_type),'K');
    await configure('video','frames','ltx2_22B_distilled_1_1');
    await send(1,'Use video as control video');
    assert.equal(await state(()=>window.galleryInputs.getState().targets.filter(t=>t.kind==='video' && t.label==='control video').length),1);

    await configure('video','animate','viggle_animate');
    await send(1,'Use video as Animate control video');
    await send(0,'Use as edited frame');
    assert.deepEqual(await state(()=>[window.store.getState().params.video_guide,window.store.getState().params._viggle_edited_frame,
      window.store.getState().studioVideoWorkflow]),['/uploads/clip.mp4','/uploads/same.png','animate']);
    await page.evaluate(()=>window.store.setState(state=>({params:{...state.params,viggle_character:{character_name:'Test'}}})));
    await send(0,'Use as character image');
    assert.equal(await state(()=>window.store.getState().params.viggle_character.reference_path),'/uploads/same.png');

    await configure('image','generate','qwen_image_21_7B');
    await send(0,'Use as reference image');
    assert.equal(await state(()=>window.store.getState().imageRefs[0].name),'same.png');
    assert.equal(await state(()=>window.store.getState().params.model_type),'qwen_image_21_7B');

    await configure('tools','image','qwen_image_21_7B');
    await send(0,'Use as Upscale source');
    assert.equal(await state(()=>window.store.getState().toolsSourcePath),'/uploads/same.png');
    await configure('tools','video','minimax_h3_fused_turbo');
    await send(1,'Use video as Upscale source');
    assert.equal(await state(()=>window.store.getState().toolsSourcePath),'/uploads/clip.mp4');

    for(const [workflow,label] of [['retake','Retake'],['inpaint','Inpaint'],['restyle','Repaint'],['recast','Recast'],['outpaint','Outpaint'],['edit_anything','Prompt Edit']]) {
      await configure('avatar',workflow,'minimax_h3_fused_turbo');
      await send(1,`Use video as ${label} source`);
      const loaded=await state(()=>({path:window.store.getState().editVideoPath,duration:window.store.getState().editVideoDuration,
        resolution:window.store.getState().editVideoResolution,mode:window.store.getState().editSubMode}));
      assert.equal(loaded.path,'/uploads/clip.mp4');assert.ok(loaded.duration>=2);assert.equal(loaded.resolution,'64x64');assert.equal(loaded.mode,workflow);
      if(workflow==='recast') {
        await send(0,'Use as Character A reference');
        assert.equal(await state(()=>window.store.getState().editRecastMappings[0].refPath),'/uploads/same.png');
      }
      if(workflow==='restyle') {
        await send(0,'Use as Repaint edited frame');
        assert.equal(await state(()=>window.store.getState().editRepaintFramePath),'/uploads/same.png');
      }
    }
    await configure('video','extend','minimax_h3_fused_turbo');
    await send(1,'Use video as Extend source');
    assert.equal(await state(()=>window.store.getState().continueVideo.name),'clip.mp4');
    await configure('video','blend','ltx2_22B_distilled_1_1');
    await send(1,'Use video as Blend clip A');
    await send(0,'Use as Blend clip B');
    assert.deepEqual(await state(()=>[window.store.getState().blendClipA.name,window.store.getState().blendClipB.name]),['clip.mp4','same.png']);

    await configure('video','animate','viggle_animate');
    failUpload=true;
    await (await menu(1,'Use video as Animate control video')).click();
    await page.locator('[data-feed-index="1"]').getByRole('alert').filter({hasText:'Could not add media'}).waitFor();
    assert.equal(await state(()=>window.store.getState().params.video_guide),undefined,'A failed upload cannot claim success or set an input');
    failUpload=false;

    // A removed input can never receive a late gallery fetch.
    const oldId=await state(()=>window.galleryInputs.getState().targets[0].id);
    await configure('image','generate','qwen_image_21_7B');
    assert.match(await page.evaluate(async id=>{
      try {await window.sendToInput(id,new File(['x'],'old.mp4',{type:'video/mp4'}));return 'accepted';}
      catch(error){return error.message;}
    },oldId),/input changed/);
    await page.setViewportSize({width:390,height:844});
    await settle();
    await page.evaluate(()=>window.store.getState().setSidebarOpen(false));
    await menu(0,'Use as reference image');
    await page.screenshot({path:path.join(fixtures,'menu-mobile.png')});
    await send(0,'Use as reference image');
    assert.equal(await state(()=>window.store.getState().sidebarOpen),true,'Sending opens the mobile sidecar');
    assert.equal(await state(()=>window.store.getState().studioImageWorkflow),'generate');
    assert.equal(errors.length,0,errors.join('\n'));
    console.log('Gallery routing passed: Studio/Director references, selected frames, frame/control inputs, Animate, Qwen editing, image/video upscale, six video edits, Extend, Blend, limits, upload failures, stale targets and mobile.');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
