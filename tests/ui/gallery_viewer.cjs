// Real App/gallery/sidecar in a fully intercepted browser. No user data or GPU jobs.
// Build ui first, then: node tests/ui/gallery_viewer.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const {chromium} = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const fixtures = path.join(root, '.codex-tmp/gallery-viewer');
fs.mkdirSync(fixtures, {recursive:true});
const svg = color => `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="${color}"/><circle cx="320" cy="240" r="105" fill="white" opacity=".18"/></svg>`;
const source = svg('#2563eb');
const output = (name, type, workspace='A') => ({name, type, workspace, mode:type,
  id:`${workspace}/${name}`, url:`/api/v1/file/${name}?workspace=${workspace}`,
  size:1024, created_at:1, favorite:false, metadata_ready:true});
const files = [output('result.png','image'), output('clip.mp4','video'), output('result.png','image','B')];

(async () => {
  const videoPath = path.join(fixtures, 'clip.mp4');
  execFileSync(process.env.MAESTRO_FFMPEG || 'ffmpeg', ['-y','-v','error','-f','lavfi','-i',
    'color=c=green:s=180x320:r=10:d=3','-f','lavfi','-i','sine=frequency=440:duration=3',
    '-vf','drawbox=x=0:y=0:w=iw:h=20:color=blue:t=fill,drawbox=x=0:y=ih-20:w=iw:h=20:color=red:t=fill',
    '-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-c:a','aac','-b:a','32k','-af','volume=0.01','-shortest','-movflags','+faststart',videoPath]);
  const video = fs.readFileSync(videoPath);
  const bundle = await esbuild.build({stdin:{contents:`
    import React from 'react'; import {createRoot} from 'react-dom/client';
    import App from './src/App'; import {useStore} from './src/stores/useStore';
    import {useGalleryInputs,captureGallerySourceImages} from './src/lib/galleryInputs';
    import {getVideoPosterUrl,requestThumbnail} from './src/lib/thumbnailCache';
    const idle=async()=>{}; window.store=useStore; window.galleryInputs=useGalleryInputs;
    window.captureSources=captureGallerySourceImages;
    window.posterUrl=getVideoPosterUrl; window.requestThumbnail=requestThumbnail;
    window.setup=(files,source)=>{
      const before=new File([source],'source.svg',{type:'image/svg+xml'});
      useStore.setState({loadModels:idle,loadOutputs:idle,loadWorkspaces:idle,loadSystemConfig:idle,
        loadServicesConfig:idle,loadLlmStatus:idle,loadLlmModels:idle,loadPipelineList:idle,
        reconnectJobs:idle,loadSystemStats:idle,loadSystemDetect:idle,
        generationMode:'image',studioImageWorkflow:'generate',sidebarMode:'studio',sidebarOpen:false,
        models:[],families:[],modelOptions:null,imageRefs:[before],startImage:null,endImage:null,
        outputs:files,outputsTotal:files.length,selectedOutput:0,activeWorkspace:'A',browsingAllFolders:true,
        browsingUploads:false,mediaFilter:'all',jobs:[],pipelineId:null,pipelineStatus:null,
        isGenerating:false,isEnhancing:false,settingsOpen:false,servicesConfig:{},
        params:{...useStore.getState().params,model_type:'qwen_image_21_7B',image_mode:1,prompt:'A test image'}});
      createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
    };`,resolveDir:path.join(root,'ui'),loader:'tsx'},bundle:true,write:false,
    jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
  const assets=path.join(root,'ui/dist/assets');
  const css=fs.readFileSync(path.join(assets,fs.readdirSync(assets).find(name=>name.endsWith('.css'))),'utf8');
  const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{
    executablePath:process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
  const failures=[];
  try {
    for(const scenario of [{name:'desktop',width:1280,height:900,touch:false},
      {name:'phone',width:390,height:844,touch:true}]) {
      const context=await browser.newContext({viewport:{width:scenario.width,height:scenario.height},
        hasTouch:scenario.touch,isMobile:scenario.touch,...(scenario.touch ? {
          userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 Version/26.0 Mobile/15E148 Safari/604.1',
        } : {})});
      const page=await context.newPage();
      page.setDefaultTimeout(10000);
      const errors=[],mutations=[],mediaRequests=[],posterRequests=[]; let failFavorite=false;
      const favorites=new Map();
      page.on('pageerror',error=>errors.push(error.message));
      await page.addInitScript(touch=>{
        localStorage.setItem('maestro_welcome_seen_v1','1');
        window.createdUrls=[];window.revokedUrls=[];
        const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
        URL.createObjectURL=blob=>{const url=create(blob);window.createdUrls.push(url);return url};
        URL.revokeObjectURL=url=>{window.revokedUrls.push(url);revoke(url)};
        const nativePlay=HTMLMediaElement.prototype.play;
        window.dialogPlayCalls=[];
        const soundAuthorized=new WeakSet();
        // Emulate WebKit's sound grant belonging to a media element, not the
        // whole page. A new/remounted player must ask for sound again.
        document.addEventListener('click',event=>{
          const button=event.target.closest?.('button'),label=button?.getAttribute('aria-label');
          if(!event.isTrusted || !['Tap for sound','Unmute video','Play video'].includes(label))return;
          const media=button.closest('[role="dialog"]')?.querySelector('video');
          if(media)soundAuthorized.add(media);
        },true);
        HTMLMediaElement.prototype.play=function(){
          if(this.closest('[role="dialog"]')) {
            window.dialogPlayCalls.push({muted:this.muted});
            if(window.playPolicy==='all' || (window.playPolicy==='unmuted' && !this.muted)
              || (window.playPolicy==='per-element' && !this.muted && !soundAuthorized.has(this))) {
              this.pause();
              return Promise.reject(new DOMException('Playback requires a gesture','NotAllowedError'));
            }
          }
          return nativePlay.call(this);
        };
        window.fullscreenRequests=[];
        const nativeRequest=Element.prototype.requestFullscreen;
        if(touch) {
          // Mobile emulation normally gives vh/dvh/the visual viewport the same
          // height. Model Safari's visible region independently of its layout.
          const nativeViewport=window.visualViewport, viewport=new EventTarget();
          let bounds=null;
          for(const property of ['height','width','offsetTop','offsetLeft','scale']) {
            Object.defineProperty(viewport,property,{get:()=>bounds?.[property] ?? nativeViewport[property]});
          }
          Object.defineProperty(window,'visualViewport',{value:viewport,configurable:true});
          for(const name of ['resize','scroll']) nativeViewport.addEventListener(name,()=>viewport.dispatchEvent(new Event(name)));
          window.setVisualViewport=(next,event='resize')=>{bounds=next;viewport.dispatchEvent(new Event(event))};
          Object.defineProperty(Element.prototype,'requestFullscreen',{value:undefined,configurable:true});
          Object.defineProperty(HTMLElement.prototype,'webkitRequestFullscreen',{value:undefined,configurable:true});
          Object.defineProperty(document,'fullscreenEnabled',{value:false,configurable:true});
        } else if(nativeRequest) {
          Element.prototype.requestFullscreen=function(options){
            window.fullscreenRequests.push({activated:navigator.userActivation.isActive,connected:this.isConnected,
              host:this.hasAttribute('data-gallery-viewer-portal')});
            return window.rejectFullscreen ? Promise.reject(new Error('Unavailable')) : nativeRequest.call(this,options);
          };
        }
      },scenario.touch);
      await page.route('**/*', async route=>{
        const request=route.request(),url=new URL(request.url()),key=url.pathname;
        const json=body=>route.fulfill({json:body});
        if(key==='/') return route.fulfill({contentType:'text/html',body:
          '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root" style="height:100dvh"></div>'});
        if(key.startsWith('/api/v1/favorites/')) {
          mutations.push(url.pathname+url.search);
          if(failFavorite) return route.fulfill({status:500,json:{detail:'Favorite unavailable'}});
          const id=url.pathname+url.search, favorite=!favorites.get(id);favorites.set(id,favorite);
          return json({name:'result.png',favorite});
        }
        if(key.startsWith('/api/v1/thumbnail/')) {
          posterRequests.push(url.pathname+url.search);
          return route.fulfill({contentType:'image/svg+xml',body:svg('#008000')});
        }
        if(key.startsWith('/api/v1/file/') || key.startsWith('/api/v1/uploads/')) {
          if(key.endsWith('.mp4')) {
            mediaRequests.push(url.pathname+url.search);
            const range=/bytes=(\d+)-(\d*)/.exec(request.headers().range || '');
            if(range) {
              const start=Number(range[1]),end=range[2]?Math.min(Number(range[2]),video.length-1):video.length-1;
              return route.fulfill({status:206,contentType:'video/mp4',headers:{'Accept-Ranges':'bytes',
                'Content-Range':`bytes ${start}-${end}/${video.length}`},body:video.subarray(start,end+1)});
            }
            return route.fulfill({contentType:'video/mp4',body:video});
          }
          return route.fulfill({contentType:'image/svg+xml',body:svg(url.searchParams.get('workspace')==='B'?'#22c55e':'#f97316')});
        }
        if(key.endsWith('/metadata')) return json({params:{prompt:'Gallery fixture',seed:42}});
        if(key==='/api/v1/media-flow/capabilities') return json({neural_rendering:{available:false},frame_generation:{available:false,factors:[]}});
        if(key==='/api/v1/system-stats') return json({gpu:{available:false},ram:{used_gb:1,total_gb:32,percent:3},cpu:{percent:1}});
        if(!['GET','HEAD'].includes(request.method()) && !['/api/v1/studio-preferences','/api/v1/loras/check-updates'].includes(key)) {
          mutations.push(`${request.method()} ${key}`);
        }
        return json({checks:[],downloads:[],jobs:[],items:[],entries:[],queue:[],pipelines:[],recipes:[],
          loras:[],models:[],families:[],presets:[],capabilities:{},workspaces:[],characters:[],status:'idle'});
      });
      try {
        await page.goto('http://gallery-viewer.test');
        await page.addStyleTag({content:css});
        await page.addScriptTag({content:bundle.outputFiles[0].text});
        await page.evaluate(({files,source})=>window.setup(files,source),{files,source});
        const image=page.getByRole('button',{name:'Enlarge result.png'}).first();
        await image.waitFor();
        const initialVideo=page.locator('video[data-gallery-media]').first();
        assert.equal(await initialVideo.getAttribute('poster'),'/api/v1/thumbnail/clip.mp4?workspace=A');
        assert.equal(await initialVideo.getAttribute('preload'),'none');
        await page.evaluate(async()=>{
          const poster=new Image(); poster.src=document.querySelector('video[data-gallery-media]').poster;
          await poster.decode();
        });
        assert.ok(posterRequests.length>0,'Gallery poster images load without video playback');
        assert.equal(mediaRequests.length,0,'Browsing posters does not download or decode whole videos');
        assert.equal(await page.evaluate(()=>window.posterUrl('/api/v1/file/a%20%23b.mp4?workspace=A%20B&v=2#t=0.1')),
          '/api/v1/thumbnail/a%20%23b.mp4?workspace=A%20B&v=2','Poster keeps encoded names and workspace without a video seek fragment');
        assert.equal(await page.evaluate(()=>window.posterUrl('https://elsewhere.test/api/v1/file/clip.mp4')),null,
          'External URLs are not sent to the local poster endpoint');
        assert.equal(await page.evaluate(()=>window.requestThumbnail('/api/v1/file/clip.mp4?workspace=__uploads__','clip.mp4')),
          '/api/v1/thumbnail/clip.mp4?workspace=__uploads__','Uploads also use image posters');
        await page.waitForFunction(()=>window.galleryInputs.getState().targets.some(t=>t.getImages?.().length));
        await image.click();
        const dialog=page.getByRole('dialog');
        const activeMedia=dialog.locator('[data-gallery-swipe-current]');
        const navigate=async(direction)=>{
          await dialog.getByRole('button',{name:'Close viewer',exact:true}).focus();
          await page.keyboard.press(direction>0?'ArrowDown':'ArrowUp');
        };
        await dialog.waitFor();
        assert.deepEqual(await dialog.evaluate(el=>({width:Math.round(el.getBoundingClientRect().width),
          height:Math.round(el.getBoundingClientRect().height)})),{width:scenario.width,height:scenario.height});
        assert.equal(await page.locator('#root').evaluate(el=>el.inert),true,'Background is inert');
        assert.equal(await dialog.locator('header,footer').count(),0,'No bars reserve space above or below the media');
        assert.equal(await dialog.getByRole('button',{name:/^(Enter|Exit) fullscreen$/}).count(),0);
        assert.equal(await dialog.locator('[data-gallery-viewer-actions]').evaluate(el=>getComputedStyle(el).position),'absolute','Heart and close overlay the media');
        if(!scenario.touch) {
          await page.waitForFunction(()=>document.fullscreenElement?.hasAttribute('data-gallery-viewer-portal'));
          assert.deepEqual(await page.evaluate(()=>window.fullscreenRequests[0]),{activated:true,connected:true,host:true},'Native fullscreen is requested inside the opening click');
        } else {
          await dialog.getByText(/without the Safari toolbar/).waitFor();
          await dialog.getByRole('button',{name:'Dismiss fullscreen help',exact:true}).click();
        }
        // Image zoom must own multi-touch without paging or scaling the UI.
        const zoomImage=activeMedia.locator('[data-gallery-image-zoom] img');
        await page.waitForFunction(()=>document.querySelector('[data-gallery-image-zoom] img')?.naturalWidth>0);
        const normalImage=await zoomImage.boundingBox();
        const normalActions=await dialog.locator('[data-gallery-viewer-actions]').boundingBox();
        const expectImageFit=async()=>{
          await page.waitForFunction(width=>Math.abs(document.querySelector('[data-gallery-image-zoom] img')?.getBoundingClientRect().width-width)<1,normalImage.width);
          assert.equal(await dialog.getByRole('button',{name:'Reset zoom',exact:true}).count(),0);
        };
        if(scenario.touch) {
          const zoomCdp=await context.newCDPSession(page);
          const area=await activeMedia.boundingBox(),cx=area.x+area.width/2,cy=area.y+area.height/2;
          const touch=async(type,points)=>zoomCdp.send('Input.dispatchTouchEvent',{type,touchPoints:points});
          const pair=distance=>[{id:1,x:cx-distance,y:cy},{id:2,x:cx+distance,y:cy}];
          await touch('touchStart',pair(40));
          await touch('touchMove',pair(140));
          await touch('touchEnd',[]);
          await dialog.getByRole('button',{name:'Reset zoom',exact:true}).waitFor();
          let zoomed=await zoomImage.boundingBox();
          assert.ok(zoomed.width>normalImage.width*2,'Two fingers enlarge the image');
          assert.deepEqual(await dialog.locator('[data-gallery-viewer-actions]').boundingBox(),normalActions,'Heart/close remain unscaled');
          assert.equal(await page.evaluate(()=>window.visualViewport.scale),1,'Pinching changes the image, not the browser page');
          await touch('touchStart',[{id:1,x:cx,y:cy}]);
          await touch('touchMove',[{id:1,x:cx+120,y:cy+90}]);
          await touch('touchEnd',[]);
          const panned=await zoomImage.boundingBox();
          assert.ok(panned.x>zoomed.x+50,'One finger pans the enlarged image');
          assert.ok(await zoomImage.getAttribute('src').then(src=>src.includes('workspace=A')),'Panning does not change the gallery item');
          await page.screenshot({path:path.join(fixtures,'phone-image-zoom.png')});
          // Pinch back to fit, then start a swipe before adding the other finger.
          await touch('touchStart',pair(140));
          await touch('touchMove',pair(25));
          await touch('touchEnd',[]);
          await expectImageFit();
          await touch('touchStart',[{id:1,x:cx-40,y:cy}]);
          await touch('touchMove',[{id:1,x:cx-40,y:cy-35}]);
          await page.waitForFunction(()=>new DOMMatrixReadOnly(getComputedStyle(document.querySelector('[data-gallery-swipe-current]')).transform).m42 < -20);
          await touch('touchStart',[{id:1,x:cx-40,y:cy-35},{id:2,x:cx+40,y:cy+35}]);
          await touch('touchMove',[{id:1,x:cx-140,y:cy-100},{id:2,x:cx+140,y:cy+100}]);
          await touch('touchEnd',[]);
          await dialog.getByRole('button',{name:'Reset zoom',exact:true}).waitFor();
          assert.ok(await zoomImage.getAttribute('src').then(src=>src.includes('workspace=A')),'Adding a second finger cancels the captured swipe');
          await page.waitForFunction(()=>Math.abs(new DOMMatrixReadOnly(getComputedStyle(document.querySelector('[data-gallery-swipe-current]')).transform).m42)<1);
          await dialog.getByRole('button',{name:'Reset zoom',exact:true}).click();
          await expectImageFit();
          // The next ordinary swipe works once the image is back at 1x.
          await touch('touchStart',[{id:1,x:cx,y:area.y+area.height*.7}]);
          await touch('touchMove',[{id:1,x:cx,y:area.y+area.height*.3}]);
          await touch('touchEnd',[]);
          await dialog.locator('video').waitFor();
          await navigate(-1);
          await zoomImage.waitFor();
          await expectImageFit();
          await zoomCdp.detach();
        } else {
          await zoomImage.dblclick();
          await dialog.getByRole('button',{name:'Reset zoom',exact:true}).waitFor();
          assert.ok((await zoomImage.boundingBox()).width>normalImage.width*1.5,'Desktop double click enlarges the image');
          await dialog.getByRole('button',{name:'Reset zoom',exact:true}).click();
          await expectImageFit();
          await zoomImage.dblclick();
          await dialog.getByRole('button',{name:'Reset zoom',exact:true}).waitFor();
          await navigate(1);
          await dialog.locator('video').waitFor();
          await navigate(-1);
          await zoomImage.waitFor();
          await expectImageFit();
        }
        // The rest uses the public accessible controls, not component state.
        await dialog.getByRole('button',{name:'Compare images',exact:true}).click();
        const slider=dialog.getByRole('slider');
        await slider.waitFor();
        await slider.focus();
        await page.keyboard.press('Home');
        assert.equal(await slider.inputValue(),'0','Left reveals the entire new image');
        await page.keyboard.press('End');
        assert.equal(await slider.inputValue(),'100','Right reveals the source image');
        await page.keyboard.press('ArrowLeft');
        assert.equal(await dialog.locator('video[data-gallery-video-active="true"]').count(),0,'Slider arrows do not navigate the gallery');
        assert.equal(await dialog.getByText('No images or videos to show.',{exact:true}).count(),0);
        const before=dialog.getByRole('combobox',{name:'Before comparison image'});
        assert.ok((await before.inputValue()).startsWith('source:'),'Defaults to the active sidecar source');
        await before.selectOption({label:'B / result.png'});
        assert.ok((await before.inputValue()).includes('B/result.png'),'Can compare another gallery image');
        const privateImage=Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="480"><rect width="160" height="480" fill="#a855f7"/></svg>');
        await dialog.locator('input[aria-label="Choose Before image from this device"]').setInputFiles({name:'private.svg',mimeType:'image/svg+xml',buffer:privateImage});
        assert.ok((await before.inputValue()).startsWith('device:'),'Can compare a private local image');
        await page.waitForFunction(()=>document.querySelector('img[alt="Before: private.svg"]')?.naturalWidth===160);
        const stage=dialog.getByRole('group',{name:'Before and after image comparison'});
        await slider.focus();await page.keyboard.press('Home');
        await page.keyboard.press('ArrowRight');
        const bounds=await stage.boundingBox();
        await page.mouse.move(bounds.x+bounds.width*.01,bounds.y+bounds.height*.5);
        await page.mouse.down();
        await page.mouse.move(bounds.x+bounds.width*.55,bounds.y+bounds.height*.5,{steps:6});
        await page.mouse.up();
        assert.ok(Number(await slider.inputValue())>50,'Dragging the divider reveals more Before');
        await slider.focus();await page.keyboard.press('End');
        assert.equal(await dialog.locator('img[alt="Before: private.svg"]').evaluate(el=>getComputedStyle(el.parentElement).backgroundColor),'rgb(0, 0, 0)','Opaque Before prevents After leaking into letterboxes');
        await page.keyboard.press('Home');
        for(let i=0;i<12;i++) await page.keyboard.press('Tab');
        assert.equal(await dialog.evaluate(el=>el.contains(document.activeElement)),true,'Tab remains in the viewer');
        await slider.fill('50');
        await page.screenshot({path:path.join(fixtures,scenario.name+'-compare.png')});
        // Comparison selections survive navigating through video items.
        await navigate(1);
        await dialog.locator('video').waitFor();
        await navigate(-1);
        assert.ok((await before.inputValue()).startsWith('device:'));
        await page.keyboard.press('Escape');
        await dialog.waitFor({state:'hidden'});
        await page.waitForFunction(()=>!document.fullscreenElement && !document.webkitFullscreenElement);
        assert.equal(await page.locator('[data-gallery-viewer-portal]').count(),0,'Closing releases the fullscreen surface');
        assert.equal(await page.locator('#root').evaluate(el=>el.inert),false);
        assert.equal(await page.evaluate(()=>document.activeElement?.getAttribute('aria-label')),'Enlarge result.png');
        assert.ok(await page.evaluate(()=>window.revokedUrls.length>0),'Temporary comparison URLs are released');
        assert.equal(await page.evaluate(()=>window.store.getState().imageRefs[0].name),'source.svg','Comparison leaves generation source intact');

        await image.click();
        await dialog.waitFor();
        if(scenario.touch) assert.equal(await dialog.getByRole('button',{name:'Dismiss fullscreen help',exact:true}).count(),0,'Dismissed Home Screen help stays dismissed when reopening');
        const favorite=dialog.getByRole('button',{name:'Add to favorites',exact:true});
        await favorite.click();
        await dialog.getByRole('button',{name:'Remove from favorites',exact:true}).waitFor();
        assert.equal(await page.evaluate(()=>window.store.getState().outputs[0].favorite),true);
        assert.ok(mutations[0].includes('workspace=A'),'Favorite uses the media folder');
        failFavorite=true;
        await dialog.getByRole('button',{name:'Remove from favorites',exact:true}).click();
        await dialog.getByRole('alert').waitFor();
        assert.equal(await page.evaluate(()=>window.store.getState().outputs[0].favorite),true,'Failed save retains favorite');
        failFavorite=false;
        await navigate(1);
        await dialog.locator('video').waitFor();
        await page.waitForFunction(()=>document.querySelector('[role="dialog"] video')?.readyState>=2);
        await page.waitForFunction(()=>!document.querySelector('[role="dialog"] video')?.paused);
        assert.equal(await dialog.locator('video').evaluate(video=>video.controls),false,'Native controls cannot dim a newly opened video');
        assert.equal(await dialog.getByRole('slider',{name:'Video position',exact:true}).count(),0,'Playback controls start hidden');
        assert.equal(await dialog.locator('[data-gallery-viewer-actions] button').count(),2,'Video has just favorite and close actions');
        assert.equal(await dialog.locator('[data-gallery-swipe-deck]').evaluate(el=>Math.round(el.getBoundingClientRect().height)),scenario.height,'Media gets the full viewer height');
        const checkVisibleBounds=async(controls=false)=>{
          const result=await dialog.evaluate((el,controls)=>{
            const viewport=window.visualViewport, box=el.getBoundingClientRect();
            const rect=node=>{const b=node.getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height}};
            const targets=[el,el.querySelector('[data-gallery-swipe-deck]'),el.querySelector('video')];
            const widgets=controls ? [...el.querySelectorAll('[data-gallery-playback-controls] button,[data-gallery-playback-controls] input')] : [];
            return {
              viewport:{x:viewport.offsetLeft,y:viewport.offsetTop,width:viewport.width,height:viewport.height},
              media:targets.map(rect),
              controls:widgets.map(node=>{
                const b=node.getBoundingClientRect();
                return {inside:b.top>=box.top && b.bottom<=box.bottom && b.left>=box.left && b.right<=box.right,
                  hittable:node.contains(document.elementFromPoint(b.x+b.width/2,b.y+b.height/2))};
              }),
              fit:getComputedStyle(el.querySelector('video')).objectFit,
            };
          },controls);
          for(const box of result.media) for(const key of ['x','y','width','height']) {
            assert.ok(Math.abs(box[key]-result.viewport[key])<1,`Media ${key} fits visible viewport: ${JSON.stringify(result)}`);
          }
          assert.equal(result.fit,'contain','The entire frame stays visible');
          assert.ok(result.controls.every(control=>control.inside && control.hittable),'Playback controls are inside the visible area and directly clickable');
        };
        if(scenario.touch) await page.evaluate(()=>window.setVisualViewport({height:640,offsetTop:30}));
        await checkVisibleBounds();
        await page.evaluate(()=>window.viewerVideo=document.querySelector('[role="dialog"] video'));
        const deck=dialog.locator('[data-gallery-swipe-deck]');
        const cdp=scenario.touch ? await context.newCDPSession(page) : null;
        const dragStart=async(x,y)=>{
          if(cdp) await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});
          else {await page.mouse.move(x,y);await page.mouse.down()}
        };
        const dragMove=async(x,y)=>{
          if(cdp) await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y}]});
          else await page.mouse.move(x,y,{steps:4});
        };
        const dragEnd=async()=>{
          if(cdp) await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
          else await page.mouse.up();
        };
        const waitOffset=async(value)=>page.waitForFunction(value=>{
          const panel=document.querySelector('[data-gallery-swipe-current]');
          return panel && Math.abs(new DOMMatrixReadOnly(getComputedStyle(panel).transform).m42-value)<1
            && (value!==0 || panel.style.transition==='none');
        },value);
        let deckBounds=await deck.boundingBox();
        let x=deckBounds.x+deckBounds.width/2,y=deckBounds.y+deckBounds.height*.65;
        // A small drag follows the pointer, but releases back to the same video.
        await dragStart(x,y);await dragMove(x,y-25);await waitOffset(-25);
        await dragEnd();await waitOffset(0);
        assert.equal(await page.evaluate(()=>window.viewerVideo===document.querySelector('[role="dialog"] video')),true);
        assert.equal(await dialog.getByRole('slider',{name:'Video position',exact:true}).count(),0,'A short swipe does not reveal playback controls');
        assert.equal(await dialog.locator('video').evaluate(video=>video.paused),false,'A short swipe does not pause');
        // One stationary tap pauses; the next resumes, without a synthetic click toggling twice.
        await dragStart(x,y);await dragEnd();
        const position=dialog.getByRole('slider',{name:'Video position',exact:true});
        await position.waitFor();
        await page.waitForFunction(()=>document.querySelector('[role="dialog"] video')?.paused);
        await checkVisibleBounds(true);
        if(scenario.touch) {
          // Bars expand/collapse, or scroll the visible area, without moving
          // the controls below the screen or changing playback position.
          await page.evaluate(()=>window.setVisualViewport({height:710,offsetTop:8}));
          await checkVisibleBounds(true);
          await page.evaluate(()=>window.setVisualViewport({height:640,offsetTop:40},'scroll'));
          await checkVisibleBounds(true);
          await page.screenshot({path:path.join(fixtures,'phone-browser-bars-paused.png')});
          await page.setViewportSize({width:844,height:390});
          await page.evaluate(()=>window.setVisualViewport({height:310,offsetTop:15}));
          await checkVisibleBounds(true);
          await page.screenshot({path:path.join(fixtures,'phone-landscape-video-paused.png')});
          await page.setViewportSize({width:390,height:844});
          await page.evaluate(()=>window.setVisualViewport({height:640,offsetTop:40}));
          await checkVisibleBounds(true);
          deckBounds=await deck.boundingBox();x=deckBounds.x+deckBounds.width/2;y=deckBounds.y+deckBounds.height*.65;
        }
        await dragStart(x,y);await dragEnd();
        await page.waitForFunction(()=>!document.querySelector('[role="dialog"] video')?.paused);
        await position.waitFor({state:'hidden'});
        await dragStart(x,y);await dragEnd();
        await position.waitFor();
        await page.waitForFunction(()=>document.querySelector('[role="dialog"] video')?.paused);
        // Playback controls still respond directly, without triggering a swipe/tap.
        await dialog.getByRole('button',{name:'Play video',exact:true}).click();
        await page.waitForFunction(()=>!document.querySelector('[role="dialog"] video')?.paused);
        await dialog.getByRole('button',{name:'Pause video',exact:true}).click();
        await page.waitForFunction(()=>document.querySelector('[role="dialog"] video')?.paused);
        await position.fill('1.5');
        await page.waitForFunction(()=>Math.abs(document.querySelector('[role="dialog"] video').currentTime-1.5)<.1);
        await dialog.getByRole('button',{name:'Mute video',exact:true}).click();
        assert.equal(await dialog.locator('video').evaluate(video=>video.muted),true);
        await dialog.getByRole('button',{name:'Unmute video',exact:true}).click();
        assert.equal(await dialog.locator('video').evaluate(video=>video.muted),false);
        await dialog.getByRole('button',{name:'Play video',exact:true}).click();
        await position.waitFor({state:'hidden'});
        await page.screenshot({path:path.join(fixtures,scenario.name+'-clear-playback.png')});
        if(cdp) {
          await dragStart(x,y);await dragMove(x,y-280);await waitOffset(-280);
          await cdp.send('Input.dispatchTouchEvent',{type:'touchCancel',touchPoints:[]});
          await waitOffset(0);
          assert.equal(await page.evaluate(()=>window.viewerVideo===document.querySelector('[role="dialog"] video')),true,'Cancelled touch stays on the current clip');
        }
        // Keep the video mounted while revealing the next image during an unfinished swipe.
        await dragStart(x,y);await dragMove(x,y-280);await waitOffset(-280);
        const neighbor=await dialog.locator('[data-gallery-swipe-next]').boundingBox();
        assert.ok(neighbor.y<deckBounds.y+deckBounds.height-250,'Next panel enters the viewport before release');
        assert.equal(await page.evaluate(()=>window.viewerVideo===document.querySelector('[role="dialog"] video')),true,'The playing video follows the gesture');
        assert.equal(await dialog.locator('video').count(),1,'Neighbor previews do not start extra video players');
        await page.screenshot({path:path.join(fixtures,scenario.name+'-mid-swipe.png')});
        await dragEnd();
        await activeMedia.locator('img[src*="workspace=B"]').waitFor();
        assert.equal(await page.evaluate(()=>window.viewerVideo.paused),true,'Previous video stops when paging');
        // Swipe down to bring the previous video back, then navigate up again.
        deckBounds=await deck.boundingBox();x=deckBounds.x+deckBounds.width/2;y=deckBounds.y+deckBounds.height*.3;
        await dragStart(x,y);await dragMove(x,y+280);await waitOffset(280);await dragEnd();
        await dialog.locator('video').waitFor();
        assert.equal(await dialog.getByRole('slider',{name:'Video position',exact:true}).count(),0,'Swiping back starts with clear playback');
        if(scenario.touch) {
          y=deckBounds.y+deckBounds.height*.65;
          await dragStart(x,y);await dragMove(x,y-280);await dragEnd();
          await cdp.detach();
        } else {
          await page.mouse.move(x,deckBounds.y+deckBounds.height/2);
          for(let i=0;i<4;i++) await page.mouse.wheel(0,20);
        }
        await activeMedia.locator('img[src*="workspace=B"]').waitFor();
        if(scenario.touch) await page.evaluate(()=>window.setVisualViewport(null));
        await dialog.getByRole('button',{name:'Add to favorites',exact:true}).click();
        await dialog.getByRole('button',{name:'Remove from favorites',exact:true}).waitFor();
        assert.ok(mutations.at(-1).includes('workspace=B'),'Same filenames in different folders remain distinct');
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        await dialog.waitFor({state:'hidden'});
        // Removing a favorite must not unexpectedly advance a Favorites session.
        await page.evaluate(()=>window.store.setState({mediaFilter:'favorites',selectedOutput:0}));
        await page.getByRole('button',{name:'Open full-screen gallery',exact:true}).first().click();
        await dialog.getByRole('button',{name:'Remove from favorites',exact:true}).click();
        await dialog.getByRole('button',{name:'Add to favorites',exact:true}).waitFor();
        assert.ok(await activeMedia.locator('img[src*="workspace=A"]').isVisible(),'Unfavoriting retains current media');
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        await page.evaluate(()=>window.store.setState({mediaFilter:'all',selectedOutput:0}));

        // A text-to-image run has no source; a gallery choice enables comparison.
        await page.evaluate(()=>{window.rejectFullscreen=true;window.store.setState({imageRefs:[]})});
        await page.waitForFunction(()=>window.galleryInputs.getState().targets.every(t=>!t.getImages?.().length));
        await page.getByRole('button',{name:'Open full-screen gallery',exact:true}).first().click();
        await dialog.getByRole('button',{name:'Compare images',exact:true}).click();
        assert.equal(await dialog.getByRole('slider').isDisabled(),true);
        await dialog.getByRole('combobox',{name:'Before comparison image'}).selectOption({label:'B / result.png'});
        assert.equal(await dialog.getByRole('slider').isEnabled(),true);
        await dialog.locator('input[aria-label="Choose After image from this device"]').setInputFiles({name:'after.svg',mimeType:'image/svg+xml',buffer:Buffer.from(source)});
        assert.ok((await dialog.getByRole('combobox',{name:'After comparison image'}).inputValue()).startsWith('device:'));
        // Native fullscreen is optional: rejection leaves a usable full-viewport viewer.
        if(scenario.touch) assert.equal(await dialog.getByText(/without the Safari toolbar/).count(),0,'Fullscreen fallback respects dismissed Home Screen help');
        else await dialog.getByText(/could not enter fullscreen/).waitFor();
        assert.equal(await dialog.isVisible(),true);
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        await page.evaluate(()=>{window.rejectFullscreen=false});

        // Continue the current gallery through paginated results without duplicates.
        await page.evaluate(extra=>{
          window.moreCalls=0;window.store.setState({outputsTotal:4,selectedOutput:0,loadMoreOutputs:async()=>{
            window.moreCalls++;const s=window.store.getState();
            if(!s.outputs.some(file=>file.id===extra.id))window.store.setState({outputs:[...s.outputs,extra]});
          }});
        },output('next.png','image','C'));
        await page.getByRole('button',{name:'Open full-screen gallery',exact:true}).first().click();
        await navigate(1);
        await dialog.locator('video').waitFor();
        await page.waitForFunction(()=>window.moreCalls===1);
        await navigate(1);
        await activeMedia.locator('img[src*="workspace=B"]').waitFor();
        await navigate(1);
        await activeMedia.locator('img[src*="workspace=C"]').waitFor();
        await navigate(1);
        assert.equal(await activeMedia.locator('img[src*="workspace=C"]').isVisible(),true,'Last item does not wrap');
        assert.equal(await page.evaluate(()=>window.moreCalls),1);
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();

        // Opening directly from a video retains its playback position and releases it on close.
        const inlineVideo=page.locator('video[data-gallery-media]').first();
        await inlineVideo.evaluate(video=>{video.currentTime=1.25;video.pause()});
        await inlineVideo.locator('..').getByRole('button',{name:'Open full-screen gallery',exact:true}).click();
        await page.waitForFunction(()=>document.querySelector('[role="dialog"] video')?.currentTime>=1.2);
        await page.evaluate(()=>window.viewerVideo=document.querySelector('[role="dialog"] video'));
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        assert.equal(await page.evaluate(()=>window.viewerVideo.paused && !window.viewerVideo.hasAttribute('src')),true);

        // Browser autoplay restrictions still leave a clear, usable picture.
        await page.evaluate(()=>{window.playPolicy='unmuted';window.dialogPlayCalls=[]});
        await inlineVideo.locator('..').getByRole('button',{name:'Open full-screen gallery',exact:true}).click();
        await dialog.getByRole('button',{name:'Tap for sound',exact:true}).waitFor();
        await page.waitForFunction(()=>{
          const video=document.querySelector('[role="dialog"] video');
          return video && video.muted && !video.paused;
        });
        assert.equal(await dialog.getByRole('slider',{name:'Video position',exact:true}).count(),0);
        assert.ok(await page.evaluate(()=>window.dialogPlayCalls.some(call=>!call.muted) && window.dialogPlayCalls.some(call=>call.muted)),'Unmuted rejection retries muted');
        await page.evaluate(()=>{window.playPolicy=''});
        await dialog.getByRole('button',{name:'Tap for sound',exact:true}).click();
        await page.waitForFunction(()=>!document.querySelector('[role="dialog"] video').muted);
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();

        await page.evaluate(()=>{window.playPolicy='all'});
        await inlineVideo.locator('..').getByRole('button',{name:'Open full-screen gallery',exact:true}).click();
        const blockedPlay=dialog.getByRole('button',{name:'Play video',exact:true});
        await blockedPlay.waitFor();
        assert.equal(await dialog.locator('video').evaluate(video=>video.controls),false);
        assert.equal(await dialog.getByRole('slider',{name:'Video position',exact:true}).count(),0);
        const blockedPlayBounds=await blockedPlay.boundingBox();
        assert.ok(blockedPlayBounds.y>scenario.height*.65 && blockedPlayBounds.height<65,'Blocked playback uses a small bottom action');
        await page.waitForFunction(()=>{
          const video=document.querySelector('[role="dialog"] video');
          return video?.readyState>=2 && video.dataset.galleryVideoReady==='true'
            && !document.querySelector('[data-gallery-video-preview]');
        });
        await page.evaluate(()=>{window.playPolicy=''});
        await blockedPlay.click();
        await page.waitForFunction(()=>!document.querySelector('[role="dialog"] video').paused);
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();

        // Sound permission and deliberate mute persist across swipes, including
        // an intervening image. Use distinct clips and real pointer gestures.
        await page.evaluate(files=>{
          window.playPolicy='per-element';window.dialogPlayCalls=[];
          window.store.setState({outputs:files,outputsTotal:files.length,selectedOutput:0});
        },[output('sound-one.mp4','video'),output('sound-two.mp4','video'),output('between.png','image'),output('sound-three.mp4','video')]);
        await page.getByRole('button',{name:'Open full-screen gallery',exact:true}).first().click();
        await dialog.getByRole('button',{name:'Tap for sound',exact:true}).waitFor();
        await page.evaluate(()=>window.soundVideo=document.querySelector('[role="dialog"] video'));
        await dialog.getByRole('button',{name:'Tap for sound',exact:true}).click();
        await page.waitForFunction(()=>!window.soundVideo.muted && !window.soundVideo.paused);
        const soundCdp=scenario.touch ? await context.newCDPSession(page) : null;
        const swipeSound=async(direction)=>{
          const b=await deck.boundingBox(),x=b.x+b.width/2;
          const from=b.y+b.height*(direction>0 ? .7 : .3),to=b.y+b.height*(direction>0 ? .3 : .7);
          if(soundCdp) {
            await soundCdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y:from}]});
            await soundCdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:to}]});
            await soundCdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
          } else {
            await page.mouse.move(x,from);await page.mouse.down();await page.mouse.move(x,to,{steps:5});await page.mouse.up();
          }
        };
        const checkSound=async(name,muted)=>{
          await page.waitForFunction(({name,muted})=>{
            const video=document.querySelector('[role="dialog"] video');
            return video?.getAttribute('src')?.includes(name) && video.readyState>=2 && !video.paused && video.muted===muted;
          },{name,muted});
          assert.equal(await page.evaluate(()=>window.soundVideo===document.querySelector('[role="dialog"] video')),true,'Keep the sound-authorized media element');
          assert.equal(await dialog.locator('video').count(),1,'Only one video element in the viewer');
          assert.equal(await dialog.getByRole('button',{name:'Tap for sound',exact:true}).count(),0,'No repeated sound prompt');
          assert.equal(await dialog.getByRole('slider',{name:'Video position',exact:true}).count(),0,'New clips start with hidden controls');
        };
        await swipeSound(1);await checkSound('sound-two.mp4',false);
        await dialog.locator('video').click();
        await dialog.getByRole('button',{name:'Mute video',exact:true}).click();
        await swipeSound(1);
        await activeMedia.locator('img[src*="between.png"]').waitFor();
        await page.waitForFunction(()=>window.soundVideo.paused && !window.soundVideo.hasAttribute('src'));
        await swipeSound(1);await checkSound('sound-three.mp4',true);
        await dialog.locator('video').click();
        await dialog.getByRole('button',{name:'Unmute video',exact:true}).click();
        await swipeSound(-1);
        await activeMedia.locator('img[src*="between.png"]').waitFor();
        await swipeSound(-1);await checkSound('sound-two.mp4',false);
        await swipeSound(-1);await checkSound('sound-one.mp4',false);
        if(soundCdp)await soundCdp.detach();
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        assert.equal(await page.evaluate(()=>window.soundVideo.paused && !window.soundVideo.hasAttribute('src')),true,'Closing releases the persistent player');
        await page.evaluate(files=>{
          window.playPolicy='';window.store.setState({outputs:files,outputsTotal:files.length,selectedOutput:0});
        },files);

        if(scenario.touch) {
          await page.evaluate(()=>Object.defineProperty(navigator,'standalone',{value:true,configurable:true}));
          await page.setViewportSize({width:844,height:390});
          await page.getByRole('button',{name:'Open sidecar',exact:true}).click();
          await page.getByRole('button',{name:'Close sidecar',exact:true}).click();
          assert.equal(await page.locator('.maestro-sidebar').getAttribute('aria-hidden'),'true');
          const main=await page.locator('main').boundingBox();
          assert.equal(Math.round(main.width),844,'Collapsed landscape sidecar gives gallery the width');
          await page.getByRole('button',{name:'Open full-screen gallery',exact:true}).first().click();
          await dialog.waitFor();
          assert.equal(await dialog.getByText(/without the Safari toolbar/).count(),0,'Home Screen mode needs no fullscreen help');
          await page.screenshot({path:path.join(fixtures,'phone-landscape-viewer.png')});
          await page.keyboard.press('Escape');
          await page.setViewportSize({width:320,height:568});
          await page.getByRole('button',{name:'Open full-screen gallery',exact:true}).first().click();
          assert.equal(await dialog.evaluate(el=>el.scrollWidth<=el.clientWidth),true,'Viewer fits a small phone');
          await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
          await page.setViewportSize({width:390,height:844});
          assert.equal(await page.locator('.maestro-sidebar').getAttribute('aria-hidden'),'true');
          // A new document loses module state and must still remember dismissal.
          await page.reload();
          await page.addStyleTag({content:css});
          await page.addScriptTag({content:bundle.outputFiles[0].text});
          await page.evaluate(({files,source})=>window.setup(files,source),{files,source});
          await page.getByRole('button',{name:'Enlarge result.png'}).first().click();
          await dialog.waitFor();
          await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
          assert.equal(await dialog.getByText(/without the Safari toolbar/).count(),0,'Home Screen dismissal survives page refresh');
          await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        } else {
          await page.setViewportSize({width:900,height:480});
          assert.equal(await page.getByRole('button',{name:'Open sidecar',exact:true}).count(),0,'Short desktop retains desktop layout');
          assert.equal(await page.locator('.maestro-sidebar').evaluate(el=>getComputedStyle(el).position),'static');
        }
        assert.deepEqual(errors,[],'No browser runtime errors');
        assert.ok(mutations.every(url=>url.includes('/favorites/')),'No jobs, uploads, or other real mutations');
        console.log('PASS gallery viewer: '+scenario.name);
      } catch(error) {
        await page.screenshot({path:path.join(fixtures,scenario.name+'-failure.png')});
        failures.push(scenario.name+': '+error.stack+'\nBrowser errors: '+JSON.stringify(errors));
      } finally {await context.close();}
    }
  } finally {await browser.close();}
  assert.deepEqual(failures,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
