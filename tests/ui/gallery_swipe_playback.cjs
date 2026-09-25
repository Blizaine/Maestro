// Real media playback with controlled network delays. No Maestro server or user data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const esbuild = require(path.join(root, 'ui/node_modules/esbuild'));
const { chromium } = require(process.env.MAESTRO_PLAYWRIGHT || 'playwright');
const fixtures = path.join(root, '.codex-tmp/gallery-swipe-playback');
fs.mkdirSync(fixtures, { recursive: true });

(async () => {
  const clip = path.join(fixtures, 'clip.mp4');
  execFileSync(process.env.MAESTRO_FFMPEG || 'ffmpeg', ['-y', '-v', 'error', '-f', 'lavfi', '-i',
    'color=c=green:s=180x320:r=24:d=3', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-af', 'volume=0.01',
    '-shortest', '-movflags', '+faststart', clip]);
  const bytes = fs.readFileSync(clip);
  const bundle = await esbuild.build({ stdin: { contents: `
    import React, { useState } from 'react';
    import { createRoot } from 'react-dom/client';
    import { GalleryViewer } from './src/components/MainContent/GalleryViewer';
    import { createGalleryViewerSurface } from './src/lib/galleryFullscreen';
    import { setCachedThumbnail } from './src/lib/thumbnailCache';
    const items = ['one', 'two', 'three', 'broken'].map(name => ({
      id:name, name:name+'.mp4', type:'video', url:'/media/'+name+'.mp4',
      size:10000, created_at:0, workspace:'test', favorite:false,
    }));
    function Harness() {
      const [surface, setSurface] = useState(null);
      return <><button onClick={() => setSurface(createGalleryViewerSurface(false))}>Open</button>
        {surface && <GalleryViewer surface={surface} items={items} initialId="one"
          sourceImages={[]} comparisonImages={[]} allowFavorite={false}
          onFavorite={async()=>{}} onClose={() => { setSurface(null); surface.release(); }}/>}</>;
    }
    window.setup = async () => {
      const canvas = document.createElement('canvas'); canvas.width=180; canvas.height=320;
      canvas.getContext('2d').fillStyle='green'; canvas.getContext('2d').fillRect(0,0,180,320);
      for (const item of items) await setCachedThumbnail(item.url, canvas.toDataURL());
      createRoot(document.getElementById('root')).render(<React.StrictMode><Harness/></React.StrictMode>);
    };
  `, resolveDir: path.join(root, 'ui'), loader: 'tsx' }, bundle: true, write: false,
  jsx: 'automatic', define: { 'process.env.NODE_ENV': '"development"' }, logLevel: 'silent' });
  const assets = path.join(root, 'ui/dist/assets');
  const css = fs.readFileSync(path.join(assets, fs.readdirSync(assets).find(name => name.endsWith('.css'))), 'utf8');
  const browser = await chromium.launch({ headless: true, ...(process.platform === 'win32' ? {
    executablePath: process.env.MAESTRO_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  } : {}) });
  try {
    for (const touch of [false, true]) {
      const context = await browser.newContext({ viewport: touch ? { width:390,height:844 } : { width:1280,height:900 },
        hasTouch:touch, isMobile:touch });
      const page = await context.newPage();
      page.setDefaultTimeout(10000);
      const errors = [], held = new Map();
      let blockedName = '';
      page.on('pageerror', error => errors.push(error.message));
      await page.addInitScript(() => {
        window.mediaLoads = [];
        window.frameUrls = []; window.releasedFrameUrls = [];
        const createUrl=URL.createObjectURL.bind(URL), revokeUrl=URL.revokeObjectURL.bind(URL);
        URL.createObjectURL=blob=>{const url=createUrl(blob); window.frameUrls.push(url); return url;};
        URL.revokeObjectURL=url=>{window.releasedFrameUrls.push(url); revokeUrl(url);};
        const nativeLoad = HTMLMediaElement.prototype.load;
        HTMLMediaElement.prototype.load = function () {
          if (this.closest('[role="dialog"]')) window.mediaLoads.push({ src:this.getAttribute('src'), at:performance.now() });
          return nativeLoad.call(this);
        };
      });
      await page.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (url.pathname === '/') return route.fulfill({ contentType:'text/html', body:
          '<meta name="viewport" content="width=device-width,initial-scale=1"><div id="root"></div>' });
        if (url.pathname === '/media/broken.mp4') return route.fulfill({ status:404, body:'Missing test clip' });
        if (url.pathname.startsWith('/media/')) {
          if (url.pathname.endsWith('/'+blockedName+'.mp4')) {
            await new Promise(resolve => {
              const waiters = held.get(blockedName) || []; waiters.push(resolve); held.set(blockedName,waiters);
            });
          }
          const range = /bytes=(\d+)-(\d*)/.exec(route.request().headers().range || '');
          try {
            if (range) {
              const start=Number(range[1]), end=range[2] ? Math.min(Number(range[2]),bytes.length-1) : bytes.length-1;
              return await route.fulfill({ status:206, contentType:'video/mp4', headers:{
                'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${bytes.length}`,
              }, body:bytes.subarray(start,end+1) });
            }
            return await route.fulfill({ contentType:'video/mp4', body:bytes });
          } catch { /* Closing the viewer can cancel an intentionally held request. */ }
        }
        return route.abort();
      });
      const release = name => { blockedName=''; for (const resolve of held.get(name) || []) resolve(); held.delete(name); };
      try {
        await page.goto('http://gallery-swipe.test');
        await page.addStyleTag({ content:css });
        await page.addScriptTag({ content:bundle.outputFiles[0].text });
        await page.evaluate(() => window.setup());
        await page.getByRole('button',{name:'Open',exact:true}).click();
        const dialog = page.getByRole('dialog'), deck = dialog.locator('[data-gallery-swipe-deck]');
        const cdp = touch ? await context.newCDPSession(page) : null;
        const swipe = async (direction, small=false) => {
          const b=await deck.boundingBox(), x=b.x+b.width/2, y=b.y+b.height*(direction>0?.72:.28);
          const end=y-direction*(small?25:b.height*.48);
          if (cdp) {
            await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});
            await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:end}]});
            await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
          } else {
            await page.mouse.move(x,y); await page.mouse.down(); await page.mouse.move(x,end,{steps:4}); await page.mouse.up();
          }
        };
        const playing = name => page.waitForFunction(name => {
          const video=document.querySelector('[role="dialog"] video');
          return video?.getAttribute('src')?.includes('/'+name+'.mp4') && video.readyState>=2 && !video.paused
            && document.getElementById('gallery-viewer-description')?.textContent.startsWith(name+'.mp4');
        },name);
        await playing('one');
        await page.evaluate(() => { window.player=document.querySelector('[role="dialog"] video'); window.mediaLoads=[]; });
        await swipe(1,true);
        assert.equal(await page.evaluate(() => window.player.getAttribute('src')), '/media/one.mp4', 'Short canceled drag does not start another source');
        await page.waitForFunction(() => {
          const style=getComputedStyle(document.querySelector('[data-gallery-swipe-current]'));
          return Math.abs(new DOMMatrix(style.transform).m42)<1 && parseFloat(style.transitionDuration)===0;
        });
        blockedName='two';
        await swipe(1);
        await page.waitForFunction(() => window.player.getAttribute('src')==='/media/two.mp4',null,{timeout:220});
        assert.ok(await dialog.locator('#gallery-viewer-description').textContent().then(text=>text.startsWith('one.mp4')),
          'Destination starts loading while the departing item is still active');
        assert.equal(await dialog.locator('video').count(),1,'Staging retains one sound-authorized video element');
        await page.waitForFunction(() => Math.abs(new DOMMatrix(getComputedStyle(document.querySelector('[data-gallery-swipe-current]')).transform).m42)>=window.innerHeight-1);
        assert.ok(await dialog.locator('[data-gallery-swipe-next] img').isVisible(),'Incoming thumbnail remains visible while the request is delayed');
        await page.screenshot({path:path.join(fixtures,`${touch?'phone':'desktop'}-waiting.png`)});
        release('two');
        await playing('two');
        assert.equal(await page.evaluate(() => window.player===document.querySelector('[role="dialog"] video')),true);
        assert.equal(await page.evaluate(() => window.mediaLoads.filter(item=>item.src==='/media/two.mp4').length),1,
          'Prepared video is not cleared or loaded again when navigation commits');
        assert.equal(await page.evaluate(() => window.player.muted),false,'Sound choice survives staged playback');
        await page.evaluate(() => {
          window.stagedFrames=[]; window.sampleStages=true;
          const sample=()=>{
            if(!window.sampleStages)return;
            const video=document.querySelector('[role="dialog"] video');
            const label=video?.getAttribute('aria-label'), source=video?.getAttribute('src');
            if(label && source && !source.endsWith('/'+label) && video.readyState>=2) {
              const frame=document.querySelector('[data-gallery-video-frozen-frame]');
              const drawn=frame instanceof HTMLCanvasElement
                ? frame.getContext('2d').getImageData(Math.floor(frame.width/2),Math.floor(frame.height/2),1,1).data[3]>0
                : frame?.naturalWidth>0;
              window.stagedFrames.push({covered:Boolean(frame && getComputedStyle(frame).visibility==='visible' && drawn),
                cover:frame?.getAttribute('src')});
            }
            requestAnimationFrame(sample);
          };
          requestAnimationFrame(sample);
        });
        await swipe(-1); await playing('one');
        const stagedFrames=await page.evaluate(() => { window.sampleStages=false; return window.stagedFrames; });
        assert.ok(stagedFrames.length>0,'Fast destination decodes before the slide finishes');
        assert.ok(stagedFrames.every(frame=>frame.covered),'Departing clip retains its frozen frame even when the next clip decodes early');
        await swipe(1); await playing('two');

        // Resize cancels a staged target. Its delayed completion must not replace the restored clip.
        blockedName='three';
        await swipe(1);
        await page.waitForFunction(() => window.player.getAttribute('src')==='/media/three.mp4');
        await page.setViewportSize(touch ? {width:390,height:790} : {width:1280,height:850});
        await playing('two'); release('three');
        await page.evaluate(() => new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
        await playing('two');

        // Close during another pending prepare, then permit its response to arrive.
        blockedName='three';
        await swipe(1);
        await page.waitForFunction(() => window.player.getAttribute('src')==='/media/three.mp4');
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        release('three');
        await page.waitForFunction(() => window.player.paused && !window.player.hasAttribute('src'));
        assert.equal(await page.locator('[role="dialog"]').count(),0);

        await page.getByRole('button',{name:'Open',exact:true}).click(); await playing('one');
        await swipe(1); await playing('two'); await swipe(1); await playing('three'); await swipe(1);
        await dialog.getByText('Video could not be loaded.',{exact:true}).waitFor();
        await swipe(-1); await playing('three');
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();

        // A server that never responds must not trap the feed on its preview.
        await page.getByRole('button',{name:'Open',exact:true}).click(); await playing('one');
        blockedName='two'; await swipe(1);
        await page.waitForFunction(() => document.getElementById('gallery-viewer-description')?.textContent.startsWith('two.mp4'),
          null,{timeout:5000});
        await swipe(-1); await playing('one'); release('two');
        await dialog.getByRole('button',{name:'Close viewer',exact:true}).click();
        await page.waitForFunction(() => window.frameUrls.every(url=>window.releasedFrameUrls.includes(url)));
        assert.deepEqual(errors,[],'No asynchronous handoff errors');
        if(cdp) await cdp.detach();
        console.log(`PASS staged gallery playback: ${touch?'phone':'desktop'}`);
      } catch(error) {
        console.error(touch?'phone':'desktop',await page.evaluate(()=>{
          const video=document.querySelector('[role="dialog"] video');
          const panel=document.querySelector('[data-gallery-swipe-current]');
          return {src:video?.getAttribute('src'),ready:video?.readyState,paused:video?.paused,
            label:video?.getAttribute('aria-label'),description:document.getElementById('gallery-viewer-description')?.textContent,
            transform:panel && getComputedStyle(panel).transform,loads:window.mediaLoads};
        }));
        await page.screenshot({path:path.join(fixtures,`${touch?'phone':'desktop'}-failure.png`)});
        throw error;
      } finally {
        for(const name of held.keys()) release(name);
        await context.close();
      }
    }
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode=1; });
