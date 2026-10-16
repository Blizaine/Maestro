const assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
const root=path.resolve(__dirname,'../..');
const esbuild=require(path.join(root,'ui/node_modules/esbuild'));
const {chromium}=require(process.env.MAESTRO_PLAYWRIGHT||'playwright');
(async()=>{
 const bundle=await esbuild.build({absWorkingDir:path.join(root,'ui'),stdin:{contents:`import React from 'react';import {createRoot} from 'react-dom/client';import {useStore} from './src/stores/useStore';import {Flux3PromptHelper} from './src/components/Sidebar/Flux3PromptHelper';window.store=useStore;useStore.setState({params:{...useStore.getState().params,prompt:'A quiet garden.',resolution:'1024x1024'},imageRefs:[]});createRoot(document.getElementById('root')).render(<Flux3PromptHelper/>);`,resolveDir:path.join(root,'ui'),loader:'tsx'},bundle:true,write:false,jsx:'automatic',define:{'process.env.NODE_ENV':'"development"'},logLevel:'silent'});
 const browser=await chromium.launch({headless:true,...(process.platform==='win32'?{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'}:{})});
 try{for(const width of [1280,390]){
 const page=await browser.newPage({viewport:{width,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/**',r=>r.fulfill({json:{}}));await page.setContent('<div id="root"></div>');await page.addStyleTag({content:fs.readFileSync(path.join(root,'ui/dist/assets',fs.readdirSync(path.join(root,'ui/dist/assets')).find(name=>name.endsWith('.css'))),'utf8')});await page.addScriptTag({content:bundle.outputFiles[0].text});
 await page.getByRole('button',{name:'Prompt Helper'}).click();await page.getByRole('dialog').waitFor();
 await page.getByRole('button',{name:'Add element'}).click();await page.getByLabel('Element description').fill('A flowering cherry tree.');
 const rect=await page.getByTestId('flux3-layout-canvas').boundingBox();assert(rect);
 await page.mouse.move(rect.x+rect.width*.05,rect.y+rect.height*.1);await page.mouse.down();await page.mouse.move(rect.x+rect.width*.35,rect.y+rect.height*.6,{steps:5});await page.mouse.up();
 await page.getByRole('button',{name:'Save prompt'}).click();const prompt=await page.evaluate(()=>window.store.getState().params.prompt);const array=JSON.parse(prompt.slice(prompt.indexOf('[')));assert.deepEqual(array[0].bbox,[100,50,600,350]);
 await page.getByRole('button',{name:'Prompt Helper'}).click();await page.getByLabel('Scene caption').fill('A quiet garden <element_1>.');await page.getByRole('button',{name:'Save prompt'}).click();
 await page.evaluate(()=>window.store.setState({imageRefs:[new File(['fixture'],'ref.png',{type:'image/png'})]}));
 await page.getByRole('button',{name:'Prompt Helper'}).click();await page.getByLabel('Content source').selectOption('ref_image_0');await page.getByLabel('Remove this reference element').check();await page.getByRole('button',{name:'Save prompt'}).click();
 const edited=await page.evaluate(()=>window.store.getState().params.prompt);const edit=JSON.parse(edited.slice(edited.indexOf('[')))[0];assert.equal(edit.from,'ref_image_0');assert.equal(edit.tgt_bbox,null);assert.deepEqual(edit.src_bbox,[0,0,1000,1000]);assert.deepEqual(errors,[]);await page.close();
 }console.log('Flux3 Prompt Helper desktop/mobile draw, save, reopen, reference binding and removal passed.');}finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
