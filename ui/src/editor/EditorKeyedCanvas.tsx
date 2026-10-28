import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'
import type { EditorKeyingSettings } from '../types'
import { applyEditorKeyPixels, editorKeyColor, editorSpillChannel, normalizeEditorKeying } from './editorKeying'

type Source = HTMLImageElement | HTMLVideoElement
interface Renderer {
  draw: (source: Source, settings: EditorKeyingSettings) => boolean
  dispose: () => void
}

const VERTEX = `attribute vec2 position; varying vec2 uv;
void main(){uv=(position+1.0)*0.5;gl_Position=vec4(position,0.0,1.0);}`
const FRAGMENT = `precision highp float; varying vec2 uv;
uniform sampler2D sourceTexture; uniform vec3 keyColor;
uniform float similarity; uniform float softness; uniform float spill; uniform int spillChannel;
void main(){
 vec4 source=texture2D(sourceTexture,uv);
 float distance=length(source.rgb-keyColor)/sqrt(3.0);
 float matte=softness>0.0001?clamp((distance-similarity)/softness,0.0,1.0):(distance>similarity?1.0:0.0);
 vec3 rgb=source.rgb;
 if(spillChannel==1)rgb.g=max(0.0,rgb.g-spill*max(0.0,source.g-(source.r+source.b)*0.5));
 if(spillChannel==2)rgb.b=max(0.0,rgb.b-spill*max(0.0,source.b-(source.r+source.g)*0.5));
 gl_FragColor=vec4(rgb,source.a*matte);
}`

function sourceSize(source: Source): [number, number] {
  return source instanceof HTMLVideoElement ? [source.videoWidth, source.videoHeight] : [source.naturalWidth, source.naturalHeight]
}

function prepareSize(canvas: HTMLCanvasElement, source: Source, maximum: number): boolean {
  const [width, height] = sourceSize(source)
  if (!width || !height || (source instanceof HTMLVideoElement && source.readyState < 2)) return false
  const scale = Math.min(1, maximum / Math.max(width, height))
  const targetWidth = Math.max(1, Math.round(width * scale)), targetHeight = Math.max(1, Math.round(height * scale))
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) { canvas.width = targetWidth; canvas.height = targetHeight }
  canvas.dataset.keyPreviewReduced = String(scale < 1)
  return true
}

function webglRenderer(canvas: HTMLCanvasElement): Renderer {
  const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: false, preserveDrawingBuffer: true })
  if (!gl) throw new Error('WebGL preview unavailable')
  const shaders: WebGLShader[] = []
  const program = gl.createProgram(), buffer = gl.createBuffer(), texture = gl.createTexture()
  const dispose = () => {
    if (texture) gl.deleteTexture(texture)
    if (buffer) gl.deleteBuffer(buffer)
    if (program) gl.deleteProgram(program)
    shaders.forEach(shader => gl.deleteShader(shader))
    // An inactive standby deck can become active again on this same canvas.
    // Lose the context only after its DOM node has actually been removed.
    if (!canvas.isConnected) gl.getExtension('WEBGL_lose_context')?.loseContext()
  }
  try {
    if (!program || !buffer || !texture) throw new Error('Preview resources unavailable')
    for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, FRAGMENT]] as const) {
      const shader = gl.createShader(type)
      if (!shader) throw new Error('Preview shader unavailable')
      shaders.push(shader)
      gl.shaderSource(shader, source); gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error('Preview shader could not compile')
      gl.attachShader(program, shader)
    }
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Preview shader could not link')
    gl.useProgram(program)
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW)
    const position = gl.getAttribLocation(program, 'position')
    gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    gl.uniform1i(gl.getUniformLocation(program, 'sourceTexture'), 0)
    const locations = Object.fromEntries(['keyColor', 'similarity', 'softness', 'spill', 'spillChannel'].map(name => [name, gl.getUniformLocation(program, name)]))
    const textureMaximum = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
    const maximum = Math.min(4096, textureMaximum)
    return { dispose, draw(source, settings) {
      if (!prepareSize(canvas, source, maximum)) return false
      if (gl.isContextLost()) throw new Error('Preview context lost')
      if (Math.max(...sourceSize(source)) > textureMaximum) throw new Error('Source exceeds preview texture capacity')
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.useProgram(program)
      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source)
      if (gl.getError() !== gl.NO_ERROR) throw new Error('Preview could not upload this source')
      gl.uniform3fv(locations.keyColor, editorKeyColor(settings.color))
      gl.uniform1f(locations.similarity, settings.similarity)
      gl.uniform1f(locations.softness, settings.softness)
      gl.uniform1f(locations.spill, settings.spill)
      gl.uniform1i(locations.spillChannel, editorSpillChannel(settings.color))
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 6)
      canvas.dataset.keyPreviewReady = 'true'
      return true
    } }
  } catch (error) { dispose(); throw error }
}

function cpuRenderer(canvas: HTMLCanvasElement): Renderer {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Canvas preview unavailable')
  return { dispose() {}, draw(source, settings) {
    if (!prepareSize(canvas, source, 960)) return false
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(source, 0, 0, canvas.width, canvas.height)
    const frame = context.getImageData(0, 0, canvas.width, canvas.height)
    applyEditorKeyPixels(frame.data, settings)
    context.putImageData(frame, 0, 0)
    canvas.dataset.keyPreviewReady = 'true'
    return true
  } }
}

/** Keep the original media element/clock/audio; render only its keyed visual layer. */
export function EditorKeyedCanvas({ sourceRef, sourceUrl, settings, active, fit, filter }: {
  sourceRef: RefObject<Source | null>; sourceUrl: string; settings: EditorKeyingSettings
  active: boolean; fit: 'contain' | 'cover'; filter?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const latestSettings = useRef(settings)
  const redraw = useRef<(() => void) | null>(null)
  const [mode, setMode] = useState<'webgl' | 'cpu'>('webgl')
  const [failed, setFailed] = useState(false)
  const normalized = useMemo(() => normalizeEditorKeying(settings), [settings])
  useEffect(() => {
    const canvas = canvasRef.current, source = sourceRef.current
    if (!canvas || !source || !active) return
    let renderer: Renderer
    let closed = false, animation = 0, videoFrame: number | null = null, lastDraw = -Infinity
    const fallback = () => {
      if (closed) return
      if (mode === 'webgl') setMode('cpu')
      else { closed = true; setFailed(true) }
    }
    const onLost = (event: Event) => { event.preventDefault(); fallback() }
    canvas.addEventListener('webglcontextlost', onLost)
    try { renderer = mode === 'webgl' ? webglRenderer(canvas) : cpuRenderer(canvas) }
    catch { fallback(); canvas.removeEventListener('webglcontextlost', onLost); return }
    const video = source instanceof HTMLVideoElement ? source : null
    const draw = () => {
      if (closed) return false
      try { return renderer.draw(source, latestSettings.current) }
      catch { fallback(); return false }
    }
    const nextFrame = (now: number) => {
      animation = 0
      if (closed) return
      if (mode === 'webgl' || now - lastDraw >= 1000 / 15 || video?.paused) { draw(); lastDraw = now }
      if (video?.requestVideoFrameCallback) videoFrame = video.requestVideoFrameCallback(nextFrame)
      else if (video && !video.paused && !video.ended) animation = requestAnimationFrame(nextFrame)
    }
    const onMediaChange = () => {
      if (closed) return
      if (draw()) setFailed(false)
      if (video && !video.requestVideoFrameCallback && !video.paused && !animation) animation = requestAnimationFrame(nextFrame)
    }
    redraw.current = draw
    const events = ['load', 'loadeddata', 'seeked', 'resize', 'playing', 'pause', 'ended']
    events.forEach(event => source.addEventListener(event, onMediaChange))
    draw()
    if (video?.requestVideoFrameCallback) videoFrame = video.requestVideoFrameCallback(nextFrame)
    else if (video && !video.paused) animation = requestAnimationFrame(nextFrame)
    return () => {
      closed = true; redraw.current = null
      cancelAnimationFrame(animation)
      if (videoFrame !== null) video?.cancelVideoFrameCallback?.(videoFrame)
      events.forEach(event => source.removeEventListener(event, onMediaChange))
      canvas.removeEventListener('webglcontextlost', onLost)
      renderer.dispose()
    }
  }, [active, mode, sourceRef, sourceUrl])
  useEffect(() => { latestSettings.current = normalized; redraw.current?.() }, [normalized])
  return <>
    <canvas key={mode} ref={canvasRef} aria-hidden="true" data-editor-keyed-canvas data-key-renderer={mode}
      className="pointer-events-none absolute inset-0 h-full w-full" style={{ objectFit: fit, filter }} />
    {mode === 'cpu' && !failed && <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-black/70 px-2 py-1 text-[10px] text-white/80">Reduced-resolution key preview</span>}
    {failed && <span role="status" className="pointer-events-none absolute inset-0 grid place-items-center bg-bg-secondary/90 p-4 text-center text-xs text-text-secondary">Keyed preview could not read this source. The original is preserved; check its format before exporting.</span>}
  </>
}
