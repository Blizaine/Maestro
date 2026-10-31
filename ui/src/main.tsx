import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { AgentAccessSessionGate } from './components/AgentAccessSessionGate'
import { installAgentAccessAuthInterceptor } from './api/agentAccess'

// Mobile browsers require persistent notifications from a service worker;
// `new Notification()` is desktop-only on several engines, including iOS
// WebKit. This worker deliberately does not cache application assets, so a
// Maestro update can never strand users on an old UI bundle.
if ('serviceWorker' in navigator) {
  void navigator.serviceWorker.register('/maestro-sw.js', { scope: '/' }).catch(error => {
    console.warn('[Notifications] Service worker registration failed:', error)
  })
}

// Install before React mounts so any same-origin API request can notify the
// session gate when the local owner cookie expires.
installAgentAccessAuthInterceptor()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <>
      <AgentAccessSessionGate />
      <App />
    </>
  </StrictMode>,
)
