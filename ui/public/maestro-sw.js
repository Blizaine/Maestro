/* Maestro notification worker. Application assets are intentionally not
 * cached: Pinokio updates must always load the current UI bundle. */
self.addEventListener('install', () => self.skipWaiting())

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim())
})

self.addEventListener('push', event => {
  let payload = {}
  try {
    payload = event.data ? event.data.json() : {}
  } catch {
    payload = { body: event.data ? event.data.text() : '' }
  }
  const title = payload.title || 'Maestro'
  event.waitUntil((async () => {
    if (payload.onlyWhenHidden) {
      const windows = await self.clients.matchAll({
        type: 'window',
        includeUncontrolled: true,
      })
      if (windows.some(client => client.visibilityState === 'visible')) return
    }
    await self.registration.showNotification(title, {
      body: payload.body || 'Your Maestro generation has finished.',
      icon: '/maestro-icon.png',
      badge: '/maestro-icon.png',
      tag: payload.tag || 'maestro-generation',
      data: { url: payload.url || '/' },
    })
  })())
})

self.addEventListener('notificationclick', event => {
  event.notification.close()
  const targetUrl = event.notification.data?.url || '/'
  event.waitUntil((async () => {
    const windows = await self.clients.matchAll({
      type: 'window',
      includeUncontrolled: true,
    })
    for (const client of windows) {
      if ('focus' in client) {
        if ('navigate' in client) await client.navigate(targetUrl)
        return client.focus()
      }
    }
    if (self.clients.openWindow) return self.clients.openWindow(targetUrl)
    return undefined
  })())
})
