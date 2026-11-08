// The blog moved to ahammadnafiz.com/writing. This replaces the old offline-cache service worker:
// it clears the cached blog, unregisters itself and reloads open tabs so they follow the redirect.
self.addEventListener("install", () => self.skipWaiting())
self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) await caches.delete(key)
      await self.registration.unregister()
      for (const client of await self.clients.matchAll({ type: "window" })) client.navigate(client.url)
    })(),
  )
})
