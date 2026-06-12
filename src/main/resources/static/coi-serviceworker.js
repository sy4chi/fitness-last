/*! coi-serviceworker v0.1.7 | MIT License | https://github.com/gzguidoti/coi-serviceworker */
if (typeof window === 'undefined') {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
    self.addEventListener("fetch", event => {
        if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") {
            return;
        }
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    if (response.status === 0) {
                        return response;
                    }
                    const newHeaders = new Headers(response.headers);
                    newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");
                    newHeaders.set("Cross-Origin-Embedder-Policy", "require-corp");
                    return new Response(response.body, {
                        status: response.status,
                        statusText: response.statusText,
                        headers: newHeaders
                    });
                })
                .catch(e => console.error(e))
        );
    });
} else {
    const isSecureContext = window.isSecureContext;
    if (isSecureContext && 'serviceWorker' in navigator) {
        navigator.serviceWorker.register(window.document.currentScript.src)
            .then(registration => {
                registration.addEventListener("updatefound", () => {
                    navigator.serviceWorker.controller && window.location.reload();
                });
                if (navigator.serviceWorker.controller) {
                    // Service worker is active and controlling
                } else {
                    // First load, reload to activate immediately
                    window.location.reload();
                }
            })
            .catch(err => console.error("COOP/COEP Service Worker registration failed:", err));
    }
}
