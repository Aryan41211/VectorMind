# Static security headers, in a file rather than inline, because nginx's
# add_header inheritance makes the inline version silently wrong.
#
# add_header directives are inherited into a location ONLY if that
# location declares no add_header of its own. Every location here that
# sets a Cache-Control or Content-Type therefore dropped the entire
# server-level security set — including `location /`, which serves the
# SPA document itself. The headers were verified once on a proxied API
# path, which is one of the few locations that did inherit them, and the
# result was recorded as "all four present".
#
# Including this file in every location keeps the set attached to every
# response, and keeps one place to change it.
#
# HSTS is deliberately absent: it belongs on the hop that terminates
# TLS, which is Caddy (deployment/Caddyfile), not this container.
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
