# VectorMind Frontend — Docker Image
#
# Node builds the React app; nginx serves the output and proxies the API
# to the backend container. The build stage is discarded, so the shipped
# image is nginx plus ~250KB of static assets.
#
# Build:  docker build -f deployment/frontend.Dockerfile -t vectormind-frontend .

# --- Stage 1: build ---
FROM node:22-slim AS builder

WORKDIR /app

# Manifests first so this layer caches across source edits.
COPY frontend/package.json frontend/package-lock.json ./
# `npm ci` rather than `npm install`: it installs exactly the lockfile and
# fails if the two disagree, so an image cannot silently pick up a
# different dependency tree than the one that was tested.
RUN npm ci

COPY frontend/ ./

# Type-check and lint before building. A type error that reaches a
# production image is a type error nobody caught, and this is the last
# gate before it ships.
RUN npx tsc --noEmit && npm run lint && npm run build

# --- Stage 2: serve ---
FROM nginx:alpine AS production

# Drop the default site so it cannot shadow ours.
RUN rm /etc/nginx/conf.d/default.conf
COPY deployment/nginx.conf /etc/nginx/conf.d/vectormind.conf
# Outside conf.d, and not named *.conf: everything in conf.d is included
# automatically at the http level, and these add_header directives are
# meant to be included per-location, not once globally where any location
# with an add_header of its own would discard them again.
COPY deployment/security-headers.inc /etc/nginx/snippets/security-headers.inc
COPY --from=builder /app/dist /usr/share/nginx/html

# Fail the build rather than shipping an image with a bad config.
RUN nginx -t

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/nginx-health >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
