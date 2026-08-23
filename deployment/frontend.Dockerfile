# VectorMind Frontend — Docker Image
#
# Multi-stage build: Node builds the React app, nginx serves the static output.
# The frontend is self-contained after build — no runtime deps beyond nginx.
#
# Build:  docker build -f deployment/frontend.Dockerfile -t vectormind-frontend .
# Run:    docker run -p 80:80 vectormind-frontend

# --- Stage 1: Build ---
FROM node:22-slim AS builder

WORKDIR /app

# Install deps first (layer caching)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Serve ---
FROM nginx:alpine AS production

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# SPA routing: serve index.html for all unmatched paths
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
