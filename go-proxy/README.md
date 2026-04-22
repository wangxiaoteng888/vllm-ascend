## Go load-balance proxy (vLLM-style two-hop)

### What it does
- Exposes OpenAI-compatible endpoints:
  - `POST /v1/completions`
  - `POST /v1/chat/completions`
- Two-hop routing:
  - First sends a rewritten request to a **prefiller** to fetch `kv_transfer_params`
  - Then forwards the original request (plus `kv_transfer_params`) to a **decoder**
  - Streams decoder response back to the client
- Admin endpoints:
  - `GET /healthcheck`
  - `POST /instances/add`
  - `POST /instances/remove`

### Run
```bash
cd go-proxy
go run ./cmd/proxy -listen 127.0.0.1:9000 -prefillers 127.0.0.1:8100,127.0.0.1:8101 -decoders 127.0.0.1:8200,127.0.0.1:8201
```

### Tests (includes ~10k concurrency)
```bash
cd go-proxy
go test ./... -count=1
```

