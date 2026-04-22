package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type backendCounters struct {
	prefillCalls atomic.Int64
	decodeCalls  atomic.Int64
}

type inmemTransport struct {
	mu       sync.RWMutex
	handlers map[string]http.Handler // key: host:port
}

func newInmemTransport() *inmemTransport {
	return &inmemTransport{handlers: map[string]http.Handler{}}
}

func (t *inmemTransport) Register(hostport string, h http.Handler) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.handlers[hostport] = h
}

func (t *inmemTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.mu.RLock()
	h := t.handlers[req.URL.Host]
	t.mu.RUnlock()
	if h == nil {
		return nil, fmt.Errorf("no inmem handler for %s", req.URL.Host)
	}
	rr := httptest.NewRecorder()
	r2 := req.Clone(req.Context())
	r2.URL = &url.URL{}
	*r2.URL = *req.URL
	h.ServeHTTP(rr, r2)
	return rr.Result(), nil
}

func newMockPrefillerHandler(t *testing.T, c *backendCounters) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != "/v1/completions" && r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		c.prefillCalls.Add(1)

		var m map[string]any
		if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if v, _ := m["stream"].(bool); v {
			http.Error(w, "prefill must be non-stream", http.StatusBadRequest)
			return
		}
		if mt, ok := m["max_tokens"].(float64); !ok || int(mt) != 1 {
			http.Error(w, "max_tokens must be 1", http.StatusBadRequest)
			return
		}
		if kv, ok := m["kv_transfer_params"].(map[string]any); !ok || kv["do_remote_decode"] != true {
			http.Error(w, "missing kv_transfer_params", http.StatusBadRequest)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"kv_transfer_params": map[string]any{
				"remote_host": "decoder",
				"remote_port": 0,
			},
		})
	})
}

func newMockDecoderHandler(t *testing.T, c *backendCounters) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Path != "/v1/completions" && r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		c.decodeCalls.Add(1)

		var m map[string]any
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &m)

		streamFlag, _ := m["stream"].(bool)
		if streamFlag {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			fl, _ := w.(http.Flusher)
			for i := 0; i < 3; i++ {
				msg := fmt.Sprintf("data: %s\n\n", `{"choices":[{"text":"x"}]}`)
				_, _ = w.Write([]byte(msg))
				if fl != nil {
					fl.Flush()
				}
			}
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			if fl != nil {
				fl.Flush()
			}
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"id":      r.Header.Get("X-Request-Id"),
			"choices": []any{map[string]any{"text": "ok"}},
		})
	})
}

func newMockPrefiller(t *testing.T, c *backendCounters) *httptest.Server {
	t.Helper()
	return httptest.NewServer(newMockPrefillerHandler(t, c))
}

func newMockDecoder(t *testing.T, c *backendCounters) *httptest.Server {
	t.Helper()
	return httptest.NewServer(newMockDecoderHandler(t, c))
}

func TestProxy_10000Concurrent(t *testing.T) {
	t.Parallel()

	var b0, b1 backendCounters

	tr := newInmemTransport()
	tr.Register("prefill0:1", newMockPrefillerHandler(t, &b0))
	tr.Register("prefill1:1", newMockPrefillerHandler(t, &b1))
	tr.Register("decode0:1", newMockDecoderHandler(t, &b0))
	tr.Register("decode1:1", newMockDecoderHandler(t, &b1))

	p, err := New(Config{
		Prefillers: []string{"prefill0:1", "prefill1:1"},
		Decoders:   []string{"decode0:1", "decode1:1"},
		MaxRetries: 3,
		RetryDelay: 1 * time.Millisecond,
		Client:     &http.Client{Transport: tr},
	})
	if err != nil {
		t.Fatal(err)
	}
	h := p.Handler()

	const N = 10000
	var okCount atomic.Int64
	var status4xx atomic.Int64
	var status5xx atomic.Int64
	var wg sync.WaitGroup
	wg.Add(N)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	firstErr := make(chan string, 20)
	for i := 0; i < N; i++ {
		go func(i int) {
			defer wg.Done()
			reqBody := map[string]any{
				"model":      "m",
				"prompt":     fmt.Sprintf("p-%d", i),
				"max_tokens": 16,
				"stream":     false,
			}
			b, _ := json.Marshal(reqBody)
			req := httptest.NewRequest(http.MethodPost, "http://proxy.local/v1/completions", bytes.NewReader(b)).WithContext(ctx)
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			if rr.Code != http.StatusOK {
				body := rr.Body.String()
				if rr.Code >= 400 && rr.Code < 500 {
					status4xx.Add(1)
				} else if rr.Code >= 500 {
					status5xx.Add(1)
				}
				select {
				case firstErr <- fmt.Sprintf("status %d body=%q", rr.Code, body):
				default:
				}
				return
			}
			var out map[string]any
			if err := json.NewDecoder(rr.Body).Decode(&out); err != nil {
				select {
				case firstErr <- fmt.Sprintf("decode err: %v", err):
				default:
				}
				return
			}
			choices, _ := out["choices"].([]any)
			if len(choices) == 0 {
				select {
				case firstErr <- "empty choices":
				default:
				}
				return
			}
			okCount.Add(1)
		}(i)
	}

	wg.Wait()
	if got := okCount.Load(); got != N {
		var samples []string
		for {
			select {
			case s := <-firstErr:
				samples = append(samples, s)
			default:
				goto done
			}
		}
	done:
		t.Fatalf("ok %d/%d (4xx=%d 5xx=%d) samples=%v", got, N, status4xx.Load(), status5xx.Load(), samples)
	}

	p0c := b0.prefillCalls.Load()
	p1c := b1.prefillCalls.Load()
	d0c := b0.decodeCalls.Load()
	d1c := b1.decodeCalls.Load()
	if p0c == 0 || p1c == 0 || d0c == 0 || d1c == 0 {
		t.Fatalf("uneven distribution: prefill(%d,%d) decode(%d,%d)", p0c, p1c, d0c, d1c)
	}
}

func TestProxy_Stream(t *testing.T) {
	t.Parallel()
	var b backendCounters
	p0 := newMockPrefiller(t, &b)
	defer p0.Close()
	d0 := newMockDecoder(t, &b)
	defer d0.Close()

	p, err := New(Config{
		Prefillers: []string{trimHTTP(p0.URL)},
		Decoders:   []string{trimHTTP(d0.URL)},
		MaxRetries: 1,
		RetryDelay: 1 * time.Millisecond,
		Client:     defaultHTTPClient(),
	})
	if err != nil {
		t.Fatal(err)
	}
	proxySrv := httptest.NewServer(p.Handler())
	defer proxySrv.Close()

	reqBody := map[string]any{
		"model":      "m",
		"prompt":     "hello",
		"max_tokens": 16,
		"stream":     true,
	}
	bb, _ := json.Marshal(reqBody)
	resp, err := http.Post(proxySrv.URL+"/v1/completions", "application/json", bytes.NewReader(bb))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type %q", ct)
	}
	out, _ := io.ReadAll(resp.Body)
	if !bytes.Contains(out, []byte("[DONE]")) {
		t.Fatalf("missing DONE: %q", string(out))
	}
}

func trimHTTP(url string) string {
	return strings.TrimPrefix(url, "http://")
}

