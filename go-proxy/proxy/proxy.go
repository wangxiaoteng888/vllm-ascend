package proxy

import (
	"bufio"
	"bytes"
	"container/heap"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

type InstanceType string

const (
	InstancePrefill InstanceType = "prefill"
	InstanceDecode  InstanceType = "decode"
)

type Config struct {
	Prefillers []string // host:port
	Decoders   []string // host:port

	MaxRetries int
	RetryDelay time.Duration

	Client *http.Client // optional; if nil a default high-concurrency client is used
}

type Server struct {
	Addr string // host:port
	URL  string // http://host:port/v1 (or http://[v6]/v1)
}

type Proxy struct {
	cfg Config

	mu sync.Mutex

	prefillServers []Server
	decodeServers  []Server

	prefillSel *selector
	decodeSel  *selector
}

func New(cfg Config) (*Proxy, error) {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	if cfg.RetryDelay <= 0 {
		cfg.RetryDelay = 200 * time.Millisecond
	}
	if cfg.Client == nil {
		cfg.Client = defaultHTTPClient()
	}

	p := &Proxy{cfg: cfg}
	for _, a := range cfg.Prefillers {
		s, err := newServer(a)
		if err != nil {
			return nil, fmt.Errorf("prefill addr %q: %w", a, err)
		}
		p.prefillServers = append(p.prefillServers, s)
	}
	for _, a := range cfg.Decoders {
		s, err := newServer(a)
		if err != nil {
			return nil, fmt.Errorf("decode addr %q: %w", a, err)
		}
		p.decodeServers = append(p.decodeServers, s)
	}
	if len(p.prefillServers) == 0 || len(p.decodeServers) == 0 {
		return nil, errors.New("need at least 1 prefiller and 1 decoder")
	}
	p.prefillSel = newSelector(len(p.prefillServers))
	p.decodeSel = newSelector(len(p.decodeServers))
	return p, nil
}

func defaultHTTPClient() *http.Client {
	dialer := &net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	tr := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          200000,
		MaxIdleConnsPerHost:   200000,
		MaxConnsPerHost:       200000,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
	return &http.Client{Transport: tr}
}

func newServer(addr string) (Server, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return Server{}, err
	}
	urlHost := host
	if ip := net.ParseIP(host); ip != nil && strings.Contains(host, ":") {
		urlHost = "[" + host + "]"
	}
	return Server{
		Addr: addr,
		URL:  fmt.Sprintf("http://%s:%s/v1", urlHost, port),
	}, nil
}

func (p *Proxy) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthcheck", p.handleHealthcheck)
	mux.HandleFunc("/instances/add", p.handleInstancesAdd)
	mux.HandleFunc("/instances/remove", p.handleInstancesRemove)
	mux.HandleFunc("/v1/completions", p.handleCompletions("/completions"))
	mux.HandleFunc("/v1/chat/completions", p.handleCompletions("/chat/completions"))
	return mux
}

func (p *Proxy) handleHealthcheck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	p.mu.Lock()
	prefillN, decodeN := len(p.prefillServers), len(p.decodeServers)
	p.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":            "ok",
		"prefill_instances": prefillN,
		"decode_instances":  decodeN,
	})
}

type adjustReq struct {
	Type      InstanceType `json:"type"`
	Instances any          `json:"instances"` // string or []string
}

func (p *Proxy) handleInstancesAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req adjustReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	addrs, err := normalizeInstances(req.Instances)
	if err != nil {
		http.Error(w, "bad instances", http.StatusBadRequest)
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	switch req.Type {
	case InstancePrefill:
		added := p.addServersLocked(&p.prefillServers, p.prefillSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("add prefill instances: %v", added),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	case InstanceDecode:
		added := p.addServersLocked(&p.decodeServers, p.decodeSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("add decode instances: %v", added),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	default:
		http.Error(w, "unsupported type", http.StatusBadRequest)
	}
}

func (p *Proxy) handleInstancesRemove(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req adjustReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	addrs, err := normalizeInstances(req.Instances)
	if err != nil {
		http.Error(w, "bad instances", http.StatusBadRequest)
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	switch req.Type {
	case InstancePrefill:
		removed := p.removeServersLocked(&p.prefillServers, &p.prefillSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("remove prefill instances: %v", removed),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	case InstanceDecode:
		removed := p.removeServersLocked(&p.decodeServers, &p.decodeSel, addrs)
		writeJSON(w, http.StatusOK, map[string]any{
			"message":                   fmt.Sprintf("remove decode instances: %v", removed),
			"current_prefill_instances": addrsOf(p.prefillServers),
			"current_decode_instances":  addrsOf(p.decodeServers),
		})
	default:
		http.Error(w, "unsupported type", http.StatusBadRequest)
	}
}

func normalizeInstances(v any) ([]string, error) {
	switch t := v.(type) {
	case string:
		return []string{t}, nil
	case []any:
		out := make([]string, 0, len(t))
		for _, x := range t {
			s, ok := x.(string)
			if !ok {
				return nil, errors.New("instances must be strings")
			}
			out = append(out, s)
		}
		return out, nil
	case []string:
		return t, nil
	default:
		return nil, errors.New("instances must be string or []string")
	}
}

func addrsOf(ss []Server) []string {
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		out = append(out, s.Addr)
	}
	return out
}

func (p *Proxy) addServersLocked(dst *[]Server, sel *selector, addrs []string) []string {
	exists := map[string]bool{}
	for _, s := range *dst {
		exists[s.Addr] = true
	}
	var added []string
	for _, a := range addrs {
		if exists[a] {
			continue
		}
		s, err := newServer(a)
		if err != nil {
			continue
		}
		*dst = append(*dst, s)
		added = append(added, a)
	}
	if len(added) > 0 {
		*sel = *newSelector(len(*dst))
	}
	return added
}

func (p *Proxy) removeServersLocked(dst *[]Server, selPtr **selector, addrs []string) []string {
	toRemove := map[string]bool{}
	for _, a := range addrs {
		toRemove[a] = true
	}
	var kept []Server
	var removed []string
	for _, s := range *dst {
		if toRemove[s.Addr] {
			removed = append(removed, s.Addr)
			continue
		}
		kept = append(kept, s)
	}
	*dst = kept
	if len(*dst) > 0 {
		*selPtr = newSelector(len(*dst))
	}
	return removed
}

func (p *Proxy) handleCompletions(apiPath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body failed", http.StatusBadRequest)
			return
		}
		var reqData map[string]any
		if err := json.Unmarshal(bodyBytes, &reqData); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		requestLen := len(bodyBytes)

		ctx := r.Context()
		requestID := newRequestID()

		prefillScore := calculatePrefillScore(requestLen)
		decodeScore := calculateDecodeScore(requestLen)

		p.mu.Lock()
		prefillIdx := p.prefillSel.acquire(prefillScore)
		prefill := p.prefillServers[prefillIdx]
		p.mu.Unlock()

		kvParams, err := p.doPrefill(ctx, prefill, apiPath, prefillIdx, reqData, requestID)
		p.mu.Lock()
		p.prefillSel.release(prefillIdx, prefillScore)
		p.mu.Unlock()
		if err != nil {
			http.Error(w, "prefill failed", http.StatusBadGateway)
			return
		}
		if kvParams != nil {
			reqData["kv_transfer_params"] = kvParams
		}

		p.mu.Lock()
		decodeIdx := p.decodeSel.acquire(decodeScore)
		decoder := p.decodeServers[decodeIdx]
		p.mu.Unlock()

		defer func() {
			p.mu.Lock()
			p.decodeSel.release(decodeIdx, decodeScore)
			p.mu.Unlock()
		}()

		streamFlag := false
		if v, ok := reqData["stream"].(bool); ok && v {
			streamFlag = true
		}

		if err := p.streamDecode(ctx, w, r, decoder, apiPath, reqData, requestID, streamFlag); err != nil {
			if !errors.Is(err, context.Canceled) {
				http.Error(w, "decode failed", http.StatusBadGateway)
			}
			return
		}
	}
}

func calculatePrefillScore(requestLen int) float64 {
	lengthScore := float64(requestLen) / 4.0
	return lengthScore*0.0345 + 120.0745
}

func calculateDecodeScore(requestLen int) float64 {
	return float64(requestLen)
}

func (p *Proxy) doPrefill(
	ctx context.Context,
	server Server,
	apiPath string,
	prefillIdx int,
	reqData map[string]any,
	requestID string,
) (map[string]any, error) {
	prefillReq := cloneMap(reqData)
	prefillReq["kv_transfer_params"] = map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  nil,
		"remote_block_ids":  nil,
		"remote_host":       nil,
		"remote_port":       nil,
		"aborted_request":   []string{},
	}
	prefillReq["stream"] = false
	prefillReq["max_tokens"] = 1
	prefillReq["min_tokens"] = 1
	if _, ok := prefillReq["max_completion_tokens"]; ok {
		prefillReq["max_completion_tokens"] = 1
	}
	delete(prefillReq, "stream_options")

	var respBytes []byte
	var err error
	for attempt := 1; attempt <= p.cfg.MaxRetries; attempt++ {
		respBytes, err = p.postJSON(ctx, server.URL+apiPath, prefillReq, requestID)
		if err == nil {
			break
		}
		if attempt < p.cfg.MaxRetries {
			select {
			case <-time.After(p.cfg.RetryDelay * time.Duration(1<<(attempt-1))):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
	if err != nil {
		return nil, err
	}
	var resp map[string]any
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		return nil, err
	}
	if kv, ok := resp["kv_transfer_params"].(map[string]any); ok {
		return kv, nil
	}
	return nil, nil
}

func (p *Proxy) streamDecode(
	ctx context.Context,
	w http.ResponseWriter,
	r *http.Request,
	server Server,
	apiPath string,
	reqData map[string]any,
	requestID string,
	streamFlag bool,
) error {
	targetURL := server.URL + apiPath

	for attempt := 1; attempt <= p.cfg.MaxRetries; attempt++ {
		reqBody, err := json.Marshal(reqData)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(reqBody))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Request-Id", requestID)
		if auth := r.Header.Get("Authorization"); auth != "" {
			req.Header.Set("Authorization", auth)
		}

		resp, err := p.cfg.Client.Do(req)
		if err != nil {
			if attempt < p.cfg.MaxRetries {
				select {
				case <-time.After(p.cfg.RetryDelay * time.Duration(1<<(attempt-1))):
					continue
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return err
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			_ = resp.Body.Close()
			if attempt < p.cfg.MaxRetries {
				select {
				case <-time.After(p.cfg.RetryDelay * time.Duration(1<<(attempt-1))):
					continue
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return fmt.Errorf("decoder status %d", resp.StatusCode)
		}

		defer resp.Body.Close()

		for k, vv := range resp.Header {
			for _, v := range vv {
				w.Header().Add(k, v)
			}
		}
		if streamFlag {
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		w.WriteHeader(http.StatusOK)

		flusher, _ := w.(http.Flusher)
		bw := bufio.NewWriterSize(w, 32*1024)
		defer bw.Flush()

		buf := make([]byte, 32*1024)
		for {
			n, readErr := resp.Body.Read(buf)
			if n > 0 {
				if _, err := bw.Write(buf[:n]); err != nil {
					return err
				}
				if flusher != nil {
					bw.Flush()
					flusher.Flush()
				}
			}
			if readErr != nil {
				if errors.Is(readErr, io.EOF) {
					return nil
				}
				return readErr
			}
		}
	}
	return nil
}

func (p *Proxy) postJSON(ctx context.Context, url string, payload any, requestID string) ([]byte, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", requestID)

	resp, err := p.cfg.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

func cloneMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m)+4)
	for k, v := range m {
		out[k] = v
	}
	return out
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func newRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

type selector struct {
	cur []float64
	ver []uint64
	h   selHeap
}

func newSelector(n int) *selector {
	s := &selector{
		cur: make([]float64, n),
		ver: make([]uint64, n),
	}
	s.h = make(selHeap, 0, n*2)
	for i := 0; i < n; i++ {
		heap.Push(&s.h, selItem{score: 0, idx: i, ver: 0})
	}
	return s
}

func (s *selector) acquire(add float64) int {
	for {
		it := heap.Pop(&s.h).(selItem)
		if it.idx < 0 || it.idx >= len(s.cur) {
			continue
		}
		if it.ver != s.ver[it.idx] || it.score != s.cur[it.idx] {
			continue
		}
		s.cur[it.idx] += add
		s.ver[it.idx]++
		heap.Push(&s.h, selItem{score: s.cur[it.idx], idx: it.idx, ver: s.ver[it.idx]})
		return it.idx
	}
}

func (s *selector) release(idx int, sub float64) {
	if idx < 0 || idx >= len(s.cur) {
		return
	}
	s.cur[idx] -= sub
	if s.cur[idx] < 0 {
		s.cur[idx] = 0
	}
	s.ver[idx]++
	heap.Push(&s.h, selItem{score: s.cur[idx], idx: idx, ver: s.ver[idx]})
}

type selItem struct {
	score float64
	idx   int
	ver   uint64
}

type selHeap []selItem

func (h selHeap) Len() int           { return len(h) }
func (h selHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h selHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *selHeap) Push(x any) { *h = append(*h, x.(selItem)) }
func (h *selHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

