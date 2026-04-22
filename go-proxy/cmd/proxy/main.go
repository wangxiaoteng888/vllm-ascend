package main

import (
	"flag"
	"log"
	"net/http"
	"strings"
	"time"

	"example.com/loadbalance-proxy/proxy"
)

func main() {
	var (
		listenAddr = flag.String("listen", "127.0.0.1:9000", "proxy listen addr")
		prefillers = flag.String("prefillers", "127.0.0.1:8100", "comma-separated prefill addrs host:port")
		decoders   = flag.String("decoders", "127.0.0.1:8200", "comma-separated decode addrs host:port")
		maxRetries = flag.Int("max-retries", 3, "max retries for backend requests")
		retryDelay = flag.Duration("retry-delay", 200*time.Millisecond, "base retry delay")
	)
	flag.Parse()

	p, err := proxy.New(proxy.Config{
		Prefillers: splitCSV(*prefillers),
		Decoders:   splitCSV(*decoders),
		MaxRetries: *maxRetries,
		RetryDelay: *retryDelay,
	})
	if err != nil {
		log.Fatal(err)
	}

	srv := &http.Server{
		Addr:         *listenAddr,
		Handler:      p.Handler(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // streaming
		IdleTimeout:  90 * time.Second,
	}
	log.Printf("proxy listening on %s", *listenAddr)
	log.Fatal(srv.ListenAndServe())
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

