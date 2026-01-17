package main

import (
	"flag"
	"log"

	"parakeet/internal/server"
)

func main() {
	cfg := server.Config{}

	flag.IntVar(&cfg.Port, "port", 5092, "Server port")
	flag.StringVar(&cfg.ModelsDir, "models", "./models", "Models directory")
	flag.BoolVar(&cfg.Debug, "debug", false, "Enable debug logging")
	flag.Parse()

	srv, err := server.New(cfg)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}
	defer srv.Close()

	log.Fatal(srv.Run())
}
