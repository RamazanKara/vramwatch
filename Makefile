BINARY  := vramwatch
PKG     := ./cmd/vramwatch
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -ldflags "-X main.Version=$(VERSION)"

.PHONY: build test vet fmt tidy run watch demo card clean

build: ## build the CLI
	go build $(LDFLAGS) -o $(BINARY) $(PKG)

test: ## run the test suite
	go test ./...

vet:
	go vet ./...

fmt: ## check formatting (fails if any file needs gofmt)
	@test -z "$$(gofmt -l .)" || { echo "gofmt needed:"; gofmt -l .; exit 1; }

tidy:
	go mod tidy

run: build ## one-shot snapshot of the local machine
	./$(BINARY) snapshot

watch: build ## live TUI against the local machine
	./$(BINARY) watch

demo: build ## live TUI against the synthetic growing-KV demo source
	./$(BINARY) watch --source demo

card: build ## regenerate the committed sample scorecard + JSON
	./$(BINARY) snapshot --source mock:testdata/scenarios/24gb-70b-oom.json --static --svg docs/sample/vramwatch-card.svg
	./$(BINARY) snapshot --source mock:testdata/scenarios/24gb-70b-oom.json --static --json > docs/sample/snapshot.json

clean:
	rm -f $(BINARY) $(BINARY).exe
	rm -rf dist
