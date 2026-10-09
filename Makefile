BINARY  := vramwatch
PKG     := ./cmd/vramwatch
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -ldflags "-X main.Version=$(VERSION)"

.PHONY: build test vet staticcheck vuln fmt tidy run watch demo card gif clean

build: ## build the CLI
	go build $(LDFLAGS) -o $(BINARY) $(PKG)

test: ## run the test suite
	@if [ "$$(go env CGO_ENABLED)" = "1" ]; then \
		race=-race; \
	else \
		printf '%s\n' 'CGO disabled: running tests without -race'; \
		race=; \
	fi; \
	for module in . docs/gifgen; do \
		(cd "$$module" && go test $$race -timeout=60s ./...) || exit $$?; \
	done

vet:
	go vet ./...
	cd docs/gifgen && go vet ./...

staticcheck:
	staticcheck ./...
	cd docs/gifgen && staticcheck ./...

vuln:
	govulncheck ./...
	cd docs/gifgen && govulncheck ./...

fmt: ## check formatting (fails if any file needs gofmt)
	@files="$$(gofmt -l $$(git ls-files -co --exclude-standard '*.go'))" || exit 1; \
		test -z "$$files" || { printf 'gofmt needed:\n%s\n' "$$files"; exit 1; }

tidy:
	go mod tidy

run: build ## show the launch CLI
	./$(BINARY) help

watch: build ## live TUI against the local machine
	./$(BINARY) watch

demo: build ## live TUI against the synthetic growing-KV demo source
	./$(BINARY) watch --source demo

card: ## regenerate the deterministic report card used in the docs
	go run ./tools/reportfixture > docs/sample/vramwatch-card.svg

gif: ## regenerate docs/demo.gif (standalone module; fetches golang.org/x/image)
	cd docs/gifgen && go run . ../demo.gif

clean:
	rm -f $(BINARY) $(BINARY).exe
	rm -rf dist
