# Security Policy

vramwatch does not write to the GPU or control a loader. Live observation shells
out to `nvidia-smi`/`amd-smi`, reads OS counters, and calls configured loader APIs.
It writes only prediction-ledger records and report files requested by the user.

Remote `fit` requests contact the selected Hugging Face or Ollama registry (or an
explicit HTTPS GGUF URL) for metadata. GGUF range reads and JSON responses share a
16 MiB limit, range bodies are closed as soon as architecture parsing completes,
unknown sizes fail closed, and parser counts/depths are bounded.
`doctor --online` also makes explicit registry probes. There is no telemetry.

`HF_TOKEN` is used as an authorization header for Hugging Face metadata and is not
printed or placed in SVG output. Local ledger records and raw JSON can retain the
original model reference, so treat the state directory as private diagnostic data.
The shareable SVG path is separately scrubbed of local paths, URL queries, host
identity, PIDs, bus addresses, and serial-number fields.

## Reporting a vulnerability

If you find a security issue (for example command execution through crafted
provider output, a GGUF parser resource bypass, ledger path traversal, token
disclosure, or private data in an SVG), please report it privately:

- Use GitHub's **"Report a vulnerability"** (Security → Advisories) on this repo, or
- email the maintainer listed on the GitHub profile.

Please do not open a public issue for security reports. You'll get an
acknowledgement within a few days.

## Supported versions

Only the latest released version is supported.
