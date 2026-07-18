# FAQ

### Does `fit` download the model?

No model tensors are intentionally downloaded. Hugging Face and Ollama fits use
repository/manifest metadata for byte size and one bounded HTTP Range request for
the GGUF header. Direct HTTPS GGUFs use the same range path. The response is closed
as soon as the architecture is complete. The combined remote metadata budget is
16 MiB; an ignored large Range response is refused.

For a private Hugging Face repository, export `HF_TOKEN`. The token is sent only
to Hugging Face metadata requests and is not written to the report card.

### Which `MODEL` syntax should I use?

```sh
vramwatch fit ollama:llama3.2:3b-instruct --quant q4_k_m --context 32768
vramwatch fit hf:owner/repo --quant q4_k_m --context 32768
vramwatch fit owner/repo --quant q4_k_m --context 32768  # inferred Hugging Face
vramwatch fit /models/model.gguf --context 32768
vramwatch fit https://host/model.gguf --context 32768
```

Use `--file` when a Hub repository contains more than one matching GGUF. For an
Ollama name containing `/`, keep the explicit `ollama:` prefix so it is not
interpreted as a Hugging Face repository.

### Why are there “on device” and “right now” verdicts?

`on device` answers whether the complete model fits the accelerator's capacity.
`right now` subtracts current allocations. A model can therefore fit the card but
not fit until another process/model is unloaded.

If the provider can read capacity but not current usage, `right now` is `UNKNOWN`.
vramwatch does not assume an unmeasured card is empty. Run `vramwatch doctor` for
the failing counter/provider.

### Why can a required value exceed the displayed conservative footprint?

The launch verdict also includes a per-device safety reserve:

```text
required = conservative footprint + max(512 MiB, 5% of capacity)
```

`fit` prints this `[A]` margin and the final required bytes under each target.

### What do `[M]`, `[R]`, `[E]`, `[A]`, and `[U]` mean?

- `[M]`: directly measured by a driver or OS counter.
- `[R]`: returned by a loader API.
- `[E]`: estimated from GGUF metadata, model math, or a remainder.
- `[A]`: a conservative vramwatch policy assumption.
- `[U]`: a value you supplied, such as `--vram 24GiB`.

The badges are part of console, JSON provenance fields, watch, and SVG reporting.

### Why does the KV cache look too large?

The default is f16. Tell vramwatch when the loader uses a quantized cache:

```sh
vramwatch fit MODEL --context 32768 --kv-cache-type q8_0
vramwatch watch --kv-cache-type q8_0
```

Supported preflight types are f32, f16/bf16, q8_0, q5_0/q5_1, and q4_0/q4_1.
vramwatch currently assumes K and V use the same type.

### Does fit support split GPUs or partial CPU offload?

Not yet. `conservative-v1` evaluates full model residency on each accelerator
independently. A tensor-split loader may fit a model that vramwatch says does not
fit one card. Conversely, a partially offloaded llama.cpp model may use less GPU
memory than the full GGUF size. Those require loader-specific planners.

### Why does `report` say accuracy is pending?

The saved prediction has not been matched to a resident model yet. Load the same
model with the same quant and context, then run `vramwatch watch` or
`vramwatch report` again. Pairing is deliberately strict and requires exactly one
resident model on the device so an unrelated allocation is not scored as the
prediction.

You can select a non-latest record with:

```sh
vramwatch report --prediction 0123456789abcdef
```

### Where is prediction history stored?

- Linux: `$XDG_STATE_HOME/vramwatch` or `~/.local/state/vramwatch`
- macOS: `~/Library/Application Support/vramwatch`
- Windows: `%LOCALAPPDATA%\vramwatch`

Override it with `VRAMWATCH_STATE_DIR`. Each prediction is a private JSON file;
there is no telemetry service. Use `fit --no-record` to disable persistence for a
single prediction.

### Is the SVG safe to share?

The SVG omits hostname, PID, PCI bus, serial number, local path, and signed URL
query fields. It includes the human-visible GPU/model identity, quant, context,
prediction ID, and accuracy because those are the purpose of the card.

The local ledger and raw `report --json` are not scrubbed; they can contain the
original model reference and should be treated as local diagnostic data.

### Does vramwatch phone home?

There is no account or telemetry. `watch` and normal `doctor` use local drivers and
loopback loader APIs. Remote `fit` necessarily contacts the selected Hugging Face
or Ollama registry for metadata. `doctor --online` makes explicit registry probes.

### What does Apple support mean when memory is unified?

On Apple silicon, capacity is Metal's recommended maximum working set and current
availability comes from reclaimable Mach VM pages, clamped to that budget. Output
uses `unified_memory`, not `dedicated_vram`. This is a planning budget shared with
the OS and applications, so it is more dynamic than a discrete card's VRAM.

Release binaries for macOS are built natively with Metal support. A custom
`CGO_ENABLED=0` build compiles, but its Apple Metal provider is intentionally
unavailable.

### `doctor` found the loader but cannot confirm acceleration

The loader endpoint is healthy, but neither the loader nor the driver showed GPU
memory for the resident model. Check the loader's CUDA, ROCm, Vulkan, or Metal
backend log and its layer-offload settings. If no model is resident, load one and
run doctor again.

### Can I use vramwatch without detected hardware?

Yes. Preflight a known budget with `fit --vram 24GiB`. For the live UI and provider
development, `watch --source demo` synthesizes growth and `watch --source
mock:path.json` replays a fixture.

### What output is stable for scripts?

`fit --json`, `doctor --json`, and `report --json` emit envelopes with
`schema_version: 1`. Fit exits `0` when any target fits, `3` when the prediction is
valid but no target fits, `2` for command usage, and `1` for operational errors
or an indeterminate hardware budget.

### Where did `predict`, `snapshot`, and `devices` go?

They were the exploratory 0.x surface. Use `fit`, `report`, and `doctor`
respectively. The old names return a direct migration message rather than silently
changing behavior.
