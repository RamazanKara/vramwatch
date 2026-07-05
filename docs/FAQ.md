# FAQ

### Why does it say `estimated`?

Because that figure was derived, not measured. vramwatch does not hook the
CUDA/HIP allocator (yet), so weights and the KV cache are computed from the model’s
architecture and reported footprint. The device total/used/free and NVIDIA
per-process VRAM *are* measured. See [METHODOLOGY.md](METHODOLOGY.md) for exactly
what falls in each bucket.

### My KV cache looks 2× too big

You’re probably running a quantized KV cache and vramwatch is assuming f16.
Tell it the dtype:

```sh
vramwatch watch --kv-cache-type q8_0     # or q4_0, f32, bf16, ...
# or: export VRAMWATCH_KV_CACHE_TYPE=q8_0
```

With Ollama that’s whatever you set `OLLAMA_KV_CACHE_TYPE` to; with llama.cpp it’s
your `--cache-type-k` / `--cache-type-v`.

### The numbers don’t match `nvidia-smi`

The device total/used/free should match `nvidia-smi` closely, since vramwatch reads
those from it. The split within a process (weights vs KV vs compute) is what
`nvidia-smi` doesn’t provide and vramwatch estimates. If the *device* numbers
disagree, check that you’re looking at the same GPU index and that no other tool is
allocating between samples.

### `weights` looks wrong for my llama.cpp model

vramwatch uses the GGUF file size as the weights estimate, which only equals VRAM
weights when the whole model is offloaded to the GPU. If you ran with partial offload
(`-ngl` less than the layer count), the file size over-states GPU weights.

### Does AMD per-process VRAM work?

On Linux, yes. vramwatch reads it from `/proc/<pid>/fdinfo` (the same DRM
interface `nvtop`/`amdgpu_top` use), so it attributes VRAM to the real
`ollama`/`llama-server` process. It only sees processes you have permission to read,
so if your loader runs as another user (e.g. a systemd service), run vramwatch as
that user or as root. On Windows/macOS, AMD falls back to the loader’s reported VRAM.

### It didn’t detect my model / GPU

Run the diagnostic:

```sh
vramwatch devices
```

It lists which GPU tools (`nvidia-smi`, `rocm-smi`) and loaders (Ollama, llama.cpp)
were found. Common causes:

- **Ollama** isn’t serving on `127.0.0.1:11434`. Set `OLLAMA_HOST`.
- **llama.cpp** isn’t serving on `127.0.0.1:8080`. Set `LLAMACPP_HOST`.
- The model isn’t actually loaded (Ollama unloads idle models), so send it a request
  first.

### Does it support vLLM / MLX / TGI / LM Studio?

Not yet. Ollama and llama.cpp are supported today. vLLM, MLX and Apple Metal are
on the roadmap. A new loader is a small, self-contained contribution; see
[CONTRIBUTING.md](../CONTRIBUTING.md).

### Does it work without a GPU?

Yes, for trying it out. `--source demo` synthesises a card whose KV cache grows until
OOM, and `--source mock:PATH` replays a scenario JSON. These drive the exact same
attribution engine as live data.

### Does it phone home / need an account?

No. It shells out to local vendor tools and makes HTTP requests to loopback only.
Nothing leaves your machine.

### Can I get machine-readable output?

`snapshot --json` and `predict --json` emit structured JSON for scripting and
dashboards. `snapshot --svg` writes a shareable scorecard.

### How accurate is the OOM prediction?

The KV growth is exact for an f16/bf16/f32 cache (and a small conservative
over-estimate for a quantized one), so the *max context* estimate is good when
weights and overhead stay constant. It’s a planning number, not a guarantee: a
fragmenting allocator or a second process can still OOM you earlier.
