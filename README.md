# vllm-top

Live terminal dashboard for a [vLLM](https://docs.vllm.ai) inference server:
prefill/decode histograms, TTFT and cache-miss plots, latency percentiles,
and the health counters vLLM actually exposes. Portable binary, no storage,
no daemon. A port of [sgtop](https://github.com/mratsim/sgtop) from sglang's
metrics to vLLM's.

## Features

- **Rate histograms.** Prefill and decode tok/s share the full terminal
  width on a fixed 60s wall-clock axis, newest sample pinned to the right
  edge. Dashed gridlines at every 10/100/1k… multiple through the observed
  peak, anchored to the session peak.
- **Stall detection.** Decode-rate collapse below 25% of the running median
  while prefill is active and requests are in flight; red ticks mark the
  intervals on the decode and latency plots.
- **Memory pools.** KV (absolute tokens derived from the reported capacity
  × usage %) and the host CPU-KV-offload tier (usage %, the only reading
  vLLM reports). vLLM exposes no mamba/SWA pool, so those rows are gone.
- **Latency plots.** TTFT (p95 shaded) per interval over the 60s window,
  next to the cache-miss plot: computed prompt tokens/s from
  `prompt_tokens_by_source_total{source="local_compute"}` — the prompt text
  the prefix cache did not hold. Both carry red stall ticks.
- **Latency table.** TTFT, ITL and queue time as p50/p95/p99 of the focused
  window, from histogram bucket deltas.
- **Health counters.** Prefix-cache hit fraction, cached vs computed
  throughput, request preemptions/s, client aborts/s, 5xx/s, CPU-KV-offload
  store/load bytes/s and allocation failures, scheduler compute.
- **Speculative decoding.** Acceptance rate and accepted tokens per draft
  from the `spec_decode_*` counters.

## Quick start

```
cargo install --path .      # installs `vllm-top`
```

then:

```
vllm-top                                   # watches vLLM's default port (8000)
vllm-top --url http://localhost:8000 --interval 1.0
vllm-top --url https://gpu-box.home.example.org --insecure   # self-signed proxy
vllm-top --api-key sk-...                   # if the server uses an API key
vllm-top --once                             # one text snapshot, for scripts/cron
```

`--insecure` accepts any server certificate: the bearer token and the metrics
transit without TLS verification.

## Keys

| Key | Action |
|---|---|
| `q` / `Esc` | quit |
| `space` | pause scraping |
| `c` | toggle compact / full layout |
| `g` | toggle graphs |
| `1` `2` `3` | focus the 5s / 15s / 60s window (graphs stay 60s) |
| `t` | cycle theme (gruvbox, catppuccin, tokyonight) |
| `e` | explain screen |
| `?` | keymap |
| `+` / `-` | poll interval (clamped 0.5–10s) |
| `↑` `↓` | scroll the explain screen |

## Maintainers

The sglang → vLLM metric mapping used for the port is documented separately
in [`sgtop_vllm-top.md`](sgtop_vllm-top.md).

## Build

```sh
cargo build --release        # binary at target/release/vllm-top
cargo test                   # parser + model + formatter + render tests
```
