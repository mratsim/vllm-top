# sgtop → vllm-top metric mapping

How the [sgtop](https://github.com/mratsim/sgtop) metric vocabulary maps onto
vLLM's `/metrics` output. This is a maintainer/reference document for the port;
it is deliberately not in the user-facing README because the mapping is an
implementation detail of the port, not something a vLLM operator reads.

## Direct equivalents

| sgtop metric | vLLM equivalent | Notes |
|---|---|---|
| `sglang:realtime_tokens_total{mode="decode"}` | `vllm:generation_tokens_total` | Counter, decode lane |
| `sglang:realtime_tokens_total{mode="prefill_compute"}` | `vllm:prompt_tokens_by_source_total{source="local_compute"}` | Counter, prefill lane |
| `sglang:prefill_effective_tokens_total{mode="input"}` | `vllm:prompt_tokens_by_source_total{source="local_compute"}` | Cache-miss / computed lane |
| `sglang:prefill_effective_tokens_total{mode="device_hit"}` | `vllm:prompt_tokens_by_source_total{source="local_cache_hit"}` | Cache-hit lane |
| `sglang:num_running_reqs` | `vllm:num_requests_running` | Gauge |
| `sglang:num_queue_reqs` | `vllm:num_requests_waiting` | Gauge |
| `sglang:time_to_first_token_seconds` | `vllm:time_to_first_token_seconds` | Histogram |
| `sglang:inter_tokens_latency_seconds` | `vllm:inter_token_latency_seconds` | Histogram |
| `sglang:queue_time_seconds` | `vllm:request_queue_time_seconds` | Histogram |
| `sglang:prefill_total_tokens` | `vllm:request_prompt_tokens` | Histogram (prompt length, all) |
| `sglang:cache_hit_rate` | `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` | Derived ratio |
| `sglang:spec_accept_rate` | `vllm:spec_decode_num_accepted_tokens_total` | Derived |
| `sglang:spec_accept_length` | `vllm:spec_decode_num_draft_tokens_total` | Derived |
| `sglang:scheduler_process_cpu_seconds_total` | `vllm:scheduler_compute_seconds_total` | Counter, `{class="prefill"/"decode"}` |

`vllm:prompt_tokens_total` is deliberately **not** used for the prefill lane: it
counts prompt tokens from every source, cache hits included. A compaction or a
cached-prompt burst inflates it, and the stall detector (which requires a
positive prefill rate) would read that as a stall. Both the prefill lane and the
cache-miss lane therefore read the compute-only counter
`prompt_tokens_by_source_total{source="local_compute"}`.

## Approximate — relabeled honestly

These have a vLLM counter that is the closest match but not the same quantity,
so the UI label states what the counter actually measures.

| sgtop metric | vLLM equivalent | Honest label |
|---|---|---|
| `sglang:num_retracted_reqs` | `vllm:request_success_total{finished_reason="abort"}` | `abort/s` |
| `sglang:evicted_tokens_total` | `vllm:num_preemptions_total` | `preempt/s` |
| `sglang:http_responses_total{status_code="503"}` | `vllm:http_requests_total{status="5xx"}` | `5xx/s` |
| `sglang:hicache_dropped_tokens_total` | `vllm:kv_offload_allocation_failure_total` | `alloc fail/s` |

## Pool reconstruction

vLLM exposes the KV cache as a percentage, not absolute tokens, so the absolute
count is reconstructed:

- **KV pool** = `vllm:kv_cache_usage_perc` × `vllm:cache_config_info{kv_cache_size_tokens}`.
- **Host pool** = `vllm:kv_offload_cpu_cache_usage_perc` — vLLM reports only a
  usage percentage here, so this is the one cell that displays a percentage
  rather than an absolute count.

## No vLLM equivalent — dropped

| sgtop metric | Reason |
|---|---|
| `sglang:mamba_*` | Model has `mamba_cache_mode="none"` on this deployment |
| `sglang:swa_*` | Model has `swa_block_size="None"` on this deployment |
| `sglang:tokenizer_*` / `sglang:detokenizer_*` CPU | vLLM exposes no per-tokenizer CPU split |
| `sglang:decode_sum_seq_lens` | vLLM has no equivalent sequence-length sum |

## Parser compatibility

The Prometheus parser is unchanged from sgtop. It already handles vLLM's
format: family names containing colons (`vllm:...`), `_bucket` / `_sum` /
`_count` histogram triplets, and labeled gauges/counters. One robustness fix
was made during the port: many vLLM `# TYPE` families are declared but carry no
data record yet, and some summary metrics are represented as `_count`/`_sum`;
the parser treats these as absent rather than failing.
