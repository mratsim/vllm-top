use std::time::Duration;

use crate::history::{counter_delta, hist_window_p95, History};
use crate::metrics::{Sample, SeriesKey};

pub const WINDOWS: [Duration; 3] = [
    Duration::from_secs(5),
    Duration::from_secs(15),
    Duration::from_secs(60),
];

pub const WIN_LABELS: [&str; 3] = ["5s", "15s", "60s"];

fn fam(name: &'static str) -> impl Fn(&SeriesKey) -> bool + Copy {
    move |k: &SeriesKey| k.name == name
}

fn fam_labeled(
    name: &'static str,
    lk: &'static str,
    lv: &'static str,
) -> impl Fn(&SeriesKey) -> bool + Copy {
    move |k: &SeriesKey| k.name == name && k.has_label(lk, lv)
}

/// vLLM decode (generation) throughput counter: cumulative generated tokens.
const GEN_TOTAL: &str = "vllm:generation_tokens_total";

fn decode_lane() -> impl Fn(&SeriesKey) -> bool + Copy {
    fam(GEN_TOTAL)
}

/// vLLM prompt-token counter, split by where the tokens came from.
/// `local_compute` is the prompt text the prefix cache did not hold
/// (the cache misses the device chewed) — the prefill-compute lane.
/// `prompt_tokens_total` counts cache hits too, so it is not a prefill
/// rate: a compaction or cached-prompt burst inflates it, and the stall
/// detector (which requires a positive prefill rate) would read that as
/// a stall. The prefill lane and the cache-miss lane therefore both read
/// the compute-only counter.
const PROMPT_BY_SOURCE: &str = "vllm:prompt_tokens_by_source_total";
const SOURCE_COMPUTE: &str = "local_compute";

fn prefill_lane() -> impl Fn(&SeriesKey) -> bool + Copy {
    fam_labeled(PROMPT_BY_SOURCE, "source", SOURCE_COMPUTE)
}

fn input_lane() -> impl Fn(&SeriesKey) -> bool + Copy {
    fam_labeled(PROMPT_BY_SOURCE, "source", SOURCE_COMPUTE)
}

/// `source` labels of the prompt-token counter that count as cache hits:
/// the prefix cache (local) and external KV transfer. The single
/// vocabulary the cached-rate lane reads.
const SOURCE_CACHED: &[&str] = &["local_cache_hit", "external_kv_transfer"];

/// Prompt tokens the cache served (prefix cache or external KV transfer).
fn any_hit_lane() -> impl Fn(&SeriesKey) -> bool + Copy {
    move |k: &SeriesKey| {
        k.name == PROMPT_BY_SOURCE && SOURCE_CACHED.iter().any(|m| k.has_label("source", m))
    }
}

/// One cache pool (KV / host) with its current usage ratio.
#[derive(Clone, Copy)]
pub struct Pool {
    pub name: &'static str,
    pub usage: f64,
    /// absolute counts backing the usage ratio, when the engine
    /// exposes them (engine-wide, summed across ranks)
    pub used: Option<f64>,
    pub total: Option<f64>,
    /// what the counts count: "tokens" for token pools.
    pub unit: &'static str,
}

/// Rate values indexed by window (5s / 15s / 60s).
pub type Triple = [Option<f64>; 3];

/// p50/p95/p99 for one metric.
#[derive(Clone, Copy, Default)]
pub struct Quantiles {
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

/// Quantiles for each window (5s / 15s / 60s).
pub type LatencyTriple = [Quantiles; 3];

#[derive(Clone, Copy, Default, Debug)]
pub struct Stalls {
    pub count: u32,
    pub seconds: f64,
}

/// One rate lane over the trailing 60s window: one rate per scrape
/// interval, with the interval's endpoint presence flags.
#[derive(Clone, Default)]
pub struct Lane {
    /// Rate in the lane's unit (tokens/s) per scrape interval. An interval
    /// whose series is missing from either endpoint sample pairs nothing
    /// and reads 0: data absence, never a measured collapse.
    pub vals: Vec<f64>,
    /// The interval's earlier sample lacks the series: a first appearance,
    /// a reappearance after an absence, or the return leg of a non-finite
    /// reading dropped at the parse boundary. Recorded for both lanes:
    /// every rate carries its gap provenance, and the stall fold consumes
    /// the decode lane's flags.
    pub absent_old: Vec<bool>,
    /// The interval's later sample lacks the series: the absence starting.
    pub absent_new: Vec<bool>,
}

/// The trailing 60s window in the shape the graphs draw, produced by one
/// pass over the ring (`scan_window`) and cached next to it.
#[derive(Clone, Default)]
pub struct Graphs {
    pub decode: Lane,
    pub prefill: Lane,
    /// flagged intervals (see the stall definition on `scan_window`)
    pub stall: Vec<bool>,
    /// Evicted device-KV tokens/s per scrape interval, under the rate lanes' gap-aware
    /// pairing; an interval with a positive value draws an eviction tick on the latency plots.
    pub evictions: Lane,
    /// Time-to-first-token per interval position, in seconds: the wait for the first token
    /// (queue + prefill). Cumulative bucket counters make each position's quantile the delta
    /// between the window's first sample and the position's later sample, so the newest point
    /// is the window's headline quantile. None while that span is too sparse to interpolate
    /// (`hist_window_p95`): a gap in the plot is absence of measured data, never a zero.
    pub ttft_p95: Vec<Option<f64>>,
    /// Computed prompt tokens/s per scrape interval: the effective-prefill
    /// counter's `input` mode, the prompt text the prefix cache did not hold
    /// (the cache misses the device chewed). An interval whose endpoint
    /// samples lack the counter family stores None, a gap in the plot:
    /// a zero bar means a present counter was measured at zero.
    pub cache_misses: Vec<Option<f64>>,
    /// Per-stream decode speed per interval: the interval's decode rate
    /// divided by its later sample's running count. None while idle.
    /// Also None when the decode series is missing from either endpoint
    /// sample (missing data never reads as a measured collapse).
    pub per_stream: Vec<Option<f64>>,
    /// interval durations in seconds, parallel to the vectors above
    pub dt: Vec<f64>,
}

/// Everything the UI needs for one draw: rates, quantiles, and pools
/// computed per call, plus the graphs and stall flags from the cached
/// per-push window scan (`History::scan`).
pub struct Derived {
    pub model: String,
    pub engine: String,
    pub running: Option<f64>,
    pub queue: Option<f64>,
    pub decode_rate: Triple,
    /// instantaneous decode rate (most recent scrape interval)
    pub decode_instant: Option<f64>,
    /// instantaneous prefill-compute rate
    pub prefill_instant: Option<f64>,
    pub prefill_rate: Triple,
    pub pools: Vec<Pool>,
    /// Prefix-cache hit fraction: rate of the effective-prefill counter's
    /// hit modes over all its modes (60s window); the engine's
    /// cache_hit_rate gauge is the fallback while the family is absent.
    pub cache_hit: Option<f64>,
    /// Cached and computed prompt throughput over the 60s window, in tok/s:
    /// the counter's hit modes summed, and its `input` mode, the cache misses
    /// the device chewed. Present once the counter family exists.
    pub cache_cached: Option<f64>,
    pub cache_computed: Option<f64>,
    pub spec_accept: Option<f64>,
    pub spec_accept_len: Option<f64>,
    pub ttft: LatencyTriple,
    pub itl: LatencyTriple,
    pub e2e: LatencyTriple,
    pub queue_time: LatencyTriple,
    pub prompt_len_p50: Option<f64>,
    pub prompt_len_p95: Option<f64>,
    /// Uncached (computed) prompt length p50/p95, tokens: the pair gloss's
    /// second half, the size of the work the prefix cache did not absorb.
    pub computed_p50: Option<f64>,
    pub computed_p95: Option<f64>,
    pub stalls: [Stalls; 3],
    pub evict_rate: Option<f64>,
    pub retract_rate: Option<f64>,
    pub http_503_rate: Option<f64>,
    /// L2/HiCache: per-tier prefill hit rates (fraction of prefill tokens
    /// served from device/host tier, windowed)
    pub l2_device: Option<f64>,
    pub l2_host: Option<f64>,
    /// host-tier traffic: tokens/s written back / read back
    pub l2_wb: Option<f64>,
    pub l2_rb: Option<f64>,
    /// device KV tokens/s destroyed without a host backup (should be 0)
    pub l2_drop: Option<f64>,
    /// generated tokens (sum) over the currently running requests, read
    /// from the engine's decode sequence-length sum
    pub gen_total: Option<f64>,
    /// average generation length (tokens) of currently running requests
    pub gen_progress: Option<f64>,
    pub cpu_tokenizer: Option<f64>,
    pub cpu_detokenizer: Option<f64>,
    pub cpu_scheduler: Option<f64>,
    pub graphs: Graphs,
    pub window_focus: usize,
}

fn rate_triple(h: &History, pred: impl Fn(&SeriesKey) -> bool + Copy) -> Triple {
    let mut out = [None; 3];
    for (i, w) in WINDOWS.iter().enumerate() {
        out[i] = h.rate_sum(pred, *w);
    }
    out
}

fn latency_triple(h: &History, name: &'static str) -> LatencyTriple {
    let mut out: LatencyTriple = Default::default();
    for (i, w) in WINDOWS.iter().enumerate() {
        out[i] = Quantiles {
            p50: h.hist_quantile(fam(name), 0.50, *w),
            p95: h.hist_quantile(fam(name), 0.95, *w),
            p99: h.hist_quantile(fam(name), 0.99, *w),
        };
    }
    out
}

/// Fold the shared stall flags into the 5s/15s/60s time budgets: each
/// window's trailing span counts the flagged intervals and their frozen
/// seconds (the oldest interval may be cut off mid-way).
fn fold_stalls(flags: &[bool], dt: &[f64]) -> [Stalls; 3] {
    let mut out = [Stalls::default(); 3];
    for (i, w) in WINDOWS.iter().enumerate() {
        let mut budget = w.as_secs_f64();
        let mut count = 0u32;
        let mut seconds = 0.0;
        for j in (0..dt.len()).rev() {
            if budget <= 0.0 {
                break;
            }
            let take = dt[j].min(budget);
            if flags[j] {
                count += 1;
                seconds += take;
            }
            budget -= dt[j];
        }
        out[i] = Stalls { count, seconds };
    }
    out
}

/// Session peak rates, accumulated by the scraper across the whole run.
/// `decode_single` is the fastest per-stream rate seen: an interval's decode
/// rate divided by the requests that were running during it.
#[derive(Default, Clone, Copy)]
pub struct Peaks {
    pub decode: Option<f64>,
    pub prefill: Option<f64>,
    pub decode_single: Option<f64>,
}

/// Session-sticky alarm latches for the health strip. Each channel flips
/// true the first time its per-interval rate reads nonzero and stays latched
/// for the rest of the session: a quiet alarm collapses into the dim
/// summary line, a fired one regains its full row immediately.
#[derive(Default, Clone, Copy)]
pub struct Alarms {
    pub abort: bool,
    pub http_5xx: bool,
    pub alloc_fail: bool,
}

/// Latch any alarm channel once a scrape interval measured a nonzero rate.
/// Runs beside `update_peaks` on every pushed sample.
pub fn update_alarms(alarms: &mut Alarms, h: &History) {
    let n = h.len();
    if n < 2 {
        return;
    }
    let (Some(prev), Some(cur)) = (h.get(n - 2), h.get(n - 1)) else {
        return;
    };
    let dt = (cur.t - prev.t).as_secs_f64();
    if dt <= 0.0 {
        return;
    }
    if tokens_had_reset(&prev.sample, &cur.sample) {
        // the engine restarted: the old process's alarms are not the new
        // process's alarms
        *alarms = Alarms::default();
        return;
    }
    macro_rules! latch {
        ($field:ident, $pred:expr) => {
            if counter_rate(&prev.sample, &cur.sample, $pred) / dt > 0.0 {
                alarms.$field = true;
            }
        };
    }
    latch!(abort, fam_labeled("vllm:request_success_total", "finished_reason", "abort"));
    latch!(http_5xx, fam_labeled("http_requests_total", "status", "5xx"));
    latch!(alloc_fail, fam("vllm:kv_offload_allocation_failure_total"));
}

/// Fold the most recent scrape interval into the session peaks. A counter
/// reset (engine restart) clears all session peaks, since the old process's
/// peaks are not this engine's.
/// True when a realtime-token counter dropped between the two samples:
/// the engine restarted and its counters began again, so session-sticky
/// state (peaks, alarm latches) belongs to the previous process.
fn tokens_had_reset(prev: &Sample, cur: &Sample) -> bool {
    // the engine restarted and its counters began again: either the
    // generation counter or the computed-prompt counter dropped
    let check = |name: &str, source: Option<&str>| {
        prev.simple
            .iter()
            .filter(|(k, _)| k.name == name && source.map_or(true, |s| k.has_label("source", s)))
            .any(|(k, v_old)| cur.simple.get(k).map(|v| v < v_old).unwrap_or(false))
    };
    check(GEN_TOTAL, None) || check(PROMPT_BY_SOURCE, Some(SOURCE_COMPUTE))
}

pub fn update_peaks(peaks: &mut Peaks, h: &History) {
    let n = h.len();
    if n < 2 {
        return;
    }
    let (Some(prev), Some(cur)) = (h.get(n - 2), h.get(n - 1)) else {
        return;
    };
    let dt = (cur.t - prev.t).as_secs_f64();
    if dt <= 0.0 {
        return;
    }
    if tokens_had_reset(&prev.sample, &cur.sample) {
        // the engine restarted: the old process's peaks are not this
        // engine's peaks
        *peaks = Peaks::default();
        return;
    }
    let decode = counter_rate(&prev.sample, &cur.sample, decode_lane()) / dt;
    let prefill = counter_rate(&prev.sample, &cur.sample, prefill_lane()) / dt;
    if decode > peaks.decode.unwrap_or(0.0) {
        peaks.decode = Some(decode);
    }
    if prefill > peaks.prefill.unwrap_or(0.0) {
        peaks.prefill = Some(prefill);
    }
    let running = cur.sample.gauge("vllm:num_requests_running").unwrap_or(0.0);
    if running > 0.0 {
        let single = decode / running;
        if single > peaks.decode_single.unwrap_or(0.0) {
            peaks.decode_single = Some(single);
        }
    }
}

fn counter_rate(old: &Sample, new: &Sample, of_series: impl Fn(&SeriesKey) -> bool + Copy) -> f64 {
    let mut delta = 0.0;
    for (k, v_new) in &new.simple {
        if of_series(k) {
            // a key absent from the earlier sample is a new series
            // (first appearance, or reappearance after a gap): never
            // pair it with a pre-gap sample, so no fabricated spike
            // reaches the peaks or the graph lanes
            let Some(v_old) = old.simple.get(k) else {
                continue;
            };
            delta += counter_delta(*v_old, *v_new);
        }
    }
    delta
}

impl Lane {
    /// Append one scrape interval: the paired counter deltas over `dt`
    /// become the rate, or 0 when the series is missing at one end.
    fn push_interval(
        &mut self,
        old: &Sample,
        new: &Sample,
        of_series: impl Fn(&SeriesKey) -> bool + Copy,
        dt: f64,
    ) {
        self.vals.push(counter_rate(old, new, of_series) / dt);
        self.absent_old.push(!old.simple.keys().any(of_series));
        self.absent_new.push(!new.simple.keys().any(of_series));
    }
}

/// One pass over the trailing 60s window, rebuilt once per scrape push,
/// cached next to the ring (`History::scan`): per-series interval rates
/// with their presence flags, the stall flags, and the graph lanes
/// UI draws. Banner counters and red graph ticks read this shared
/// pass, so no consumer re-derives stalls.
///
/// # Stall definition
///
/// An interval (a pair of consecutive ring samples) is stalled
/// whenever the engine was working through it without producing
/// decode output:
///
/// - window: the trailing 60s (`WINDOWS[2]`)
/// - median over: the decode rate (tokens/s) of every interval, gaps
///   included. The full-window median is the bar, so the window's busy
///   level sets it
/// - threshold: decode rate below 0.25 × that median, while the interval's
///   later sample reports requests running and prefill was computing
///   (a positive prefill rate)
///
/// Gaps: the parse boundary drops non-finite counter readings, so a NaN
/// blink is a sample whose series is absent. Unreadable data is a gap,
/// intended behavior rather than something to suppress. An interval
/// leading out of a gap pairs nothing (rate 0) and is flagged whenever
/// the series was carried by an earlier window sample: evidence
/// of a real gap (a reappearance, NaN blink, or ongoing absence),
/// never a first appearance. The running-count and prefill
/// conditions still apply, so the red tick lands on the position
/// of the unreadable sample. A series never present in an earlier window
/// sample (a first appearance) is not a stall: no collapse was ever
/// observed, and a leading interval's zero rate is absence.
/// An interval where the absence starts (the later sample lacks the series)
/// is not flagged either: nothing measurable collapsed. The same
/// per-pair absence rule drives the rate sums, so a counter that only
/// moves across gap intervals shows a zero real rate.
pub fn scan_window(h: &History) -> Graphs {
    let entries = h.window_entries(WINDOWS[2]);
    let mut decode = Lane::default();
    let mut prefill = Lane::default();
    let mut evictions = Lane::default();
    let mut running: Vec<f64> = Vec::new();
    let mut ttft_p95: Vec<Option<f64>> = Vec::new();
    let mut cache_misses: Vec<Option<f64>> = Vec::new();
    let mut per_stream: Vec<Option<f64>> = Vec::new();
    let mut dt = Vec::new();
    // gap intervals with evidence: the decode series existed in an earlier
    // sample of the window, so an absent predecessor is a reappearance
    // after a gap rather than a first appearance (never a stall)
    let mut gap_after_seen: Vec<bool> = Vec::new();
    let mut decode_seen = false;

    for pair in entries.windows(2) {
        let d = (pair[1].t - pair[0].t).as_secs_f64();
        if d <= 0.0 {
            continue;
        }
        decode.push_interval(&pair[0].sample, &pair[1].sample, decode_lane(), d);
        prefill.push_interval(&pair[0].sample, &pair[1].sample, prefill_lane(), d);
        evictions.push_interval(
            &pair[0].sample,
            &pair[1].sample,
            fam("vllm:num_preemptions_total"),
            d,
        );
        // a missing counter family stores a gap, never a measured zero:
        // the rate lanes keep their 0-on-absence convention for the stall
        // fold, while the misses plot has no such consumer
        let misses_absent = !pair[0].sample.simple.keys().any(input_lane())
            || !pair[1].sample.simple.keys().any(input_lane());
        cache_misses.push(if misses_absent {
            None
        } else {
            Some(counter_rate(&pair[0].sample, &pair[1].sample, input_lane()) / d)
        });
        let reappearance = decode.absent_old.last() == Some(&true) && decode_seen;
        gap_after_seen.push(reappearance);
        // the pair's earlier sample counts as evidence for the next interval
        decode_seen |= decode.absent_old.last() != Some(&true);
        let running_at = pair[1]
            .sample
            .gauge("vllm:num_requests_running")
            .unwrap_or(0.0);
        running.push(running_at);
        // an absent decode endpoint leaves the lane's rate at 0. Displayed
        // against a positive running count, that absence reads as a measured
        // collapse: missing data stores no speed instead
        let decode_missing =
            decode.absent_old.last() == Some(&true) || decode.absent_new.last() == Some(&true);
        per_stream.push(if running_at > 0.0 && !decode_missing {
            Some(decode.vals.last().copied().unwrap_or(0.0) / running_at)
        } else {
            None
        });
        // per-position latency: each point is the histogram bucket delta between the window's
        // first sample and this interval's later sample; the series starts sparse and ends
        // at the headline quantile
        let t95 = hist_window_p95(
            &entries[0].sample,
            &pair[1].sample,
            &fam("vllm:time_to_first_token_seconds"),
        );
        ttft_p95.push(t95);
        dt.push(d);
    }

    // the median bar spans the whole window, so the flags are set after
    // the walk, as an arithmetic fold over the collected lanes
    let median = median_of(&decode.vals);
    let stall: Vec<bool> = (0..decode.vals.len())
        .map(|i| {
            running[i] > 0.0
                && prefill.vals[i] > 0.0
                && (gap_after_seen[i]
                    || (!decode.absent_old[i]
                        && !decode.absent_new[i]
                        && decode.vals[i] < 0.25 * median))
        })
        .collect();

    Graphs {
        decode,
        prefill,
        stall,
        evictions,
        cache_misses,
        ttft_p95,
        per_stream,
        dt,
    }
}

fn median_of(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    match s.len() {
        0 => 0.0,
        n => s[n / 2],
    }
}

/// Read a label value from a single-series labeled gauge, e.g. the
/// `vllm:cache_config_info` config record (`kv_cache_size_tokens`).
fn gauge_label(sample: &Sample, name: &str, label: &str) -> Option<f64> {
    sample
        .simple
        .iter()
        .find(|(k, _)| k.name == name)
        .and_then(|(k, _)| k.labels.iter().find(|(lk, _)| lk == label))
        .and_then(|(_, lv)| lv.parse::<f64>().ok())
}

/// Host (CPU KV offload) pool usage; present once the engine reports it.
fn host_pool(h: &History) -> Option<f64> {
    // vLLM reports the offload cache as a percentage only, with no absolute
    // token count: the pool line renders the reading, not a count.
    h.gauge_pred(fam("vllm:kv_offload_cpu_cache_usage_perc"))
        .map(|u| u.clamp(0.0, 1.0))
}

pub fn derive(h: &History, window_focus: usize) -> Option<Derived> {
    let last = h.last()?;
    let running = h.gauge_pred(fam("vllm:num_requests_running"));
    let queue = h.gauge_pred(fam("vllm:num_requests_waiting"));

    // pools: KV always (absolute counts derived from the config capacity),
    // and the host (CPU KV offload) tier once the engine reports it.
    // vLLM exposes neither a mamba nor an SWA pool, so those rows are gone.
    let mut pools = Vec::new();
    if let Some(u) = h.gauge_pred(fam("vllm:kv_cache_usage_perc")) {
        // vLLM reports usage as a percentage; the absolute capacity lives in
        // the cache_config_info label record (one logical pool, not sharded
        // per rank like a sharded token pool).
        let total = gauge_label(
            &last.sample,
            "vllm:cache_config_info",
            "kv_cache_size_tokens",
        );
        let usage = u.clamp(0.0, 1.0);
        let used = total.map(|t| usage * t);
        pools.push(Pool {
            name: "KV",
            usage,
            used,
            total,
            unit: "tokens",
        });
    }
    if let Some(u) = host_pool(h) {
        pools.push(Pool {
            name: "host",
            usage: u,
            used: None,
            total: None,
            unit: "tokens",
        });
    }

    let gen_progress = None; // vLLM exposes no decode sequence-length sum

    let decode_rate = rate_triple(h, decode_lane());
    let prefill_rate = rate_triple(h, prefill_lane());
    let evict_rate = h.rate_sum(fam("vllm:num_preemptions_total"), WINDOWS[2]);
    let retract_rate = h.rate_sum(
        fam_labeled("vllm:request_success_total", "finished_reason", "abort"),
        WINDOWS[2],
    );
    let http_503_rate =
        h.rate_sum(fam_labeled("http_requests_total", "status", "5xx"), WINDOWS[2]);
    // vLLM exposes no tokenizer/detokenizer CPU split; the scheduler
    // compute counter (prefill + decode classes) is the closest analog.
    let cpu_scheduler = h.rate_sum(fam("vllm:scheduler_compute_seconds_total"), WINDOWS[2]);

    let ttft = latency_triple(h, "vllm:time_to_first_token_seconds");
    let itl = latency_triple(h, "vllm:inter_token_latency_seconds");
    let e2e = latency_triple(h, "vllm:e2e_request_latency_seconds");
    let queue_time = latency_triple(h, "vllm:request_queue_time_seconds");
    let prompt_len_p50 = h.hist_quantile(fam("vllm:request_prompt_tokens"), 0.50, WINDOWS[2]);
    let prompt_len_p95 = h.hist_quantile(fam("vllm:request_prompt_tokens"), 0.95, WINDOWS[2]);
    let computed_p50 = h.hist_quantile(
        fam("vllm:request_prefill_kv_computed_tokens"),
        0.50,
        WINDOWS[2],
    );
    let computed_p95 = h.hist_quantile(
        fam("vllm:request_prefill_kv_computed_tokens"),
        0.95,
        WINDOWS[2],
    );
    // prefix-cache hit fraction from the cumulative query/hit counters.
    let cache_cached = h.rate_sum(any_hit_lane(), WINDOWS[2]);
    let cache_computed = h.rate_sum(input_lane(), WINDOWS[2]);
    let cache_hit = match (
        h.gauge_pred(fam("vllm:prefix_cache_hits_total")),
        h.gauge_pred(fam("vllm:prefix_cache_queries_total")),
    ) {
        (Some(hits), Some(q)) if q > 0.0 => Some((hits / q).clamp(0.0, 1.0)),
        _ => None,
    };

    let graphs = h.scan().cloned()?;
    let stalls = fold_stalls(&graphs.stall, &graphs.dt);
    let decode_instant = graphs.decode.vals.last().copied();
    let prefill_instant = graphs.prefill.vals.last().copied();

    let mut model = String::from("?");
    let mut engine = String::from("?");
    if let Some(k) = last
        .sample
        .simple
        .keys()
        .find(|k| k.name == "vllm:num_requests_running")
    {
        for (lk, lv) in &k.labels {
            match lk.as_str() {
                "model_name" => model = lv.clone(),
                "engine" => engine = lv.clone(),
                _ => {}
            }
        }
    }

    // spec-decode acceptance from the cumulative counters: the fraction of
    // drafted tokens accepted, and accepted tokens per draft.
    let spec_accept = match (
        h.gauge_pred(fam("vllm:spec_decode_num_accepted_tokens_total")),
        h.gauge_pred(fam("vllm:spec_decode_num_draft_tokens_total")),
    ) {
        (Some(a), Some(d)) if d > 0.0 => Some((a / d).clamp(0.0, 1.0)),
        _ => None,
    };
    let spec_accept_len = match (
        h.gauge_pred(fam("vllm:spec_decode_num_accepted_tokens_total")),
        h.gauge_pred(fam("vllm:spec_decode_num_drafts_total")),
    ) {
        (Some(a), Some(d)) if d > 0.0 => Some(a / d),
        _ => None,
    };

    Some(Derived {
        model,
        engine,
        running,
        queue,
        decode_rate,
        prefill_rate,
        pools,
        cache_hit,
        cache_cached,
        cache_computed,
        spec_accept,
        spec_accept_len,
        ttft,
        itl,
        e2e,
        queue_time,
        prompt_len_p50,
        prompt_len_p95,
        computed_p50,
        computed_p95,
        stalls,
        evict_rate,
        retract_rate,
        http_503_rate,
        l2_device: None,
        l2_host: None,
        l2_wb: h.rate_sum(fam("vllm:kv_offload_store_bytes_total"), WINDOWS[2]),
        l2_rb: h.rate_sum(fam("vllm:kv_offload_load_bytes_total"), WINDOWS[2]),
        l2_drop: h.rate_sum(fam("vllm:kv_offload_allocation_failure_total"), WINDOWS[2]),
        gen_total: None,
        gen_progress,
        cpu_tokenizer: None,
        cpu_detokenizer: None,
        cpu_scheduler,
        decode_instant,
        prefill_instant,
        graphs,
        window_focus,
    })
}
