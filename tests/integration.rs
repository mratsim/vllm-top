use std::time::{Duration, Instant};

use vllm_top::derive::{derive, WINDOWS};
use vllm_top::history::{quantile_from_buckets, History, RING_CAP};
use vllm_top::metrics::{parse, Sample, SeriesKey, BUCKET_CAP, SERIES_CAP};

fn fixture() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/live.txt"
    ))
    .unwrap()
}

fn key(name: &str, labels: &[(&str, &str)]) -> SeriesKey {
    SeriesKey {
        name: name.into(),
        labels: labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

#[test]
fn parses_live_fixture() {
    let s = parse(&fixture()).unwrap();
    let running = key(
        "vllm:num_requests_running",
        &[("engine", "0"), ("model_name", "DeepSeek-V4-Flash-Vision-Exp")],
    );
    // two requests were running at scrape time
    assert_eq!(s.simple.get(&running), Some(&2.0));
    let usage = s.gauge("vllm:kv_cache_usage_perc").unwrap();
    assert!((0.0..=1.0).contains(&usage));

    // histogram ladder: ITL — bucket counts must match the emitted _count
    let itl = &s.hist[&key(
        "vllm:inter_token_latency_seconds",
        &[("engine", "0"), ("model_name", "DeepSeek-V4-Flash-Vision-Exp")],
    )];
    let expected_count: u64 = fixture()
        .lines()
        .find(|l| l.starts_with("vllm:inter_token_latency_seconds_count"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| v as u64)
        .expect("fixture must contain an ITL _count line");
    assert_eq!(itl.count, expected_count);
    assert_eq!(itl.le.first().unwrap().to_string(), "0.01");
    assert!(itl.le.last().unwrap().is_infinite());
    assert_eq!(itl.counts.last().unwrap(), &(expected_count as f64));

    // series must exist for every # HELP family carrying readable data.
    // a family whose every record holds a non-finite value is no data:
    // it parses to no series, and the UI renders the missing marker
    // rather than a maximal value
    let mut family_has_finite_record: std::collections::BTreeMap<String, bool> = Default::default();
    for l in fixture().lines() {
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let name = l
            .split('{')
            .next()
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap();
        let base = name
            .strip_suffix("_bucket")
            .or_else(|| name.strip_suffix("_sum"))
            .or_else(|| name.strip_suffix("_count"))
            .unwrap_or(name);
        let finite = l
            .split_whitespace()
            .last()
            .and_then(|v| v.parse::<f64>().ok())
            .is_some_and(|v| v.is_finite());
        *family_has_finite_record
            .entry(base.to_string())
            .or_insert(false) |= finite;
    }
    let expected: std::collections::BTreeSet<String> = fixture()
        .lines()
        .filter(|l| l.starts_with("# HELP"))
        .filter_map(|l| l.split_whitespace().nth(2))
        // a family declared in a # TYPE but with no data record parses to
        // no series: exclude it, like a family whose every record is
        // non-finite. vLLM declares many counters/histograms it has not
        // yet emitted a reading for.
        .filter(|fam| family_has_finite_record.get(*fam).copied().unwrap_or(false))
        .map(String::from)
        .collect();
    // the parser folds histogram families to their base name but represents
    // summary families (_sum/_count, no _bucket) as separate series; fold
    // those suffixes back so summary bases match their # HELP name.
    let fold = |n: &str| {
        n.strip_suffix("_count")
            .or_else(|| n.strip_suffix("_sum"))
            .or_else(|| n.strip_suffix("_bucket"))
            .unwrap_or(n)
            .to_string()
    };
    let families: std::collections::BTreeSet<String> = s
        .simple
        .keys()
        .chain(s.hist.keys())
        .map(|k| fold(&k.name))
        .collect();
    assert_eq!(
        families, expected,
        "parsed families must match the HELP families carrying readable data"
    );
}

#[test]
fn parses_labels_with_escapes_and_timestamps() {
    let s = parse(
        "# TYPE m:gauge gauge\nm:gauge 3.5\nm:gauge{a=\"x\",b=\"say \\\"hi\\\"\"} 4.5 1234567890\n",
    )
    .unwrap();
    assert_eq!(s.gauge("m:gauge"), Some(3.5)); // label-less series
    let k = SeriesKey {
        name: "m:gauge".into(),
        labels: vec![("a".into(), "x".into()), ("b".into(), "say \"hi\"".into())],
    };
    assert_eq!(s.simple.get(&k), Some(&4.5));
}

#[test]
fn label_values_with_braces_and_escapes_parse_fully() {
    // raw payload lines: `\\` in Rust source is one backslash in the payload
    let body = "# TYPE m gauge\n\
        m{v=\"a{b}\"} 1\n\
        m{v=\"a\\\"{b}\\\"\"} 2\n\
        m{v=\"a\\\\\"} 3\n";
    let s = parse(body).unwrap();
    assert_eq!(s.simple.get(&key("m", &[("v", "a{b}")])), Some(&1.0));
    // escaped quotes survive unescaping: payload value is a"{b}"
    assert_eq!(s.simple.get(&key("m", &[("v", "a\"{b}\"")])), Some(&2.0));
    // escaped backslash: payload value is a\
    assert_eq!(s.simple.get(&key("m", &[("v", "a\\")])), Some(&3.0));
}

// A record whose label quote never closes is dropped like any other
// malformed line: there is no reliable label block to recover, and
// scanning for one must neither panic nor hang.
#[test]
fn record_with_unclosed_quote_is_dropped() {
    let body = "# TYPE m gauge\nm{v=\"open 4\nm{ok=\"x\"} 5\n";
    let s = parse(body).unwrap();
    assert_eq!(s.simple.len(), 1);
    assert_eq!(s.simple.get(&key("m", &[("ok", "x")])), Some(&5.0));
}

#[test]
fn series_cap_admits_payloads_under_the_limit() {
    // 5000 series: well under the cap, exercises the same counting path
    let mut body = String::from("# TYPE m:g gauge\n");
    for i in 0..5000 {
        body.push_str(&format!("m:g{{id=\"{i}\"}} 1\n"));
    }
    let s = parse(&body).unwrap();
    assert_eq!(s.simple.len(), 5000);
}

// One series past the cap rejects the entire payload: the error names
// the cap and the observed count, and no partial sample is returned.
#[test]
fn series_cap_rejects_oversized_payload_whole() {
    let mut body = String::from("# TYPE m:g gauge\n");
    for i in 0..=SERIES_CAP {
        body.push_str(&format!("m:g{{id=\"{i}\"}} 1\n"));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: {} series (cap {SERIES_CAP})",
            SERIES_CAP + 1
        )),
        "error message was: {err}"
    );
}

// Histogram families count as one series regardless of bucket-ladder length:
// SERIES_CAP - 1 gauges plus one 5-record histogram family
// sits exactly at the cap and must be accepted.
#[test]
fn series_cap_counts_histogram_families_not_bucket_lines() {
    let mut body = String::from("# TYPE m:g gauge\n# TYPE m:h histogram\n");
    for i in 0..SERIES_CAP - 1 {
        body.push_str(&format!("m:g{{id=\"{i}\"}} 1\n"));
    }
    body.push_str(
        "m:h_bucket{le=\"0.1\"} 1\n\
         m:h_bucket{le=\"1\"} 2\n\
         m:h_bucket{le=\"+Inf\"} 2\n\
         m:h_count 2\n\
         m:h_sum 1.0\n",
    );
    let s = parse(&body).unwrap();
    assert_eq!(s.simple.len() + s.hist.len(), SERIES_CAP);
}

// One histogram family past the bucket cap rejects the entire payload,
// like the series cap: the error names the cap and the observed count, and
// no partial sample is returned.
#[test]
fn bucket_cap_rejects_family_over_the_limit() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..=BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: m:h has {} buckets (cap {BUCKET_CAP})",
            BUCKET_CAP + 1
        )),
        "error message was: {err}"
    );
}

// The bucket-cap bail renders the full SeriesKey through Display.
// This payload carries a non-`le` label so the labeled branch
// (brace open, separators, closing brace) is exercised,
// not just the bare family name.
#[test]
fn bucket_cap_error_names_the_full_labeled_family() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..=BUCKET_CAP {
        body.push_str(&format!(
            "m:h_bucket{{mode=\"decode\",le=\"{}\"}} 1\n",
            i as f64 * 0.001
        ));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: m:h{{mode=\"decode\"}} has {} buckets (cap {BUCKET_CAP})",
            BUCKET_CAP + 1
        )),
        "error message was: {err}"
    );
}

#[test]
// a ladder of exactly BUCKET_CAP buckets is admitted
fn bucket_cap_admits_family_at_the_limit() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    let s = parse(&body).unwrap();
    assert_eq!(s.hist[&key("m:h", &[])].le.len(), BUCKET_CAP);
}

#[test]
// the cap counts _bucket records only: a family at exactly the cap takes
// its mandatory trailing _sum/_count lines and parses, while a 257th
// _bucket record still rejects the whole sample.
fn bucket_cap_admits_trailing_sum_and_count_at_the_limit_but_not_a_257th_bucket() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    body.push_str("m:h_sum 1.0\nm:h_count 2\n");
    let s = parse(&body).unwrap();
    assert_eq!(s.hist[&key("m:h", &[])].le.len(), BUCKET_CAP);
    assert_eq!(s.hist[&key("m:h", &[])].count, 2);

    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..=BUCKET_CAP {
        body.push_str(&format!("m:h_bucket{{le=\"{}\"}} 1\n", i as f64 * 0.001));
    }
    let err = parse(&body).unwrap_err().to_string();
    assert!(
        err.contains(&format!(
            "endpoint too large: m:h has {} buckets (cap {BUCKET_CAP})",
            BUCKET_CAP + 1
        )),
        "error message was: {err}"
    );
}

// A hostile label value that trips the bucket cap reaches the error
// banner only after sanitization. The hostile value carries terminal
// escape sequences, the delete byte, carriage returns, and an unescaped newline:
// the stored message renders every control as visible notation, keeps
// raw control bytes out, and stays debuggable (family, labels, cap intact).
#[test]
fn hostile_label_in_cap_error_is_banner_safe_after_sanitization() {
    let mut body = String::from("# TYPE m:h histogram\n");
    for i in 0..=BUCKET_CAP {
        body.push_str(&format!(
            "m:h_bucket{{mode=\"\u{1b}[2J\u{7f}\r\\nx\",le=\"{}\"}} 1\n",
            i as f64 * 0.001
        ));
    }
    let err = parse(&body).unwrap_err().to_string();
    // the raw bail does carry the injected controls
    assert!(err.chars().any(|c| c == '\u{1b}'), "message was: {err:?}");
    let safe = vllm_top::scrape::sanitize_for_terminal(&err);
    assert!(!safe.chars().any(char::is_control), "message was: {safe:?}");
    for notation in ["\\x1b", "\\x7f", "\\r", "\\n"] {
        assert!(safe.contains(notation), "missing {notation}: {safe:?}");
    }
    // still debuggable: family, label name, and cap survive
    assert!(safe.contains("m:h{mode="));
    assert!(safe.contains("(cap 256)"));
}

#[test]
fn quantile_linear_interpolation() {
    // cumulative buckets: <=1s holds 10 obs, <=2s holds 20
    let b = vec![(1.0, 10.0), (2.0, 20.0), (f64::INFINITY, 20.0)];
    let p50 = quantile_from_buckets(&b, 0.5).unwrap(); // rank 10 -> boundary of first bucket
    assert!((p50 - 1.0).abs() < 1e-9);
    let p75 = quantile_from_buckets(&b, 0.75).unwrap(); // rank 15 -> 1.0 + 5/10 = 1.5
    assert!((p75 - 1.5).abs() < 1e-9);
    assert!(quantile_from_buckets(&[], 0.5).is_none());
    assert!(quantile_from_buckets(&[(1.0, 0.0), (2.0, 0.0)], 0.5).is_none());
}

fn one_gauge(name: &str, v: f64) -> Sample {
    parse(&format!("# TYPE {name} gauge\n{name} {v}\n")).unwrap()
}

#[test]
fn rates_and_reset_clamping() {
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, one_gauge("m:c", 100.0));
    h.push(t0 + Duration::from_secs(10), one_gauge("m:c", 150.0));
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    assert!((rate - 5.0).abs() < 1e-9);

    // counter reset: value drops to 2 -> treated as +2 over the span, not -148
    h.push(t0 + Duration::from_secs(20), one_gauge("m:c", 2.0));
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    // 0->10s: +50; 10->20s: reset clamps to +2 => 52/20 = 2.6
    assert!((rate - 2.6).abs() < 1e-9, "rate was {rate}");
}

#[test]
fn window_quantile_falls_back_when_sparse() {
    let mut h = History::default();
    let t0 = Instant::now();
    let body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 10\nm:lat_bucket{le=\"1\"} 19\nm:lat_bucket{le=\"+Inf\"} 20\n\
        m:lat_count 20\nm:lat_sum 5.0\n";
    h.push(t0, parse(body).unwrap());
    // identical -> zero new observations
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
    // sparse (<5 observations in window) -> snapshot fallback must kick in
    let q = h.hist_quantile(|k| k.name == "m:lat", 0.5, Duration::from_secs(60));
    let snapshot = h.hist_snapshot_quantile(&|k: &SeriesKey| k.name == "m:lat", 0.5);
    assert_eq!(q, snapshot);
    // snapshot p50: rank 10 sits exactly at the le=0.1 boundary
    assert!((snapshot.unwrap() - 0.1).abs() < 1e-9);
}

fn mix(decode: f64, prefill: f64) -> Sample {
    parse(&format!(
        "# TYPE vllm:generation_tokens_total counter\n\
         vllm:generation_tokens_total {decode}\n\
         # TYPE vllm:prompt_tokens_by_source_total counter\n\
         vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {prefill}\n\
         # TYPE vllm:num_requests_running gauge\n\
         vllm:num_requests_running 4\n"
    ))
    .unwrap()
}

#[test]
fn stall_signature_detected() {
    let mut h = History::default();
    let t0 = Instant::now();
    // calm decode traffic
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(40.0, 0.0));
    // prefill burst: decode freezes while prefill_compute surges
    h.push(t0 + Duration::from_secs(2), mix(40.0, 100.0));
    h.push(t0 + Duration::from_secs(3), mix(42.0, 200.0));
    let d = derive(&h, 0).unwrap();
    let stalls = d.stalls[0]; // 5s window
    assert!(
        stalls.count >= 1,
        "expected stall detection, got {stalls:?}"
    );
    assert!(stalls.seconds > 0.0, "stall seconds must be positive");

    let mut h = History::default();
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(40.0, 0.0));
    h.push(t0 + Duration::from_secs(2), mix(80.0, 0.0));
    h.push(t0 + Duration::from_secs(3), mix(120.0, 0.0));
    let d = derive(&h, 0).unwrap();
    assert_eq!(d.stalls[0].count, 0);
}

// A decode freeze with requests queued is a stall even when no prefill is
// computing: the engine has work waiting but is not converting it into
// decode output. The queue is what marks it — the same freeze with no queue
// and no prefill is an idle engine, not a stall. A queue with decode still
// advancing is pressure, not a block.
#[test]
fn stall_fires_on_queued_decode_without_prefill() {
    let build = |decode: f64, waiting: f64| {
        parse(&format!(
            "# TYPE vllm:generation_tokens_total counter\n\
             vllm:generation_tokens_total {decode}\n\
             # TYPE vllm:prompt_tokens_by_source_total counter\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} 0\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n\
             # TYPE vllm:num_requests_waiting gauge\n\
             vllm:num_requests_waiting {waiting}\n"
        ))
        .unwrap()
    };
    // decode runs 40 tok/s for three intervals, then freezes at 120 while a
    // queue of 3 forms. The freeze intervals are flagged; the busy ones are not.
    let mut h = History::default();
    let t0 = Instant::now();
    for (i, (decode, waiting)) in [
        (0.0, 0.0),
        (40.0, 0.0),
        (80.0, 0.0),
        (120.0, 3.0),
        (120.0, 3.0),
        (120.0, 3.0),
    ]
    .iter()
    .enumerate()
    {
        h.push(t0 + Duration::from_secs(i as u64), build(*decode, *waiting));
    }
    let d = derive(&h, 0).unwrap();
    assert_eq!(d.graphs.stall, vec![false, false, false, true, true]);
    assert_eq!(d.stalls[0].count, 2);

    // the same freeze with no queue and no prefill: an idle engine, not a stall
    let mut h = History::default();
    for (i, decode) in [0.0, 40.0, 80.0, 120.0, 120.0, 120.0]
        .iter()
        .enumerate()
    {
        h.push(
            t0 + Duration::from_secs(i as u64),
            build(*decode, 0.0),
        );
    }
    let d = derive(&h, 0).unwrap();
    assert_eq!(d.graphs.stall, vec![false, false, false, false, false]);
    assert_eq!(d.stalls[0].count, 0);
}

#[test]
fn windows_cover_expected_spans() {
    assert_eq!(WINDOWS.len(), 3);
    assert_eq!(WINDOWS[2], Duration::from_secs(60));
}

#[test]
fn session_peaks_accumulate_and_reset_on_restart() {
    use vllm_top::derive::{update_peaks, Peaks};

    let mut h = History::default();
    let mut peaks = Peaks::default();
    let t0 = Instant::now();
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(100.0, 500.0));
    update_peaks(&mut peaks, &h);
    h.push(t0 + Duration::from_secs(2), mix(240.0, 900.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, Some(140.0)); // 240-100 over the last 1s
    assert_eq!(peaks.prefill, Some(500.0)); // first interval 0->500 beats 400
                                            // single = 140 decode / 4 running
    assert_eq!(peaks.decode_single, Some(35.0));

    // lower follow-up intervals must not lower the peaks
    h.push(t0 + Duration::from_secs(3), mix(250.0, 910.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, Some(140.0));
    assert_eq!(peaks.decode_single, Some(35.0));

    // counter reset (server restart) clears the session peaks
    h.push(t0 + Duration::from_secs(4), mix(3.0, 2.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, None);
    assert_eq!(peaks.prefill, None);
    assert_eq!(peaks.decode_single, None);
}

#[test]
fn histogram_bucket_without_le_is_skipped_not_fatal() {
    let body = "\
# TYPE m:lat histogram
m:lat_count{mode=\"decode\"} 2
m:lat_bucket{mode=\"decode\"} 1
m:lat_bucket{mode=\"decode\",le=\"NaN\"} 5
m:lat_bucket{mode=\"decode\",le=\"10\"} 2
m:other_gauge 7
";
    let sample = parse(body).unwrap();
    let key = key("m:lat", &[("mode", "decode")]);
    let h = sample.hist.get(&key).expect("histogram family present");
    // only the bucket with a usable `le` survives
    assert_eq!(h.le, vec![10.0]);
    assert_eq!(h.counts, vec![2.0]);
    assert_eq!(h.count, 2);
    assert_eq!(sample.simple.len(), 1);
}

#[test]
fn interval_rejects_non_finite() {
    use clap::Parser as _;
    use vllm_top::args::Args as _Args;
    for bad in ["nan", "NaN", "inf", "-inf", "infinity"] {
        let res = _Args::try_parse_from(["vllm-top", &format!("--interval={bad}")]);
        assert!(res.is_err(), "--interval {bad} must be rejected");
        let msg = format!("{}", res.unwrap_err().render());
        assert!(msg.contains("finite"), "error for {bad}: {msg}");
    }
    for ok in ["0.5", "10", "1.5"] {
        assert!(
            _Args::try_parse_from(["vllm-top", "--interval", ok]).is_ok(),
            "--interval {ok} must be accepted"
        );
    }
}

// ---- display hardening: non-finite gauges, pool visibility ----

fn host_pool_sample(usage: Option<f64>) -> Sample {
    let usage_line = match usage {
        Some(v) => format!("vllm:kv_offload_cpu_cache_usage_perc {v}\n"),
        None => String::new(),
    };
    parse(&format!(
        "# TYPE vllm:kv_offload_cpu_cache_usage_perc gauge\n\
         {usage_line}"
    ))
    .unwrap()
}

// NaN and ±Inf gauge values are unreadable data, not extreme readings:
// skip the record like any other malformed line, leaving the series
// absent so consumers see no data, never a maximal-looking value.
#[test]
fn non_finite_gauge_values_read_as_no_data() {
    let body = "# TYPE m:g gauge\n\
                m:g NaN\n\
                m:g{r=\"0\"} +Inf\n\
                m:g{r=\"1\"} -Inf\n\
                m:g{r=\"2\"} 7.0\n";
    let s = parse(body).unwrap();
    assert_eq!(
        s.gauge("m:g"),
        Some(7.0),
        "only the finite reading may survive"
    );
    assert_eq!(s.simple.len(), 1);
}

// A NaN reading inside a windowed series must produce the same derived
// shape as the series being absent altogether: same gauge lookup, same
// pool membership.
#[test]
fn nan_reading_in_a_windowed_series_has_the_shape_of_a_missing_sample() {
    let t0 = Instant::now();
    let mut h_nan = History::default();
    h_nan.push(t0, host_pool_sample(Some(0.5)));
    h_nan.push(
        t0 + Duration::from_secs(1),
        host_pool_sample(Some(f64::NAN)),
    );
    let mut h_missing = History::default();
    h_missing.push(t0, host_pool_sample(Some(0.5)));
    h_missing.push(t0 + Duration::from_secs(1), host_pool_sample(None));

    let usage = |k: &SeriesKey| k.name == "vllm:kv_offload_cpu_cache_usage_perc";
    assert_eq!(
        h_nan.gauge_pred(usage),
        h_missing.gauge_pred(usage),
        "a non-finite reading must not look like a present value"
    );

    let d_nan = derive(&h_nan, 0).unwrap();
    let d_missing = derive(&h_missing, 0).unwrap();
    assert_eq!(
        d_nan.pools.len(),
        d_missing.pools.len(),
        "pool membership must match the absent-series case"
    );
}

// A completely full optional pool (zero available slots, all used) has
// data and must keep rendering at exactly 100% usage.
#[test]
fn host_pool_at_full_capacity_still_renders() {
    let body = "\
# TYPE vllm:kv_offload_cpu_cache_usage_perc gauge
vllm:kv_offload_cpu_cache_usage_perc 1.0
";
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
    let d = derive(&h, 0).unwrap();
    let host = d
        .pools
        .iter()
        .find(|p| p.name == "host")
        .expect("full host pool must keep rendering");
    assert_eq!(host.usage, 1.0);
}

// The visibility gate is "ever had data", not "ever existed": a pool
// reporting zero capacity and zero usage stays hidden.
#[test]
fn host_pool_stays_hidden_without_an_offload_reading() {
    let body = "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 2\n";
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
    let d = derive(&h, 0).unwrap();
    assert!(
        d.pools.iter().all(|p| p.name != "host"),
        "a host pool with no offload reading must stay hidden"
    );
}

// A usage-ratio gauge that blinks non-finite mid-window must not hide
// a pool that has reported data: visibility follows the ever-had-data
// rule, and the usage shown falls back to the latest in-window reading.
#[test]
fn host_pool_reads_a_nan_reading_as_absence() {
    let t0 = Instant::now();
    let live = "# TYPE vllm:kv_offload_cpu_cache_usage_perc gauge\nvllm:kv_offload_cpu_cache_usage_perc 0.5\n";
    let blink = live.replace(
        "vllm:kv_offload_cpu_cache_usage_perc 0.5",
        "vllm:kv_offload_cpu_cache_usage_perc NaN",
    );
    // a NaN usage reading is dropped at the parse boundary, so the
    // latest sample's pool membership matches the absent-series case
    let mut h_blink = History::default();
    h_blink.push(t0, parse(live).unwrap());
    h_blink.push(t0 + Duration::from_secs(1), parse(&blink).unwrap());
    let d_blink = derive(&h_blink, 0).unwrap();
    assert!(
        d_blink.pools.iter().all(|p| p.name != "host"),
        "a NaN offload reading must read as absence"
    );
}

// ---- windowed-delta semantics: reappearance, label order, ladder pairing ----

fn mix_without_decode(prefill: f64) -> Sample {
    parse(&format!(
        "# TYPE vllm:prompt_tokens_by_source_total counter\n\
         vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {prefill}\n\
         # TYPE vllm:num_requests_running gauge\n\
         vllm:num_requests_running 4\n"
    ))
    .unwrap()
}

fn counter_with_gap(values: [Option<f64>; 5]) -> History {
    let mut h = History::default();
    let t0 = Instant::now();
    for (i, v) in values.iter().enumerate() {
        let mut body = String::from("# TYPE m:other gauge\nm:other 1\n");
        if let Some(v) = v {
            body.push_str(&format!("# TYPE m:c counter\nm:c {v}\n"));
        }
        h.push(
            t0 + Duration::from_secs(i as u64 * 10),
            parse(&body).unwrap(),
        );
    }
    h
}

// A counter that vanishes for one or more scrapes and later reappears
// counts as a new series from the reappearance point. Pairing the first
// reappearance sample with a pre-gap sample would fabricate a spike
// (higher value) or a fake reset (lower value).
#[test]
fn reappearance_after_gap_is_not_paired_with_the_pre_gap_sample() {
    let h = counter_with_gap([Some(100.0), Some(150.0), None, Some(160.0), Some(220.0)]);
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    // paired deltas 50 + 60 over 40s (the reappearance interval adds 0)
    assert!((rate - 2.75).abs() < 1e-9, "rate was {rate}");
}

// A reappearance whose value is lower than the pre-gap reading follows
// the same rule: no pairing across the gap, and the next interval
// measures against the reappearance value itself.
#[test]
fn reappearance_after_gap_with_lower_value_starts_fresh() {
    let h = counter_with_gap([Some(100.0), Some(150.0), None, Some(40.0), Some(90.0)]);
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    // paired deltas 50 + 50 over 40s
    assert!((rate - 2.5).abs() < 1e-9, "rate was {rate}");
}

// Flagged intervals must be exactly the intervals whose earlier sample
// lacks the series, and every other interval must keep its true delta.
// The decoded rate vector below pins those positions exactly.
#[test]
fn reappearance_intervals_are_exactly_those_after_an_absent_predecessor() {
    let mut h = History::default();
    let t0 = Instant::now();
    // decode counter absent at entry 2, present everywhere else
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(10.0, 0.0));
    h.push(t0 + Duration::from_secs(2), mix_without_decode(0.0));
    h.push(t0 + Duration::from_secs(3), mix(30.0, 0.0));
    h.push(t0 + Duration::from_secs(4), mix(60.0, 0.0));
    let d = derive(&h, 0).unwrap();
    // interval 2 (reappearance) contributes 0, the vanished interval has
    // nothing to sum, and the others keep their true deltas
    let expected = [10.0, 0.0, 0.0, 30.0];
    assert_eq!(
        d.graphs.decode.vals.len(),
        expected.len(),
        "decode lanes were {:?}",
        d.graphs.decode.vals
    );
    for (i, (got, want)) in d.graphs.decode.vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "decode lane {i} was {got}, expected {want} (lanes {:?})",
            d.graphs.decode.vals
        );
    }
}

// Session peaks must not record the fabricated rate of a reappearance
// interval: the interval that brings the series back contributes no rate.
#[test]
fn reappearance_does_not_record_a_peak_spike() {
    use vllm_top::derive::{update_peaks, Peaks};

    let mut h = History::default();
    let mut peaks = Peaks::default();
    let t0 = Instant::now();
    h.push(t0, mix(0.0, 0.0));
    h.push(t0 + Duration::from_secs(1), mix(100.0, 0.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(peaks.decode, Some(100.0));
    h.push(t0 + Duration::from_secs(2), mix_without_decode(0.0));
    update_peaks(&mut peaks, &h);
    h.push(t0 + Duration::from_secs(3), mix(400.0, 0.0));
    update_peaks(&mut peaks, &h);
    assert_eq!(
        peaks.decode,
        Some(100.0),
        "the reappearance interval must not feed the session peak"
    );
}

// Two scrapes of the same metric whose labels are emitted in different
// orders are one series, not two: label order must not affect identity,
// or the "new" series fabricates a full-value delta on its first pair.
#[test]
fn label_emit_order_does_not_fork_series_identity() {
    let a = parse("# TYPE m:c counter\nm:c{a=\"x\",b=\"y\"} 100\n").unwrap();
    let b = parse("# TYPE m:c counter\nm:c{b=\"y\",a=\"x\"} 150\n").unwrap();
    let key_a = a.simple.keys().next().unwrap().clone();
    assert!(
        b.simple.contains_key(&key_a),
        "reordered labels must resolve to the same series key"
    );
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, a);
    h.push(t0 + Duration::from_secs(10), b);
    let rate = h
        .rate_sum(|k| k.name == "m:c", Duration::from_secs(60))
        .unwrap();
    assert!((rate - 5.0).abs() < 1e-9, "rate was {rate}");
}

// Histogram bucket counts must pair on the `le` boundary value, never
// on list position: a bucket inserted mid-ladder between two samples
// shifts every later index pairing, and the shifted deltas can stay
// non-negative, so only boundary-keyed pairing keeps the quantile sane.
#[test]
fn mid_ladder_bucket_insertion_pairs_quantiles_by_boundary() {
    let mut h = History::default();
    let t0 = Instant::now();
    let old_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 2\nm:lat_bucket{le=\"1\"} 19\nm:lat_bucket{le=\"+Inf\"} 20\n\
        m:lat_count 20\nm:lat_sum 9.0\n";
    let new_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 7\nm:lat_bucket{le=\"0.5\"} 19\n\
        m:lat_bucket{le=\"1\"} 29\nm:lat_bucket{le=\"+Inf\"} 30\n\
        m:lat_count 30\nm:lat_sum 14.0\n";
    h.push(t0, parse(old_body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(new_body).unwrap());
    let p50 = h
        .hist_quantile(|k| k.name == "m:lat", 0.5, Duration::from_secs(60))
        .unwrap();
    let p95 = h
        .hist_quantile(|k| k.name == "m:lat", 0.95, Duration::from_secs(60))
        .unwrap();
    // shared-boundary window deltas: +5 at le=0.1, +10 at le=1
    // p50 rank 5 sits exactly on the le=0.1 boundary, p95 rank 9.5
    // interpolates inside the (0.1, 1] bucket
    assert!((p50 - 0.1).abs() < 1e-9, "p50 was {p50}");
    assert!((p95 - 0.91).abs() < 1e-9, "p95 was {p95}");
}

// A bucket with an `le` bound of NaN cannot be ordered, so parse must
// drop it: a retained NaN bound wedges the boundary-keyed delta walk
// and silently truncates deltas at later boundaries. The +Inf bound
// is the standard top bucket and stays valid.
#[test]
fn nan_le_bound_is_dropped_and_quantiles_stay_sane() {
    let old_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 10\nm:lat_bucket{le=\"NaN\"} 15\n\
        m:lat_bucket{le=\"1\"} 19\nm:lat_bucket{le=\"+Inf\"} 20\n\
        m:lat_count 20\nm:lat_sum 9.0\n";
    let new_body = "# TYPE m:lat histogram\n\
        m:lat_bucket{le=\"0.1\"} 14\nm:lat_bucket{le=\"1\"} 29\n\
        m:lat_bucket{le=\"+Inf\"} 30\nm:lat_count 30\nm:lat_sum 14.0\n";
    let old = parse(old_body).unwrap();
    let key = old.hist.keys().next().unwrap().clone();
    let h = &old.hist[&key];
    // only usable bounds survive: the NaN bound must not be stored
    assert_eq!(h.le, vec![0.1, 1.0, f64::INFINITY]);
    assert_eq!(h.counts, vec![10.0, 19.0, 20.0]);

    let mut hist = History::default();
    let t0 = Instant::now();
    hist.push(t0, old);
    hist.push(t0 + Duration::from_secs(1), parse(new_body).unwrap());
    // shared-boundary deltas: +4 at le=0.1, +10 at le=1
    // p50 rank 5 interpolates inside the (0.1, 1] bucket
    let p50 = hist
        .hist_quantile(|k| k.name == "m:lat", 0.5, Duration::from_secs(60))
        .unwrap();
    assert!((p50 - 0.25).abs() < 1e-9, "p50 was {p50}");
}

// ---- unified stall pass: one flag set feeds the banner and the graphs ----

/// One scrape of the stall fixture holding decode and prefill
/// counters, a running gauge, and an optional helper counter `m:c`.
/// A None decode reading means an absent series: the shape left
/// behind when the parse boundary drops a non-finite value.
fn stall_sample(decode: Option<f64>, prefill: f64, running: f64, helper: Option<f64>) -> Sample {
    let mut body = String::from("# TYPE vllm:generation_tokens_total counter\n");
    if let Some(v) = decode {
        body.push_str(&format!("vllm:generation_tokens_total {v}\n"));
    }
    body.push_str(&format!(
        "# TYPE vllm:prompt_tokens_by_source_total counter\n\
         vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {prefill}\n\
         # TYPE vllm:num_requests_running gauge\n\
         vllm:num_requests_running {running}\n\
         # TYPE m:c counter\n"
    ));
    if let Some(v) = helper {
        body.push_str(&format!("m:c {v}\n"));
    }
    parse(&body).unwrap()
}

fn stall_history(samples: &[Sample]) -> History {
    let mut h = History::default();
    let t0 = Instant::now();
    for (i, s) in samples.iter().enumerate() {
        h.push(t0 + Duration::from_secs(i as u64), s.clone());
    }
    h
}

// The flagged tick positions are exactly the intervals whose predecessor
// sample lacks the decode series. An unreadable or absent reading is itself a gap:
// the interval leading out of it is flagged outright while prefill
// is being computed. The hero counters fold the same flags into their time budgets,
// so banner counts and graph ticks can never disagree.
// A counter whose only advance lands on the flagged interval pairs
// nothing across the gap (zero real rate, no fabricated spike).
// One that moves between normal intervals shows its rate.
#[test]
fn stall_ticks_equal_intervals_whose_predecessor_is_absent() {
    let samples = [
        stall_sample(Some(0.0), 0.0, 4.0, Some(10.0)),
        stall_sample(Some(100.0), 0.0, 4.0, Some(10.0)),
        stall_sample(None, 0.0, 4.0, None),
        stall_sample(Some(140.0), 50.0, 4.0, Some(90.0)),
        stall_sample(Some(240.0), 50.0, 4.0, Some(90.0)),
    ];
    let absent = stall_history(&samples);
    // a NaN reading is dropped at the parse boundary, so it must flag
    // identically to the series being absent altogether
    let nan = stall_history(&[
        stall_sample(Some(0.0), 0.0, 4.0, Some(10.0)),
        stall_sample(Some(100.0), 0.0, 4.0, Some(10.0)),
        stall_sample(Some(f64::NAN), 0.0, 4.0, None),
        stall_sample(Some(140.0), 50.0, 4.0, Some(90.0)),
        stall_sample(Some(240.0), 50.0, 4.0, Some(90.0)),
    ]);

    for h in [&absent, &nan] {
        let d = derive(h, 0).unwrap();
        assert_eq!(
            d.graphs.stall,
            vec![false, false, true, false],
            "flags must mark exactly the interval leading out of the unreadable scrape"
        );
        // the hero counters fold the same flags: one flagged interval
        // inside each untruncated budget, one second frozen
        assert_eq!(d.stalls[0].count, 1);
        assert_eq!(d.stalls[2].count, 1);
        assert!((d.stalls[2].seconds - 1.0).abs() < 1e-9);
    }
    assert_eq!(
        derive(&absent, 0).unwrap().graphs.stall,
        derive(&nan, 0).unwrap().graphs.stall,
        "a non-finite blink and an absent series must flag identically"
    );

    // the helper counter's only advance lands on the flagged interval:
    // the per-pair absence rule pairs nothing there, so the real rate is zero
    // (measured stillness, not the fabricated jump across the gap)
    let helper_rate = absent.rate_sum(|k| k.name == "m:c", Duration::from_secs(60));
    assert_eq!(helper_rate, Some(0.0), "helper rate was {helper_rate:?}");
    // decode moves between normal intervals and shows its rate: 200
    // tokens over the 4s span
    let decode_rate = absent
        .rate_sum(|k| k.name == "vllm:generation_tokens_total", Duration::from_secs(60))
        .unwrap();
    assert!((decode_rate - 50.0).abs() < 1e-9, "rate was {decode_rate}");
}

// A first appearance is not a stall: the decode series materializing
// mid-window (cold start) has never collapsed, so no interval flags,
// even while requests run and prefill computes throughout. Its leading
// interval pairs nothing (rate 0 is absence), and no threshold arm
// reads that absence as a measured collapse either.
#[test]
fn first_decode_appearance_is_not_a_stall() {
    // prefill advances every interval, so the prefill-computing
    // condition holds wherever the running count is positive
    let h = stall_history(&[
        stall_sample(None, 0.0, 4.0, None),
        stall_sample(None, 50.0, 4.0, None),
        stall_sample(Some(0.0), 100.0, 4.0, None),
        stall_sample(Some(100.0), 150.0, 4.0, None),
        stall_sample(Some(200.0), 200.0, 4.0, None),
    ]);
    let d = derive(&h, 0).unwrap();
    assert_eq!(
        d.graphs.stall,
        vec![false, false, false, false],
        "a first appearance must not flag: {:?}",
        d.graphs.stall
    );
    // the busy fixture's own prefill lane confirms the condition held:
    // 50 tokens/s on every interval after the first
    assert_eq!(d.graphs.prefill.vals, vec![50.0, 50.0, 50.0, 50.0]);
    assert_eq!(d.stalls[0].count, 0);
    assert_eq!(d.stalls[2].count, 0);
}

// The stall bar is the median over the whole window, not the prefix
// median at each interval: a collapse right at the start is flagged
// by the graph ticks and the hero counters alike.
#[test]
fn stall_bar_is_the_full_window_median() {
    let h = stall_history(&[
        stall_sample(Some(0.0), 0.0, 4.0, None),
        stall_sample(Some(0.0), 50.0, 4.0, None),
        stall_sample(Some(100.0), 50.0, 4.0, None),
        stall_sample(Some(200.0), 50.0, 4.0, None),
    ]);
    let d = derive(&h, 0).unwrap();
    assert_eq!(d.graphs.stall, vec![true, false, false]);
    assert_eq!(d.stalls[0].count, 1);
}

// Each interval is judged by the running count it had: a collapse
// during an idle stretch is not flagged in hindsight by a later busy
// reading.
#[test]
fn stall_flags_use_each_intervals_own_running_count() {
    let h = stall_history(&[
        stall_sample(Some(100.0), 0.0, 0.0, None),
        stall_sample(Some(200.0), 0.0, 0.0, None),
        stall_sample(Some(200.0), 60.0, 0.0, None),
        stall_sample(Some(300.0), 60.0, 4.0, None),
        stall_sample(Some(400.0), 60.0, 4.0, None),
    ]);
    let d = derive(&h, 0).unwrap();
    assert_eq!(d.graphs.stall, vec![false, false, false, false]);
    assert_eq!(d.stalls[0].count, 0);
}

// The cached window scan is rebuilt on every push, so the UI always
// reads a scan matching the adjacent ring.
#[test]
fn cached_window_scan_tracks_the_ring() {
    use vllm_top::derive::scan_window;

    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, stall_sample(Some(0.0), 0.0, 4.0, None));
    assert!(h.scan().is_some(), "a push must populate the scan cache");
    h.push(
        t0 + Duration::from_secs(1),
        stall_sample(Some(0.0), 50.0, 4.0, None),
    );
    h.push(
        t0 + Duration::from_secs(2),
        stall_sample(Some(100.0), 50.0, 4.0, None),
    );
    assert_eq!(h.scan().unwrap().stall, scan_window(&h).stall);
    // eviction must not desynchronize the cache: pushing RING_CAP more
    // samples evicts the oldest entries, and the cached scan still
    // matches a fresh walk
    for i in 0..(RING_CAP as u32) {
        h.push(
            t0 + Duration::from_secs(3 + i as u64),
            stall_sample(Some(100.0 + f64::from(i)), 50.0, 4.0, None),
        );
    }
    assert_eq!(
        h.scan().unwrap().decode.vals.len(),
        h.window_entries(Duration::from_secs(60)).len() - 1
    );
}

// ---- busy-server fixture: a loaded server, generated in code ----

/// Decode counter state the busy fixture carries at scrape `step`:
/// 40 tokens/s for the first 5 steps, then 80 tokens/s. Counters
/// advance one scrape interval (1s) per step, so window rates
/// and histogram deltas stay hand-computable.
fn busy_decode_counter(step: u32) -> f64 {
    if step <= 5 {
        40.0 * f64::from(step)
    } else {
        200.0 + 80.0 * f64::from(step - 5)
    }
}

/// What the busy fixture's decode counter line holds at one scrape:
/// a normal reading, no line at all, or a non-finite reading. Absent
/// is the shape an unreadable value leaves after the parse boundary
/// drops it.
#[derive(Clone, Copy, Debug)]
enum BusyDecode {
    Normal,
    Absent,
    NonFinite,
}

/// Cumulative histogram lines for one family: `bounds` lists the finite
/// `le` bounds, `cum` the cumulative count through each bound. A final
/// `cum` entry covers the +Inf bucket, so `cum` always has one entry
/// more than `bounds` holds.
fn hist_lines(name: &str, bounds: &[f64], cum: &[f64]) -> String {
    let mut s = format!("# TYPE {name} histogram\n");
    for (le, c) in bounds.iter().zip(cum) {
        s.push_str(&format!("{name}_bucket{{le=\"{le}\"}} {c}\n"));
    }
    let total = cum[cum.len() - 1];
    s.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {total}\n"));
    s.push_str(&format!("{name}_count {total}\n"));
    s.push_str(&format!("{name}_sum {}\n", total * 0.1));
    s
}

/// One scrape of a loaded server: running slots above zero, a queue,
/// tokens advancing, non-trivial histogram traffic, the KV and host pools,
/// and the health counters. Every counter advances linearly in `step`
/// (its per-second rate is the per-step delta), so every derived number
/// has a hand-computable expectation.
fn busy_body(step: u32, decode: BusyDecode) -> String {
    let s = f64::from(step);
    let decode_line = match decode {
        BusyDecode::Normal => format!(
            "vllm:generation_tokens_total {}\n",
            busy_decode_counter(step)
        ),
        BusyDecode::NonFinite => "vllm:generation_tokens_total NaN\n".into(),
        BusyDecode::Absent => String::new(),
    };
    format!(
        "# TYPE vllm:generation_tokens_total counter\n\
         {decode_line}\
         # TYPE vllm:prompt_tokens_total counter\n\
         vllm:prompt_tokens_total {}\n\
         # TYPE vllm:prompt_tokens_by_source_total counter\n\
         vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {}\n\
         vllm:prompt_tokens_by_source_total{{source=\"local_cache_hit\"}} {}\n\
         # TYPE vllm:num_requests_running gauge\n\
         vllm:num_requests_running{{engine=\"0\",model_name=\"busy-model\"}} 4\n\
         # TYPE vllm:num_requests_waiting gauge\n\
         vllm:num_requests_waiting 2\n\
         # TYPE vllm:kv_cache_usage_perc gauge\n\
         vllm:kv_cache_usage_perc 0.8\n\
         # TYPE vllm:cache_config_info gauge\n\
         vllm:cache_config_info{{kv_cache_size_tokens=\"10000\"}} 1.0\n\
         # TYPE vllm:kv_offload_cpu_cache_usage_perc gauge\n\
         vllm:kv_offload_cpu_cache_usage_perc 0.8\n\
         # TYPE vllm:prefix_cache_hits_total counter\n\
         vllm:prefix_cache_hits_total {}\n\
         # TYPE vllm:prefix_cache_queries_total counter\n\
         vllm:prefix_cache_queries_total {}\n\
         # TYPE vllm:num_preemptions_total counter\n\
         vllm:num_preemptions_total {}\n\
         # TYPE vllm:request_success_total counter\n\
         vllm:request_success_total{{finished_reason=\"abort\"}} {}\n\
         # TYPE http_requests_total counter\n\
         http_requests_total{{status=\"5xx\"}} {}\n\
         # TYPE vllm:kv_offload_store_bytes_total counter\n\
         vllm:kv_offload_store_bytes_total {}\n\
         # TYPE vllm:kv_offload_load_bytes_total counter\n\
         vllm:kv_offload_load_bytes_total {}\n\
         # TYPE vllm:kv_offload_allocation_failure_total counter\n\
         vllm:kv_offload_allocation_failure_total 0\n\
         # TYPE vllm:spec_decode_num_accepted_tokens_total counter\n\
         vllm:spec_decode_num_accepted_tokens_total {}\n\
         # TYPE vllm:spec_decode_num_draft_tokens_total counter\n\
         vllm:spec_decode_num_draft_tokens_total {}\n\
         # TYPE vllm:spec_decode_num_drafts_total counter\n\
         vllm:spec_decode_num_drafts_total {}\n\
         # TYPE vllm:scheduler_compute_seconds_total counter\n\
         vllm:scheduler_compute_seconds_total {}\n\
         {}{}{}{}{}",
        100.0 * s,
        10.0 * s,
        90.0 * s,
        90.0 * s,
        100.0 * s,
        5.0 * s,
        1.0 * s,
        2.0 * s,
        25.0 * s,
        15.0 * s,
        65.0 * s,
        100.0 * s,
        20.0 * s,
        0.8 * s,
        // ttft: +2 obs <= 0.1, +3 in (0.1, 1], +1 in (1, 10] per step
        hist_lines(
            "vllm:time_to_first_token_seconds",
            &[0.1, 1.0, 10.0],
            &[2.0 * s, 5.0 * s, 6.0 * s, 6.0 * s],
        ),
        // itl: +2 obs per finite bucket per step
        hist_lines(
            "vllm:inter_token_latency_seconds",
            &[0.02, 0.1, 1.0],
            &[2.0 * s, 4.0 * s, 6.0 * s, 6.0 * s],
        ),
        // e2e: +1 <= 0.5, +3 in (0.5, 2], +2 in (2, 8] per step
        hist_lines(
            "vllm:e2e_request_latency_seconds",
            &[0.5, 2.0, 8.0],
            &[1.0 * s, 4.0 * s, 6.0 * s, 6.0 * s],
        ),
        // queue time: +4 <= 0.05, +2 in (0.05, 0.5] per step
        hist_lines(
            "vllm:request_queue_time_seconds",
            &[0.05, 0.5],
            &[4.0 * s, 6.0 * s, 6.0 * s],
        ),
        // prompt length: +1 <= 128, +2 in (128, 512], +3 in (512, 2048]
        hist_lines(
            "vllm:request_prompt_tokens",
            &[128.0, 512.0, 2048.0],
            &[1.0 * s, 3.0 * s, 6.0 * s, 6.0 * s],
        ),
    )
}

/// A busy server scraped once per second for `steps` steps.
fn busy_history(steps: usize) -> History {
    let mut h = History::default();
    let t0 = Instant::now();
    for i in 0..steps {
        h.push(
            t0 + Duration::from_secs(i as u64),
            parse(&busy_body(i as u32, BusyDecode::Normal)).unwrap(),
        );
    }
    h
}

fn busy_derived() -> vllm_top::derive::Derived {
    derive(&busy_history(6), 2).unwrap()
}

// The headline identity lines come from the running-count series labels,
// so a fixture whose labels differ from the live exporter still pins
// the label-extraction path.
#[test]
fn busy_server_names_model_and_engine_from_series_labels() {
    let d = busy_derived();
    assert_eq!(d.model, "busy-model");
    assert_eq!(d.engine, "0");
}

// The focused window rides through derive untouched, so the UI's window
// switch lands on the same number derive computed.
#[test]
fn window_focus_passes_through_derive() {
    for focus in 0..3 {
        let d = derive(&busy_history(6), focus).unwrap();
        assert_eq!(d.window_focus, focus);
    }
}

// RUNNING is the running count of the latest scrape, not an average
// over the window: an engine that just got calmer reads the calm number.
#[test]
fn running_reports_the_latest_scrape() {
    let d = busy_derived();
    assert_eq!(d.running, Some(4.0));
}

// QUEUE is the latest scrape's waiting count.
#[test]
fn queue_reports_the_latest_scrape() {
    let d = busy_derived();
    assert_eq!(d.queue, Some(2.0));
}

// The three rate windows see different slices of a rate change: the 5s
// window holds only the fast interval, the 15s and 60s windows average
// the slow first half in. Deltas are 40 tokens/s for steps 1-5 and 80
// after, so 5s reads 400/5 = 80 and the full 10s span reads 600/10 = 60.
#[test]
fn decode_rates_span_all_three_windows() {
    let d = derive(&busy_history(11), 2).unwrap();
    assert_eq!(d.decode_rate[0], Some(80.0));
    assert_eq!(d.decode_rate[1], Some(60.0));
    assert_eq!(d.decode_rate[2], Some(60.0));
}

// The instant decode rate is the most recent scrape interval's rate,
// not a windowed average: with the last interval at 80 tokens/s,
// the instant reads 80, while the 60s window averages
// the slower first half to 60.
#[test]
fn decode_instant_is_the_most_recent_interval() {
    let d = derive(&busy_history(11), 2).unwrap();
    assert_eq!(d.decode_instant, Some(80.0));
}

// Prefill runs at a constant 100 tokens/s in the fixture, so all three
// windows agree with the instant rate.
#[test]
fn prefill_rates_span_all_three_windows() {
    let d = busy_derived();
    assert_eq!(d.prefill_rate[0], Some(10.0));
    assert_eq!(d.prefill_rate[1], Some(10.0));
    assert_eq!(d.prefill_rate[2], Some(10.0));
    assert_eq!(d.prefill_instant, Some(10.0));
}

// The pool rows carry the usage ratio plus the engine-reported absolute
// counts, with the unit saying what the counts count.
#[test]
fn pool_rows_carry_usage_counts_and_units() {
    let d = busy_derived();
    let find = |name: &str| {
        d.pools
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("pool {name} missing"))
    };
    // KV absolute counts are derived from the config capacity x usage %
    let kv = find("KV");
    assert_eq!(kv.usage, 0.8);
    assert_eq!(kv.used, Some(8000.0));
    assert_eq!(kv.total, Some(10000.0));
    assert_eq!(kv.unit, "tokens");
    // host (CPU KV offload) reports a usage % only, no absolute count
    let host = find("host");
    assert_eq!(host.usage, 0.8);
    assert_eq!(host.used, None);
    assert_eq!(host.total, None);
    assert_eq!(host.unit, "tokens");
}

// The eviction lane replays the per-interval evicted-tokens rate under the rate lanes'
// gap-aware pairing; the fixture advances its eviction counter 5 tokens per interval.
#[test]
fn eviction_lane_replays_the_intervals() {
    let d = busy_derived();
    assert_eq!(d.graphs.evictions.vals, vec![5.0; 5]);
    assert_eq!(d.graphs.evictions.absent_old, vec![false; 5]);
    assert_eq!(d.graphs.evictions.absent_new, vec![false; 5]);
    assert_eq!(d.graphs.dt, vec![1.0; 5]);
    // the headline eviction rate reads the same counter over the window
    assert_eq!(d.evict_rate, Some(5.0));
}

// An eviction counter absent from one sample of a pair pairs nothing, like every other
// counter: the touching intervals read 0 (no eviction tick) instead of fabricating a rate
// across the gap, and a reappearance after the absence also pairs nothing.
#[test]
fn eviction_lane_pairs_nothing_across_an_absence() {
    let with_evict = |v: f64| {
        parse(&format!(
            "# TYPE vllm:num_preemptions_total counter\n\
             vllm:num_preemptions_total {v}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n"
        ))
        .unwrap()
    };
    let without =
        parse("# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 2\n").unwrap();
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, with_evict(0.0));
    h.push(t0 + Duration::from_secs(1), with_evict(10.0));
    h.push(t0 + Duration::from_secs(2), without);
    h.push(t0 + Duration::from_secs(3), with_evict(30.0));
    let d = derive(&h, 2).unwrap();
    // interval 0 pairs 10 tokens/s; the intervals touching the absent
    // sample and the reappearance after it all pair nothing
    assert_eq!(d.graphs.evictions.vals, vec![10.0, 0.0, 0.0]);
    assert_eq!(d.graphs.evictions.absent_old, vec![false, false, true]);
    assert_eq!(d.graphs.evictions.absent_new, vec![false, true, false]);
}

// Per-position latency (TTFT family): each point is the histogram bucket delta
// between the window's first sample and the position's later sample, so the newest point
// is the window's headline quantile and earlier points cover a shorter growing span.
// Hand-computed: position 1 sees 5 observations in (0, 0.5]: p50 rank 2.5 gives 0.25, p95
// rank 4.75 gives 0.475. Position 2 adds 10 in (0.5, 2] and 20 in (2, 8]: p95
// rank 23.75 gives 7.25. A per-interval delta reading of position 2 would give
// 2.0 instead, since rank 10 of 10 in the (0.5, 2] bucket interpolates toward
// its upper bound: this pin tells the two readings apart.
#[test]
fn latency_series_is_the_window_span_delta_per_position() {
    let ttft = |cum: [f64; 4]| {
        parse(&format!(
            "# TYPE vllm:time_to_first_token_seconds histogram\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.5\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"2\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"8\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"+Inf\"}} {}\n\
             vllm:time_to_first_token_seconds_count {}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n",
            cum[0], cum[1], cum[2], cum[3], cum[3]
        ))
        .unwrap()
    };
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, ttft([0.0, 0.0, 0.0, 0.0]));
    h.push(t0 + Duration::from_secs(1), ttft([5.0, 5.0, 5.0, 5.0]));
    h.push(t0 + Duration::from_secs(2), ttft([5.0, 15.0, 25.0, 25.0]));
    let d = derive(&h, 2).unwrap();
    assert_eq!(d.graphs.ttft_p95, vec![Some(0.475), Some(7.25)]);
}

// A position whose span holds fewer observations than the quantile bar stores no latency:
// the plot draws a gap there, never a zero or a stale point. The bar is the same
// 5-observation minimum the headline quantile applies.
#[test]
fn latency_series_stores_no_quantile_when_the_span_is_sparse() {
    let ttft = |cum: [f64; 2]| {
        parse(&format!(
            "# TYPE vllm:time_to_first_token_seconds histogram\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.5\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"+Inf\"}} {}\n\
             vllm:time_to_first_token_seconds_count {}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n",
            cum[0], cum[1], cum[1]
        ))
        .unwrap()
    };
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, ttft([0.0, 0.0]));
    h.push(t0 + Duration::from_secs(1), ttft([2.0, 2.0]));
    h.push(t0 + Duration::from_secs(2), ttft([4.0, 4.0]));
    h.push(t0 + Duration::from_secs(3), ttft([6.0, 6.0]));
    let d = derive(&h, 2).unwrap();
    // positions 1 and 2 hold 2 and 4 observations, below the bar. Position 3's
    // 6-observation span interpolates: p95 rank 5.7 lands on 0.475, compared
    // with an epsilon because the rank itself carries binary float rounding
    let p95 = d.graphs.ttft_p95[2].unwrap();
    assert!((p95 - 0.475).abs() < 1e-9, "p95 was {p95}");
}

// The newest plot point is computed from the same span as the headline 60s quantile:
// on a fixture with enough traffic the two agree. The real contract is the sparse
// asymmetry in the second half: a window with too few observations makes the headline
// fall back to the since-start snapshot while the plot point stays None, never
// echoing the fallback.
#[test]
fn newest_plot_point_matches_the_headline_quantile() {
    let d = busy_derived();
    assert_eq!(d.graphs.ttft_p95.last().copied().flatten(), d.ttft[2].p95);

    // sparse window: 2+2 observations across the whole window, below the bar
    let ttft = |cum: [f64; 2]| {
        parse(&format!(
            "# TYPE vllm:time_to_first_token_seconds histogram\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.5\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"+Inf\"}} {}\n\
             vllm:time_to_first_token_seconds_count {}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n",
            cum[0], cum[1], cum[1]
        ))
        .unwrap()
    };
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, ttft([0.0, 0.0]));
    h.push(t0 + Duration::from_secs(1), ttft([2.0, 2.0]));
    h.push(t0 + Duration::from_secs(2), ttft([4.0, 4.0]));
    let d = derive(&h, 2).unwrap();
    // the headline echoes the since-start snapshot (4 obs <= 0.5: rank 3.8 -> 0.475)
    let headline = d.ttft[2].p95.unwrap();
    assert!((headline - 0.475).abs() < 1e-9, "headline was {headline}");
    // the plot point stays None: no measured window span to interpolate
    assert_eq!(d.graphs.ttft_p95.last().copied().flatten(), None);
}

// merge_le nets bucket deltas across label combinations, so a reset inside one
// combination can hide against growth in another (-4 and +200 merge to +196).
// The reset check therefore runs on the raw per-combination deltas,
// before any merging: the plotted span stores a gap, never a quantile
// of a broken span.
#[test]
fn a_reset_in_one_label_combination_gaps_the_window_span() {
    let ttft = |a: f64, b: f64| {
        parse(&format!(
            "# TYPE vllm:time_to_first_token_seconds histogram\n\
             vllm:time_to_first_token_seconds_bucket{{name=\"a\",le=\"0.5\"}} {a}\n\
             vllm:time_to_first_token_seconds_bucket{{name=\"a\",le=\"+Inf\"}} {a}\n\
             vllm:time_to_first_token_seconds_count{{name=\"a\"}} {a}\n\
             vllm:time_to_first_token_seconds_bucket{{name=\"b\",le=\"0.5\"}} {b}\n\
             vllm:time_to_first_token_seconds_bucket{{name=\"b\",le=\"+Inf\"}} {b}\n\
             vllm:time_to_first_token_seconds_count{{name=\"b\"}} {b}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n"
        ))
        .unwrap()
    };
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, ttft(4.0, 0.0));
    h.push(t0 + Duration::from_secs(1), ttft(0.0, 200.0));
    let d = derive(&h, 2).unwrap();
    // combination a lost its 4 observations while b grew by 200: the merged
    // ladder reads +196 and would interpolate a p95 from a span that mixes
    // a reset with growth
    assert_eq!(d.graphs.ttft_p95.last(), Some(&None));
}

// The cache hit fraction comes from the effective-prefill counter's windowed
// rates (hit modes 60+30 over all modes 100 = 0.9) while that family exists.
// The engine's cache_hit_rate gauge (0.73 in the fixture) is the fallback
// path for exporters without the counter.
// An alarm channel latches the first time its per-interval rate reads
// nonzero and stays latched: a later quiet interval never clears it.
#[test]
fn alarm_latches_flip_on_first_nonzero_interval_and_stay_latched() {
    let with_retract = |v: f64| {
        format!(
            "# TYPE vllm:request_success_total counter\n\
             vllm:request_success_total{{finished_reason=\"abort\"}} {v}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n"
        )
    };
    let mut a = vllm_top::derive::Alarms::default();
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(&with_retract(0.0)).unwrap());
    h.push(
        t0 + Duration::from_secs(1),
        parse(&with_retract(5.0)).unwrap(),
    );
    vllm_top::derive::update_alarms(&mut a, &h);
    assert!(a.abort, "first nonzero abort interval must latch");
    h.push(
        t0 + Duration::from_secs(2),
        parse(&with_retract(5.0)).unwrap(),
    );
    vllm_top::derive::update_alarms(&mut a, &h);
    assert!(
        a.abort,
        "session-sticky: a quiet interval never clears it"
    );
}

// The latches are session-sticky per engine session: the same counter reset
// evidence that clears the session peaks clears them, so a restarted engine
// does not inherit the previous process's fired alarms.
#[test]
fn alarm_latches_clear_on_the_engine_restart_evidence() {
    let with_retract = |retract: f64, decode: f64| {
        format!(
            "# TYPE vllm:request_success_total counter\n\
             vllm:request_success_total{{finished_reason=\"abort\"}} {retract}\n\
             # TYPE vllm:generation_tokens_total counter\n\
             vllm:generation_tokens_total {decode}\n"
        )
    };
    let mut a = vllm_top::derive::Alarms::default();
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(&with_retract(0.0, 100.0)).unwrap());
    h.push(
        t0 + Duration::from_secs(1),
        parse(&with_retract(5.0, 160.0)).unwrap(),
    );
    vllm_top::derive::update_alarms(&mut a, &h);
    assert!(a.abort, "the latch must fire before the restart");
    // the token counter drops: the engine restarted, so the fired alarm
    // belongs to the previous process
    h.push(
        t0 + Duration::from_secs(2),
        parse(&with_retract(5.0, 50.0)).unwrap(),
    );
    vllm_top::derive::update_alarms(&mut a, &h);
    assert!(
        !a.abort && !a.http_5xx && !a.alloc_fail,
        "restart evidence must clear the old session's latches"
    );
}

// Fixture integrity: every body line is exposition-format data. A `//`
// inside the format literal is string content, not a comment, and once
// spliced in it silently swallows the neighboring metric header.
#[test]
fn busy_body_contains_no_stray_comment_lines() {
    for step in 0..3 {
        for decode in [
            BusyDecode::Normal,
            BusyDecode::Absent,
            BusyDecode::NonFinite,
        ] {
            assert!(
                !busy_body(step, decode).contains("//"),
                "fixture body carries a comment line (string-literal splice)"
            );
        }
    }
}

// The cache-miss lane draws absence and measured zero differently:
// when the `input` counter family is missing from an interval's
// endpoint samples the point is a gap (None), while a present counter
// with no delta is a zero bar.
#[test]
fn cache_miss_lane_gaps_when_the_counter_family_is_absent() {
    let miss = "# TYPE vllm:prompt_tokens_by_source_total counter\n\
        vllm:prompt_tokens_by_source_total{source=\"local_compute\"} {v}\n";
    let quiet = "# TYPE vllm:num_requests_running gauge\n\
        vllm:num_requests_running 1\n";
    let mut h = History::default();
    let t0 = Instant::now();
    // family present with no delta between the two samples: a measured zero
    h.push(t0, parse(&miss.replace("{v}", "10")).unwrap());
    h.push(
        t0 + Duration::from_secs(1),
        parse(&miss.replace("{v}", "10")).unwrap(),
    );
    let g = h.scan().unwrap();
    assert_eq!(g.cache_misses, vec![Some(0.0)]);
    // the family vanishes from the later sample: absence, not a zero bar
    h.push(t0 + Duration::from_secs(2), parse(quiet).unwrap());
    let g = h.scan().unwrap();
    assert_eq!(g.cache_misses, vec![Some(0.0), None]);
}

#[test]
fn cache_hit_rate_comes_from_the_prefix_cache_counters() {
    let d = busy_derived();
    assert_eq!(d.cache_hit, Some(0.9));
    assert_eq!(d.cache_cached, Some(90.0));
    assert_eq!(d.cache_computed, Some(10.0));
    // the miss lane replays the per-interval computed rate
    assert_eq!(d.graphs.cache_misses, vec![Some(10.0); 5]);
}

// Without the prefix-cache counters the hit fraction reads as unknown,
// not as a fabricated value from an unrelated gauge.
#[test]
fn cache_hit_is_none_without_prefix_cache_counters() {
    let body = "# TYPE vllm:num_requests_running gauge\n\
                vllm:num_requests_running 2\n";
    let mut h = History::default();
    let t0 = Instant::now();
    h.push(t0, parse(body).unwrap());
    h.push(t0 + Duration::from_secs(1), parse(body).unwrap());
    let d = derive(&h, 2).unwrap();
    assert_eq!(d.cache_hit, None);
    assert_eq!(d.cache_cached, None);
    assert_eq!(d.cache_computed, None);
}

#[test]
fn spec_accept_rate_comes_from_the_counters() {
    assert_eq!(busy_derived().spec_accept, Some(0.65));
}

#[test]
fn spec_accept_length_comes_from_the_counters() {
    assert_eq!(busy_derived().spec_accept_len, Some(3.25));
}

// TTFT quantiles from window bucket deltas, hand-computed. Window
// deltas: 10 obs <= 0.1, 15 in (0.1, 1], 5 in (1, 10], total 30.
// p50 rank 15:  0.1 + 5/15 * 0.9 = 0.4
// p95 rank 28.5: 1 + 3.5/5 * 9 = 7.3
// p99 rank 29.7: 1 + 4.7/5 * 9 = 9.46
#[test]
fn ttft_quantiles_match_hand_computed_values() {
    let d = busy_derived();
    let q = &d.ttft[2];
    assert!((q.p50.unwrap() - 0.4).abs() < 1e-9, "p50 was {:?}", q.p50);
    assert!((q.p95.unwrap() - 7.3).abs() < 1e-9, "p95 was {:?}", q.p95);
    assert!((q.p99.unwrap() - 9.46).abs() < 1e-9, "p99 was {:?}", q.p99);
}

// ITL deltas over the window: 10 obs <= 0.02, 10 more in (0.02, 0.1],
// and 10 in (0.1, 1].
// p50 rank 15: 0.02 + 5/10 * 0.08 = 0.06
// p95 rank 28.5: 0.1 + 8.5/10 * 0.9 = 0.865
// p99 rank 29.7: 0.1 + 9.7/10 * 0.9 = 0.973
#[test]
fn itl_quantiles_match_hand_computed_values() {
    let d = busy_derived();
    let q = &d.itl[2];
    assert!((q.p50.unwrap() - 0.06).abs() < 1e-9, "p50 was {:?}", q.p50);
    assert!((q.p95.unwrap() - 0.865).abs() < 1e-9, "p95 was {:?}", q.p95);
    assert!((q.p99.unwrap() - 0.973).abs() < 1e-9, "p99 was {:?}", q.p99);
}

// E2E deltas: 5 obs <= 0.5, 15 in (0.5, 2], 10 in (2, 8].
// p50 rank 15: 0.5 + 10/15 * 1.5 = 1.5
// p95 rank 28.5: 2 + 8.5/10 * 6 = 7.1
// p99 rank 29.7: 2 + 9.7/10 * 6 = 7.82
#[test]
fn e2e_quantiles_match_hand_computed_values() {
    let d = busy_derived();
    let q = &d.e2e[2];
    assert!((q.p50.unwrap() - 1.5).abs() < 1e-9, "p50 was {:?}", q.p50);
    assert!((q.p95.unwrap() - 7.1).abs() < 1e-9, "p95 was {:?}", q.p95);
    assert!((q.p99.unwrap() - 7.82).abs() < 1e-9, "p99 was {:?}", q.p99);
}

// Queue-time deltas: 20 obs <= 0.05, 10 in (0.05, 0.5].
// p50 rank 15 sits in the first bucket, which starts at 0:
// 15/20 * 0.05 = 0.0375
// p95 rank 28.5: 0.05 + 8.5/10 * 0.45 = 0.4325
// p99 rank 29.7: 0.05 + 9.7/10 * 0.45 = 0.4865
#[test]
fn queue_time_quantiles_match_hand_computed_values() {
    let d = busy_derived();
    let q = &d.queue_time[2];
    assert!(
        (q.p50.unwrap() - 0.0375).abs() < 1e-9,
        "p50 was {:?}",
        q.p50
    );
    assert!(
        (q.p95.unwrap() - 0.4325).abs() < 1e-9,
        "p95 was {:?}",
        q.p95
    );
    assert!(
        (q.p99.unwrap() - 0.4865).abs() < 1e-9,
        "p99 was {:?}",
        q.p99
    );
}

// Prompt-length deltas: 5 obs <= 128, 10 in (128, 512], then 15 more
// in (512, 2048]. p50 rank 15 lands exactly on the 512 boundary.
// p95 rank 28.5: 512 + 13.5/15 * 1536 = 1894.4.
#[test]
fn prompt_len_quantiles_match_hand_computed_values() {
    let d = busy_derived();
    assert_eq!(d.prompt_len_p50, Some(512.0));
    assert!((d.prompt_len_p95.unwrap() - 1894.4).abs() < 1e-9);
}

// The stall banner stays silent on a uniformly busy server: decode well
// above a quarter of the window median, prefill computing, nothing
// frozen in any window.
#[test]
fn busy_server_has_no_stalls_in_any_window() {
    let d = busy_derived();
    for (i, s) in d.stalls.iter().enumerate() {
        assert_eq!(s.count, 0, "window {i} counted {s:?}");
        assert_eq!(s.seconds, 0.0);
    }
}

// A decode reading dropped mid-window is a gap rather than a collapse:
// the interval leading out of the gap is flagged
// (running > 0, prefill computing, rate 0), the interval into the gap
// is not, and a NaN reading flags identically because the parse boundary
// dropped it before the scan ever saw it.
#[test]
fn stall_banner_counts_one_frozen_second_per_gap() {
    for variant in [BusyDecode::Absent, BusyDecode::NonFinite] {
        let mut h = History::default();
        let t0 = Instant::now();
        for i in 0..6u32 {
            let decode = if i == 2 { variant } else { BusyDecode::Normal };
            h.push(
                t0 + Duration::from_secs(u64::from(i)),
                parse(&busy_body(i, decode)).unwrap(),
            );
        }
        let d = derive(&h, 2).unwrap();
        assert_eq!(
            d.graphs.stall,
            vec![false, false, true, false, false],
            "flags for {variant:?} were {:?}",
            d.graphs.stall
        );
        assert_eq!(d.stalls[2].count, 1);
        assert!((d.stalls[2].seconds - 1.0).abs() < 1e-9);
    }
}

#[test]
fn evict_rate_counts_evicted_tokens_per_second() {
    assert_eq!(busy_derived().evict_rate, Some(5.0));
}

#[test]
fn retract_rate_counts_retracted_requests_per_second() {
    assert_eq!(busy_derived().retract_rate, Some(1.0));
}

// Only the 5xx series feeds the error rate: other status buckets in the
// same family must not leak into it.
#[test]
fn http_5xx_rate_isolates_the_5xx_series() {
    assert_eq!(busy_derived().http_503_rate, Some(2.0));
}

// CPU KV offload traffic reports its store/load bytes/s, and an
// allocation-failure counter that exists but never moves reads as a
// present zero (allocations tracked, none failed), not as missing data.
#[test]
fn offload_traffic_rates_report_store_load_and_fail() {
    let d = busy_derived();
    assert_eq!(d.l2_wb, Some(25.0));
    assert_eq!(d.l2_rb, Some(15.0));
    assert_eq!(d.l2_drop, Some(0.0));
}

// vLLM exposes no decode sequence-length sum, so the generation-progress
// rows are gone; the scheduler compute counter is the CPU reading.
#[test]
fn cpu_scheduler_rate_counts_seconds_per_second() {
    assert_eq!(busy_derived().cpu_scheduler, Some(0.8));
}

// The graph lanes replay the fixture's per-interval rates: 40 tokens/s
// of decode and 100 of prefill across all five intervals, no presence
// flags, one second per interval.
#[test]
fn busy_graph_lanes_replay_the_intervals() {
    let d = busy_derived();
    assert_eq!(d.graphs.decode.vals, vec![40.0; 5]);
    assert_eq!(d.graphs.prefill.vals, vec![10.0; 5]);
    assert_eq!(d.graphs.decode.absent_old, vec![false; 5]);
    assert_eq!(d.graphs.decode.absent_new, vec![false; 5]);
    assert_eq!(d.graphs.dt, vec![1.0; 5]);
}

// Each graph lane divides its interval's decode rate by the running
// count at that interval's later sample, never by the latest scrape's
// count. One shared history pins both semantics: the fixture carries
// 4 running requests at its first sample, then 8 at the second,
// so lane one reads 5 (40/8) while headline RUNNING
// and the generation average read the latest count, 4.
#[test]
fn per_stream_uses_the_running_count_of_its_own_interval() {
    let mut h = History::default();
    let t0 = Instant::now();
    for (i, running) in [4.0, 8.0, 4.0, 4.0].iter().enumerate() {
        let body = format!(
            "# TYPE vllm:generation_tokens_total counter\n\
             vllm:generation_tokens_total {}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running {running}\n",
            40.0 * i as f64
        );
        h.push(t0 + Duration::from_secs(i as u64), parse(&body).unwrap());
    }
    let d = derive(&h, 2).unwrap();
    assert_eq!(d.graphs.decode.vals, vec![40.0; 3]);
    assert_eq!(d.graphs.per_stream, vec![Some(5.0), Some(10.0), Some(10.0)]);
    assert_eq!(d.running, Some(4.0));
}

// Missing decode data must not read as a collapse. The per-stream
// lane stays empty when the decode series is missing on either side,
// even while requests ran throughout. A non-finite reading behaves
// identically, because the parse boundary drops it before the scan
// sees the series. The headline window rate stays gap-aware.
#[test]
fn per_stream_stores_no_speed_when_decode_data_is_missing() {
    for variant in [BusyDecode::Absent, BusyDecode::NonFinite] {
        let mut h = History::default();
        let t0 = Instant::now();
        for i in 0..6u32 {
            let decode = if i == 2 { variant } else { BusyDecode::Normal };
            h.push(
                t0 + Duration::from_secs(u64::from(i)),
                parse(&busy_body(i, decode)).unwrap(),
            );
        }
        let d = derive(&h, 2).unwrap();
        // intervals touching the missing sample pair nothing; healthy
        // intervals around them read 40 tok/s across 4 requests
        assert_eq!(
            d.graphs.per_stream,
            vec![Some(10.0), None, None, Some(10.0), Some(10.0)],
            "per-stream speeds for {variant:?} were {:?}",
            d.graphs.per_stream
        );
        // running was positive across the whole window, so the Nones
        // are absence, not idleness
        assert_eq!(d.running, Some(4.0));
        // the headline rate pairs only healthy intervals: 3 x 40 tokens
        // over the 5s span
        assert_eq!(d.decode_rate[2], Some(24.0));
    }
}

// Session peaks only ever move up while one engine process runs.
// Each scrape folds max(existing, new rate) into the running peaks.
// Deltas here climb 40, 80, 120, 160, then drop to 20, and the peaks
// must stay at the 160 interval.
#[test]
fn session_peaks_only_grow_until_a_counter_reset() {
    use vllm_top::derive::{update_peaks, Peaks};

    let decode_values = [0.0, 40.0, 120.0, 240.0, 400.0, 420.0];
    let mut h = History::default();
    let mut peaks = Peaks::default();
    let t0 = Instant::now();
    let mut seen_decode = Vec::new();
    for (i, v) in decode_values.iter().enumerate() {
        let body = format!(
            "# TYPE vllm:generation_tokens_total counter\n\
             vllm:generation_tokens_total {v}\n\
             # TYPE vllm:prompt_tokens_by_source_total counter\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 4\n",
            100.0 * i as f64
        );
        h.push(t0 + Duration::from_secs(i as u64), parse(&body).unwrap());
        update_peaks(&mut peaks, &h);
        seen_decode.push(peaks.decode);
        if i > 0 {
            assert_eq!(peaks.prefill, Some(100.0), "prefill peak moved at step {i}");
        }
    }
    assert_eq!(
        seen_decode,
        vec![
            None,
            Some(40.0),
            Some(80.0),
            Some(120.0),
            Some(160.0),
            Some(160.0)
        ]
    );
    assert_eq!(peaks.decode_single, Some(40.0)); // 160 tok/s across 4 requests
}
