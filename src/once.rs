use anyhow::Result;

use crate::derive::{self, Derived, WIN_LABELS};
use crate::history::History;

/// One-shot text snapshot for scripts, cron logs, and parser verification.
pub fn print_once(h: &History, peaks: &mut derive::Peaks) -> Result<()> {
    println!("{}", snapshot_text(h, peaks)?);
    Ok(())
}

/// Renders the one-shot snapshot without printing it: exactly the text
/// [`print_once`] writes to stdout, one trailing newline included.
pub fn snapshot_text(h: &History, peaks: &mut derive::Peaks) -> Result<String> {
    let d = derive::derive(h, 2).ok_or_else(|| anyhow::anyhow!("no data scraped yet"))?;
    derive::update_peaks(peaks, h);
    Ok(print_snapshot(&d, peaks))
}

fn tok(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.0} tok/s"))
        .unwrap_or_else(|| "—".into())
}

fn secs(v: Option<f64>) -> String {
    match v {
        Some(v) if v >= 10.0 => format!("{v:.1}s"),
        Some(v) => format!("{:.0}ms", v * 1000.0),
        None => "—".into(),
    }
}

fn pct(v: Option<f64>) -> String {
    v.map(|v| format!("{:.0}%", v * 100.0))
        .unwrap_or_else(|| "—".into())
}

fn num(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into())
}

/// Byte throughput (e.g. KV offload store/load, in bytes/s).
fn bytes(v: Option<f64>) -> String {
    match v {
        Some(v) if v >= 1.0e9 => format!("{:.1}G", v / 1.0e9),
        Some(v) if v >= 1.0e6 => format!("{:.1}M", v / 1.0e6),
        Some(v) if v >= 1.0e3 => format!("{:.0}k", v / 1.0e3),
        Some(v) => format!("{v:.0}"),
        None => "—".into(),
    }
}

// Label values (model, engine) reach the headline verbatim
// from the exporter; terminal sanitization therefore applies to them
// like every other display-bound string.
fn print_snapshot(d: &Derived, peaks: &derive::Peaks) -> String {
    let f = d.window_focus;
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "vllm-top {} ({}) — running {} · queue {} · pools: {}",
        crate::scrape::sanitize_for_terminal(&d.model),
        crate::scrape::sanitize_for_terminal(&d.engine),
        num(d.running),
        num(d.queue),
        d.pools
            .iter()
            .map(|p| match crate::ui::pool_value(p) {
                Some(v) => format!("{} {v}", p.name),
                None => p.name.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" / ")
    ));
    lines.push(format!(
        "tokens: decode {} · prefill {} (5s/15s/60s: {}/{}/{}) · cache hit {} · spec accept {}",
        tok(d.decode_instant),
        tok(d.prefill_instant),
        tok(d.decode_rate[0]),
        tok(d.decode_rate[1]),
        tok(d.decode_rate[2]),
        pct(d.cache_hit),
        pct(d.spec_accept)
    ));
    for (name, lat) in [("ttft", &d.ttft), ("itl", &d.itl), ("e2e", &d.e2e)] {
        let q = &lat[f];
        lines.push(format!(
            "{name} ({}): p50 {} · p95 {} · p99 {}",
            WIN_LABELS[f],
            secs(q.p50),
            secs(q.p95),
            secs(q.p99)
        ));
    }
    let pk = |v: Option<f64>| v.map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into());
    lines.push(format!(
        "offload: st {} B/s · ld {} B/s · alloc fail {} /s",
        bytes(d.l2_wb),
        bytes(d.l2_rb),
        num(d.l2_drop)
    ));
    lines.push(format!(
        "peaks (this run): decode {} · prefill {} · single stream {} tok/s",
        pk(peaks.decode),
        pk(peaks.prefill),
        pk(peaks.decode_single)
    ));
    let s = d.stalls[f];
    lines.push(format!(
        "stalls: {} × {:.1}s in {} · preempt/s {} · abort/s {} · 5xx/s {}",
        s.count,
        if s.count > 0 {
            s.seconds / s.count as f64
        } else {
            0.0
        },
        WIN_LABELS[f],
        num(d.evict_rate),
        num(d.retract_rate),
        num(d.http_503_rate)
    ));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::parse;
    use std::time::Instant;

    // The headline carries model and engine label values verbatim
    // from the exporter. A hostile endpoint must not break the script-parsed
    // first line or inject terminal control sequences through it: sanitization
    // applies to every display-bound string, and its 256-character bound
    // holds here uniformly (real labels are far shorter).
    // Pools line reports absolute counts for KV (tokens) and the usage
    // percentage for the host offload tier, which vLLM reports without
    // an absolute count.
    #[test]
    fn pools_line_reports_absolute_counts_without_percentages() {
        let body = "# TYPE vllm:kv_cache_usage_perc gauge\n\
             vllm:kv_cache_usage_perc 0.0\n\
             # TYPE vllm:cache_config_info gauge\n\
             vllm:cache_config_info{kv_cache_size_tokens=\"655360\"} 1.0\n\
             # TYPE vllm:kv_offload_cpu_cache_usage_perc gauge\n\
             vllm:kv_offload_cpu_cache_usage_perc 0.0\n";
        let mut h = History::default();
        let t0 = Instant::now();
        h.push(t0, parse(body).unwrap());
        h.push(t0 + std::time::Duration::from_secs(1), parse(body).unwrap());
        let mut peaks = derive::Peaks::default();
        let text = snapshot_text(&h, &mut peaks).unwrap();
        let headline = text.lines().next().unwrap();
        assert!(
            headline.contains("KV 0/655k tok"),
            "headline was: {headline:?}"
        );
        assert!(
            headline.contains("host 0%"),
            "headline was: {headline:?}"
        );
        assert!(
            !headline.contains("mamba") && !headline.contains("slots"),
            "a vLLM-host-only pool leaked into the headline: {headline:?}"
        );
    }

    #[test]
    fn headline_sanitizes_hostile_label_values() {
        let body = "# TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running{engine=\"unified\",model_name=\"glm\u{1b}[2J\u{7f}\\nx\"} 4\n";
        let mut h = History::default();
        h.push(Instant::now(), parse(body).unwrap());
        let mut peaks = derive::Peaks::default();
        let text = snapshot_text(&h, &mut peaks).unwrap();
        let headline = text.lines().next().unwrap();
        // no raw control byte in the parsed first line, and the injected
        // newline is visible notation, so the headline stays one line
        assert!(
            !headline.chars().any(char::is_control),
            "headline was: {headline:?}"
        );
        for notation in ["\\x1b", "\\x7f", "\\n"] {
            assert!(
                headline.contains(notation),
                "missing {notation}: {headline:?}"
            );
        }
        assert!(
            headline.contains("glm"),
            "label text survives: {headline:?}"
        );
        assert!(
            headline.contains("unified"),
            "engine text survives: {headline:?}"
        );
    }
}
