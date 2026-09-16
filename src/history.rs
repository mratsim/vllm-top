use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::metrics::{Sample, SeriesKey};

/// Number of scrapes retained. At the 1s default interval this covers ~2min,
/// comfortably more than the longest (60s) window.
pub const RING_CAP: usize = 150;

pub struct Entry {
    pub t: Instant,
    pub sample: Arc<Sample>,
}

/// Counter delta across one scrape pair: a reset (the new reading below the old one) counts as rising from zero.
pub(crate) fn counter_delta(old: f64, new: f64) -> f64 {
    if new < old {
        new
    } else {
        new - old
    }
}

/// Rolling in-memory buffer of recent scrapes. This is the tool's only
/// "storage" — everything is discarded on exit.
#[derive(Default)]
pub struct History {
    buf: VecDeque<Entry>,
    pub first_seen: Option<Instant>,
    /// Cached `derive::scan_window` output, rebuilt on every push.
    /// `None` only before the first push.
    scan: Option<crate::derive::Graphs>,
}

impl History {
    pub fn push(&mut self, t: Instant, sample: Sample) {
        if let Some(last) = self.buf.back() {
            if t <= last.t {
                return; // non-monotonic or duplicate arrival: ignore
            }
        }
        if self.first_seen.is_none() {
            self.first_seen = Some(t);
        }
        if self.buf.len() == RING_CAP {
            self.buf.pop_front();
        }
        self.buf.push_back(Entry {
            t,
            sample: Arc::new(sample),
        });
        self.scan = Some(crate::derive::scan_window(self));
    }

    /// See `derive::scan_window` for the pass this cache holds.
    pub fn scan(&self) -> Option<&crate::derive::Graphs> {
        self.scan.as_ref()
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn last(&self) -> Option<&Entry> {
        self.buf.back()
    }

    pub fn get(&self, i: usize) -> Option<&Entry> {
        self.buf.get(i)
    }

    /// Index of the oldest sample still inside `window` (counting back from
    /// the most recent sample), or None if the buffer is empty.
    fn window_base(&self, window: Duration) -> Option<usize> {
        let cutoff = self.buf.back()?.t - window;
        Some(self.buf.iter().position(|e| e.t >= cutoff).unwrap_or(0))
    }

    /// Sum of per-series rates over `window` for every series matching
    /// `pred`. Computed per scrape-interval so a mid-window counter reset
    /// only clamps its own interval (value treated as rising from zero)
    /// instead of producing a negative spike or undercounting the window.
    /// A series absent from the earlier sample of a pair (a first
    /// appearance or a reappearance after a gap) starts as a new series:
    /// the interval pairs nothing, so no fabricated spike or fake reset
    /// enters the sum.
    pub fn rate_sum<F>(&self, pred: F, window: Duration) -> Option<f64>
    where
        F: Fn(&SeriesKey) -> bool,
    {
        let base = self.window_base(window)?;
        if self.buf.len() - base < 2 {
            return None;
        }
        let mut delta_sum = 0.0;
        let mut any = false;
        for pair in self.buf.iter().skip(base).collect::<Vec<_>>().windows(2) {
            let dt = (pair[1].t - pair[0].t).as_secs_f64();
            if dt <= 0.0 {
                continue;
            }
            for (k, v_new) in &pair[1].sample.simple {
                if !pred(k) {
                    continue;
                }
                let Some(v_old) = pair[0].sample.simple.get(k) else {
                    continue;
                };
                delta_sum += counter_delta(*v_old, *v_new);
                any = true;
            }
        }
        let dt_total = (self.buf.back().unwrap().t - self.buf[base].t).as_secs_f64();
        if dt_total <= 0.0 || !any {
            return None;
        }
        Some(delta_sum / dt_total)
    }

    /// Latest value of the first series matching `pred`.
    pub fn gauge_pred<F>(&self, pred: F) -> Option<f64>
    where
        F: Fn(&SeriesKey) -> bool,
    {
        self.buf
            .back()?
            .sample
            .simple
            .iter()
            .find(|(k, _)| pred(k))
            .map(|(_, v)| *v)
    }

    /// Sum of the latest values of every series matching `pred` — engine-
    /// wide totals for per-rank gauges (tp_rank/moe_ep_rank shards).
    pub fn sum_gauge<F>(&self, pred: F) -> Option<f64>
    where
        F: Fn(&SeriesKey) -> bool,
    {
        let last = self.buf.back()?;
        let mut sum = 0.0;
        let mut any = false;
        for (k, v) in &last.sample.simple {
            if pred(k) {
                sum += v;
                any = true;
            }
        }
        any.then_some(sum)
    }

    /// Windowed quantile of a histogram family, computed from bucket-count
    /// deltas over `window` (aggregating across label combinations, e.g.
    /// streaming + non-streaming). Bucket deltas pair by `le` boundary
    /// value, so a ladder that gains or loses a bucket between the two
    /// samples never shifts the pairing. Falls back to the cumulative
    /// since-start snapshot when the two samples share no boundary,
    /// when deltas are inconsistent (reset), or when too sparse.
    pub fn hist_quantile<F>(&self, pred: F, q: f64, window: Duration) -> Option<f64>
    where
        F: Fn(&SeriesKey) -> bool,
    {
        let base = self.window_base(window)?;
        let old = &self.buf[base];
        let new = self.buf.back()?;

        let (raw, delta_count) = hist_family_deltas(&old.sample, &new.sample, &pred);
        // a reset in any single label combination is a family reset:
        // merging first can net a negative delta against a positive
        // one and read a broken span as valid
        let reset = raw.iter().any(|(_, d)| *d < -0.5);
        let merged = merge_le(raw);
        if merged.is_empty() || reset || delta_count < QUANTILE_MIN_OBS {
            return self.hist_snapshot_quantile(&pred, q);
        }
        quantile_from_buckets(&merged, q)
    }

    /// Quantile from cumulative since-start buckets (slow-moving but always
    /// available; used as fallback and before a window has enough traffic).
    pub fn hist_snapshot_quantile<F>(&self, pred: &F, q: f64) -> Option<f64>
    where
        F: Fn(&SeriesKey) -> bool,
    {
        let last = self.buf.back()?;
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (k, h) in &last.sample.hist {
            if pred(k) {
                for (i, &le) in h.le.iter().enumerate() {
                    merged.push((le, h.counts.get(i).copied().unwrap_or(0.0)));
                }
            }
        }
        quantile_from_buckets(&merge_le(merged), q)
    }

    /// Slice of entries inside `window`, oldest first.
    pub fn window_entries(&self, window: Duration) -> Vec<Arc<Entry>> {
        match self.window_base(window) {
            Some(base) => self
                .buf
                .iter()
                .skip(base)
                .map(|e| {
                    Arc::new(Entry {
                        t: e.t,
                        sample: e.sample.clone(),
                    })
                })
                .collect(),
            None => Vec::new(),
        }
    }
}

/// Merge (le, value) pairs sharing the same bound, dropping NaN values.
/// The +Inf bound is kept: it counts toward the bucket total (observations
/// above the last finite bound) even though it cannot be interpolated into.
fn merge_le(mut pairs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    // keep finite bounds and +Inf (which counts toward the total);
    // drop anything else (NaN, -Inf)
    pairs.retain(|(le, v)| v.is_finite() && (le.is_finite() || *le == f64::INFINITY));
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (le, v) in pairs {
        match out.last_mut() {
            Some((last_le, last_v)) if *last_le == le => *last_v += v,
            _ => out.push((le, v)),
        }
    }
    out
}

/// Standard Prometheus `histogram_quantile` algorithm: linear interpolation
/// inside the bucket containing the q-rank. The lowest bucket is assumed to
/// start at 0; the +Inf bound is skipped (unbounded, unusable for
/// interpolation) but counts toward the total.
pub fn quantile_from_buckets(buckets: &[(f64, f64)], q: f64) -> Option<f64> {
    let total = buckets
        .iter()
        .filter(|(le, _)| le.is_infinite())
        .map(|(_, c)| *c)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .or_else(|| buckets.last().map(|(_, c)| *c))?;
    if total <= 0.0 {
        return None;
    }
    let rank = q.clamp(0.0, 1.0) * total;
    let mut prev_le = 0.0;
    let mut prev_count = 0.0;
    for (le, count) in buckets {
        if !le.is_finite() {
            break;
        }
        let count = count.max(0.0);
        if count >= rank {
            let span = count - prev_count;
            let est = if span <= 0.0 {
                *le
            } else {
                prev_le + (rank - prev_count) / span * (le - prev_le)
            };
            return Some(est);
        }
        prev_le = *le;
        prev_count = count;
    }
    buckets
        .iter()
        .rev()
        .find(|(le, _)| le.is_finite())
        .map(|(le, _)| *le)
}

/// Histogram bucket deltas between two samples for every family matching `pred`. Bucket
/// counts pair by `le` boundary value: a ladder that gains or loses a bucket never shifts
/// the pairing, and the paired deltas sum across label combinations. Returns the raw
/// paired deltas and the total observation-count delta across the families.
///
/// Cumulative counters make any two ring samples comparable this way: a quantile is computable
/// for a position inside the window (`hist_window_quantiles`) as well as for the window as a whole
/// (`History::hist_quantile`).
///
/// Minimum bucket-delta observation count a quantile interpolation needs:
/// the sparse-data bar covers the window quantile and the per-position
/// series (`History::hist_quantile`, `hist_window_quantiles`).
pub(crate) const QUANTILE_MIN_OBS: i64 = 5;

pub(crate) fn hist_family_deltas<F>(old: &Sample, new: &Sample, pred: &F) -> (Vec<(f64, f64)>, i64)
where
    F: Fn(&SeriesKey) -> bool,
{
    let mut merged: Vec<(f64, f64)> = Vec::new();
    let mut delta_count: i64 = 0;
    for (k, h_new) in &new.hist {
        if !pred(k) {
            continue;
        }
        if let Some(h_old) = old.hist.get(k) {
            // cumulative counts are comparable only at a boundary
            // carried by both ladders: walk the two sorted `le` lists
            // in lockstep and pair the shared bounds. A bound unique
            // to either side is ladder change, not data.
            let (mut i_new, mut i_old) = (0usize, 0usize);
            while let (Some(le_new), Some(le_old)) = (h_new.le.get(i_new), h_old.le.get(i_old)) {
                match le_new
                    .partial_cmp(le_old)
                    .expect("bucket bounds are finite or +Inf, so they always order")
                {
                    std::cmp::Ordering::Equal => {
                        let c_new = h_new.counts.get(i_new).copied().unwrap_or(0.0);
                        let c_old = h_old.counts.get(i_old).copied().unwrap_or(0.0);
                        merged.push((*le_new, c_new - c_old));
                        i_new += 1;
                        i_old += 1;
                    }
                    std::cmp::Ordering::Less => i_new += 1,
                    std::cmp::Ordering::Greater => i_old += 1,
                }
            }
            delta_count += h_new.count as i64 - h_old.count as i64;
        }
    }
    (merged, delta_count)
}

/// Plotted-series p95 of a histogram family, from the bucket deltas
/// (`hist_family_deltas`). One latency point of the per-position series
/// the latency plots draw: the span runs from the window's first sample
/// to the position's later sample, so the newest point is the window's
/// headline quantile. Returns None for a span too sparse to interpolate
/// or a ladder delta that went backwards (a reset). "Too sparse" means
/// fewer than QUANTILE_MIN_OBS observations, the bar `hist_quantile` applies.
/// A span never falls back to a snapshot, so a plotted point holds
/// measured data, never a snapshot echo.
pub(crate) fn hist_window_p95<F>(old: &Sample, new: &Sample, pred: &F) -> Option<f64>
where
    F: Fn(&SeriesKey) -> bool,
{
    let (raw, delta_count) = hist_family_deltas(old, new, pred);
    // a reset in any single label combination is a family reset:
    // merging first can net a negative delta against a positive
    // one and read a broken span as valid
    let reset = raw.iter().any(|(_, d)| *d < -0.5);
    let merged = merge_le(raw);
    if merged.is_empty() || reset || delta_count < QUANTILE_MIN_OBS {
        return None;
    }
    quantile_from_buckets(&merged, 0.95)
}
