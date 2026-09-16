use std::collections::BTreeMap;

/// Identity of one time series: metric family name + its (sorted) labels.
/// For histograms, the `le` label is stripped from bucket samples and the
/// `_sum`/`_count` suffixes are folded into the parent series key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeriesKey {
    pub name: String,
    pub labels: Vec<(String, String)>,
}

impl SeriesKey {
    pub fn has_label(&self, k: &str, v: &str) -> bool {
        self.labels.iter().any(|(lk, lv)| lk == k && lv == v)
    }
}

/// Renders the key in Prometheus exposition form, without escaping
/// label values (error messages only): the family name,
/// followed by `{k="v",...}` when labels are present
/// (e.g. `m:h{mode="decode"}`), so label-distinct families are
/// distinguishable.
impl std::fmt::Display for SeriesKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        for (i, (k, v)) in self.labels.iter().enumerate() {
            write!(f, "{}{k}=\"{v}\"", if i == 0 { "{" } else { "," })?;
        }
        if !self.labels.is_empty() {
            write!(f, "}}")?;
        }
        Ok(())
    }
}

/// Cumulative histogram state at one scrape, for one label combination.
#[derive(Debug, Clone, Default)]
pub struct HistSeries {
    /// Upper bounds, ascending; last entry is `f64::INFINITY`. Bounds are
    /// finite or +Inf — NaN bounds are dropped at the parse boundary, so
    /// any two bounds always order.
    pub le: Vec<f64>,
    /// Cumulative counts, parallel to `le`.
    pub counts: Vec<f64>,
    pub count: u64,
}

/// One scrape: simple gauge/counter series plus histogram families.
#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub simple: BTreeMap<SeriesKey, f64>,
    pub hist: BTreeMap<SeriesKey, HistSeries>,
}

impl Sample {
    pub fn gauge(&self, name: &str) -> Option<f64> {
        self.simple
            .iter()
            .find(|(k, _)| k.name == name)
            .map(|(_, v)| *v)
    }
}

#[derive(Default)]
struct HistBuilder {
    buckets: Vec<(f64, f64)>,
    count: u64,
}

/// Maximum number of series (simple + histogram families combined) accepted
/// in one scrape.
///
/// 16384 sits above the ~12k series measured on a real busy multi-rank
/// deployment, with headroom. [`BUCKET_CAP`] bounds one family's ladder
/// length. The two caps are independent, so together they do not bound
/// the aggregate: per-sample heap is limited by the 64 MiB scrape body cap, and
/// a hostile exporter spread across many families can pin single-digit GB
/// through the ring (~64 MiB x 150 ≈ 9.6 GB ceiling). Removing that ceiling
/// requires ingest-time aggregation, out of scope here (deferred).
pub const SERIES_CAP: usize = 16_384;

/// Maximum number of buckets accepted in a single histogram family per
/// scrape.
///
/// A real exporter's ladder stays under 100 buckets, so anything beyond 256
/// is a hostile payload funneling unbounded `_bucket` lines into one family,
/// which the series cap treats as a single series. A 257th `_bucket` record
/// rejects the whole sample, like [`SERIES_CAP`]. The cap counts `_bucket`
/// records only: the mandatory trailing `_sum`/`_count` lines are not buckets,
/// so a family at exactly the cap parses.
pub const BUCKET_CAP: usize = 256;

/// Parse a Prometheus text exposition payload into a [`Sample`].
///
/// Handles `# HELP`/`# TYPE` headers, `name{labels} value` records, label
/// escape sequences (`\\`, `\"`, `\n`), optional timestamps, and histogram
/// bucket ladders (`_bucket` with `le`, `_sum`, `_count`). Malformed records
/// are skipped line by line.
///
/// Errors when the payload exceeds [`SERIES_CAP`] series or any histogram
/// family exceeds [`BUCKET_CAP`] `_bucket` records. The whole sample
/// is rejected, never truncated; callers surface it like a failed scrape.
pub fn parse(body: &str) -> anyhow::Result<Sample> {
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut simple = BTreeMap::new();
    let mut hists: BTreeMap<SeriesKey, HistBuilder> = BTreeMap::new();

    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# ") {
            let mut it = rest.split_whitespace();
            if it.next() == Some("TYPE") {
                if let (Some(name), Some(ty)) = (it.next(), it.next()) {
                    types.insert(name.to_string(), ty.to_string());
                }
            }
            continue;
        }

        let (name, label_src, value_src) = match line.find('{') {
            Some(open) => match label_block_end(line, open) {
                Some(close) => (
                    &line[..open],
                    Some(&line[open + 1..close]),
                    line[close + 1..].trim(),
                ),
                None => continue,
            },
            None => match line.split_once(char::is_whitespace) {
                Some((n, v)) => (n, None, v.trim()),
                None => continue,
            },
        };

        let value = match parse_value(value_src) {
            Some(v) => v,
            None => continue,
        };
        // NaN and ±Inf gauge values are unreadable data, never extreme readings:
        // skip the record like any other malformed line, leaving the series
        // absent, so gauges, pools, and graph lanes see no data,
        // never a maximal-looking value
        if !value.is_finite() {
            continue;
        }

        let labels = label_src.map(parse_labels).unwrap_or_default();

        // Histogram families announce themselves via `# TYPE <base> histogram`;
        // their samples are the base name suffixed with _bucket/_sum/_count.
        let (base, part, le) = if let Some(b) = name.strip_suffix("_bucket") {
            // a usable bound is finite or +Inf (a NaN bound cannot be ordered)
            let le = labels
                .iter()
                .find(|(k, _)| k == "le")
                .and_then(|(_, v)| parse_value(v))
                .filter(|le| le.is_finite() || *le == f64::INFINITY);
            (b, 0, le)
        } else if let Some(b) = name.strip_suffix("_sum") {
            (b, 1, None)
        } else if let Some(b) = name.strip_suffix("_count") {
            (b, 2, None)
        } else {
            (name, 3, None)
        };
        let is_hist = part < 3 && types.get(base).map(String::as_str) == Some("histogram");

        if is_hist {
            let key = SeriesKey {
                name: base.to_string(),
                labels: labels.into_iter().filter(|(k, _)| k != "le").collect(),
            };
            // The cap guards `_bucket` records only: the trailing
            // `_sum`/`_count` lines are not buckets, so a family sitting
            // at exactly the cap completes instead of being rejected
            // (whole-sample rejection, same as the series cap)
            if part == 0
                && hists
                    .get(&key)
                    .is_some_and(|b| b.buckets.len() >= BUCKET_CAP)
            {
                anyhow::bail!(
                    "endpoint too large: {} has {} buckets (cap {BUCKET_CAP})",
                    key,
                    BUCKET_CAP + 1
                );
            }
            let b = hists.entry(key).or_default();
            match part {
                // a bucket without a usable `le` bound can't participate in
                // quantiles; skip it like any other malformed line
                0 => {
                    if let Some(le) = le {
                        b.buckets.push((le, value));
                    }
                }
                // _sum is parsed and discarded: mean latency is not displayed
                1 => {}
                _ => b.count = value.max(0.0) as u64,
            }
        } else {
            let key = SeriesKey {
                name: name.to_string(),
                labels,
            };
            simple.insert(key, value);
        }

        let series = simple.len() + hists.len();
        if series > SERIES_CAP {
            anyhow::bail!("endpoint too large: {series} series (cap {SERIES_CAP})");
        }
    }

    let hist = hists
        .into_iter()
        .map(|(k, b)| {
            let mut buckets = b.buckets;
            buckets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (le, counts): (Vec<f64>, Vec<f64>) = buckets.into_iter().unzip();
            (
                k,
                HistSeries {
                    le,
                    counts,
                    count: b.count,
                },
            )
        })
        .collect();

    Ok(Sample { simple, hist })
}

/// Returns the byte offset of the `}` that closes the label block whose
/// `{` sits at `open`. Braces and backslash-escaped characters (`\"`,
/// `\\`) inside quoted label values do not close the block. A block whose
/// quote never closes has no end: the record is dropped by the caller.
fn label_block_end(line: &str, open: usize) -> Option<usize> {
    let mut in_quotes = false;
    let mut chars = line[open + 1..].char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => in_quotes = !in_quotes,
            '}' if !in_quotes => return Some(open + 1 + i),
            // consume the escaped character so it cannot toggle quote state
            '\\' if in_quotes => {
                chars.next();
            }
            _ => {}
        }
    }
    None
}

fn parse_value(s: &str) -> Option<f64> {
    // value may be followed by an optional timestamp
    let tok = s.split_whitespace().next()?;
    match tok {
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        "Nan" | "NaN" | "nan" => Some(f64::NAN),
        v => v.parse().ok(),
    }
}

fn parse_labels(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    loop {
        // skip separators
        while matches!(chars.peek(), Some(',') | Some(' ') | Some('\t')) {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        let mut name = String::new();
        while let Some(&c) = chars.peek() {
            if c == '=' {
                break;
            }
            name.push(c);
            chars.next();
        }
        if chars.next() != Some('=') {
            break;
        }
        if chars.next() != Some('"') {
            break;
        }
        let mut value = String::new();
        loop {
            match chars.next() {
                Some('"') | None => break,
                Some('\\') => match chars.next() {
                    Some('n') => value.push('\n'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some(other) => value.push(other),
                    None => break,
                },
                Some(c) => value.push(c),
            }
        }
        out.push((name, value));
    }
    // canonical order: the same label set emitted in two different orders
    // must resolve to one series key, so an exporter rotating label order
    // between scrapes cannot fork the series and fabricate a spike
    out.sort();
    out
}
