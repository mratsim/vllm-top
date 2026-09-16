use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use vllm_top::history::History;
use vllm_top::metrics::parse;
use vllm_top::scrape::Shared;
use vllm_top::ui::{draw, Overlay, Ui};

fn shared_with_fixture() -> Arc<Shared> {
    let body = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/live.txt"
    ))
    .unwrap();
    let shared = Shared {
        history: Mutex::new(History::default()),
        last_ok: Mutex::new(Some(Instant::now())),
        last_error: Mutex::new(None),
        paused: AtomicBool::new(false),
        interval_ms: std::sync::atomic::AtomicU64::new(1000),
        peaks: Mutex::new(vllm_top::derive::Peaks {
            decode: Some(402.0),
            prefill: Some(812.0),
            decode_single: Some(134.0),
        }),
        alarms: Mutex::new(vllm_top::derive::Alarms::default()),
    };
    let mut h = shared.history.lock().unwrap();
    // two samples one scrape apart, the second with a distinct decode
    // reading, so the graphs draw a real interval instead of a degenerate
    // zero-duration, zero-delta sample
    let bumped = body.replace(
        "vllm:generation_tokens_total{engine=\"0\",model_name=\"DeepSeek-V4-Flash-Vision-Exp\"} 1.952826e+06",
        "vllm:generation_tokens_total{engine=\"0\",model_name=\"DeepSeek-V4-Flash-Vision-Exp\"} 1.954826e+06",
    );
    h.push(Instant::now(), parse(&body).unwrap());
    h.push(
        Instant::now() + std::time::Duration::from_secs(1),
        parse(&bumped).unwrap(),
    );
    drop(h);
    Arc::new(shared)
}

// History seeded from explicit payloads one scrape apart, for tests
// needing a custom latest sample.
fn shared_with_bodies(bodies: &[String]) -> Arc<Shared> {
    let shared = Shared {
        history: Mutex::new(History::default()),
        last_ok: Mutex::new(Some(Instant::now())),
        last_error: Mutex::new(None),
        paused: AtomicBool::new(false),
        interval_ms: std::sync::atomic::AtomicU64::new(1000),
        peaks: Mutex::new(vllm_top::derive::Peaks::default()),
        alarms: Mutex::new(vllm_top::derive::Alarms::default()),
    };
    let mut h = shared.history.lock().unwrap();
    let t0 = Instant::now();
    for (i, body) in bodies.iter().enumerate() {
        h.push(
            t0 + std::time::Duration::from_secs(i as u64),
            parse(body).unwrap(),
        );
    }
    drop(h);
    Arc::new(shared)
}

fn render_at(cols: u16, rows: u16, overlay: Overlay) -> String {
    let shared = shared_with_fixture();
    let backend = TestBackend::new(cols, rows);
    let mut terminal = Terminal::new(backend).unwrap();
    let ui = Ui {
        overlay,
        ..ui_default()
    };
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..rows {
        for col in 0..cols {
            let cell = &terminal.backend().buffer()[(col, row)];
            out.push(cell.symbol().chars().next().unwrap_or(' '));
        }
        out.push('\n');
    }
    out
}

fn ui_default() -> Ui {
    Ui::new(&vllm_top::args::Args {
        url: "http://localhost:30000".into(),
        api_key: None,
        insecure: false,
        interval: 1.0,
        theme: "gruvbox".into(),
        compact: false,
        once: false,
    })
}

#[test]
fn full_layout_renders_key_panels() {
    // 120x30
    let out = render_at(120, 30, Overlay::None);
    assert!(out.contains("vllm-top"), "header missing");
    assert!(out.contains("DeepSeek-V4-Flash-Vision-Exp"), "model missing");
    assert!(out.contains("RUNNING"), "hero missing");
    // hero rework: running | queue | token/s | time-to-first-token |
    // stalls | memory pools last, with the pools value absolute-only
    assert!(out.contains("TOKEN/s"), "token/s hero cell missing");
    assert!(
        out.contains("in-flight request stats"),
        "Generation caption missing:\n{out}"
    );
    assert!(
        out.contains("TIME TO FIRST TOKEN"),
        "ttft hero cell missing"
    );
    assert!(
        out.contains("full = requests queue"),
        "pools occupancy gloss missing"
    );
    // each pool renders on its own line, never joined by
    // " · " on a truncated row; the KV line leads the pool block
    let kv_line = out
        .lines()
        .find(|l| l.contains("KV 128k/1.0M tok"))
        .expect("KV pool line missing");
    assert!(
        !kv_line.contains("host"),
        "pool readings must not share a line:\n{kv_line}"
    );
    assert!(
        !out.contains("KV 128k/1.0M tok ·"),
        "the joined pool line is back"
    );
    // the hero prompt gloss is the full pair on its own hero line.
    // A bare contains check would pass via the latency subtitle
    // alone, so the wide hero row pin below carries the assertion
    assert!(
        out.contains("prompt 123.0k–200.0k"),
        "prompt gloss missing from the render"
    );
    // the cache-misses plot replaced the queue plot in the latency row
    assert!(out.contains("Cache misses"), "cache-misses plot missing");
    assert!(
        !out.contains("Waiting lines"),
        "waiting-lines panel still rendered"
    );
    // the plot subtitles carry the split size gloss: the prompt half
    // rides the TTFT plot, the computed half the cache-misses plot
    // (the latency subtitle holds the full pair, so the computed half needs two hits)
    assert!(
        out.contains("prompt 123.0k–200.0k tok"),
        "TTFT plot subtitle prompt gloss missing:\n{out}"
    );
    assert!(
        out.matches("computed 1.1k–6.7k tok").count() >= 2,
        "cache-misses plot subtitle computed gloss missing:\n{out}"
    );
    // the Cache panel's cached/computed throughput line renders
    assert!(
        out.contains("cached·computed"),
        "CACHE panel cached/computed line missing:\n{out}"
    );
    assert!(out.contains("Prefill"), "prefill graph missing");
    assert!(out.contains("Decode"), "decode graph missing");
    assert!(
        out.contains("aggregate · single-stream"),
        "decode two-line legend missing"
    );
    assert!(out.contains("TTFT"), "TTFT plot missing");
    // the pool graphs are gone; the pools text stays, absolute only
    assert!(!out.contains("Speed per stream"), "stale per-stream panel");
    assert!(
        !out.contains("host full is normal"),
        "stale pools-graph subtitle"
    );
    assert!(out.contains("MEMORY POOLS"), "hero pools cell missing");
    assert!(out.contains("128k/1.0M tok"), "KV absolute counts missing");
    assert!(out.contains("Latency"), "latency panel missing");
    assert!(out.contains("p50"), "percentiles missing");
    assert!(out.contains("Health & trouble"), "health panel missing");
    assert!(out.contains("Peaks"), "peaks panel missing");
    assert!(out.contains("402 tok/s"), "peak decode missing");
}

// The hero cells render left to right in the pinned order: running, queue,
// token/s (double width), time to first token, stalls, memory pools last.
// The hero's pair gloss must render un-truncated on the hero's own line:
// at 120 cols the token/s cell clips it, so the pin renders wide and looks
// only at the hero band (rows 1-6), where no other panel joins the pair.
#[test]
fn hero_pair_gloss_renders_untruncated_on_the_hero_row() {
    let wide = render_at(200, 30, Overlay::None);
    let pair = "prompt 123.0k–200.0k · computed 1.1k–6.7k tok";
    let hero_has_pair = wide
        .lines()
        .enumerate()
        .take(7)
        .skip(1)
        .any(|(_, line)| line.contains(pair));
    assert!(hero_has_pair, "hero pair gloss missing or clipped:\n{wide}");
}

#[test]
fn hero_cells_render_in_the_pinned_order() {
    let out = render_at(120, 30, Overlay::None);
    let pos = |needle: &str| {
        out.find(needle)
            .unwrap_or_else(|| panic!("hero label {needle} missing:\n{out}"))
    };
    let running = pos("RUNNING");
    let queue = pos("QUEUE");
    let rates = pos("TOKEN/s");
    let ttft = pos("TIME TO FIRST TOKEN");
    let stalls = pos("STALLS");
    let pools = pos("MEMORY POOLS");
    assert!(running < queue, "RUNNING must precede QUEUE");
    assert!(queue < rates, "QUEUE must precede TOKEN/s");
    assert!(rates < ttft, "TOKEN/s must precede TIME TO FIRST TOKEN");
    assert!(ttft < stalls, "TIME TO FIRST TOKEN must precede STALLS");
    assert!(stalls < pools, "STALLS must precede MEMORY POOLS");
}

// The health strip's alarm rows collapse into one dim line while quiet,
// then regain their full colored row after the channel's first firing.
// The latch is session-sticky: a fired alarm stays visible.
#[test]
fn alarm_rows_collapse_while_quiet_and_fire_full() {
    let quiet = render_at(120, 30, Overlay::None);
    assert!(
        quiet.contains("quiet: abort·5xx·alloc fail"),
        "quiet alarm line missing:\n{quiet}"
    );
    assert!(
        !quiet.contains("abort/s"),
        "a quiet alarm must not render its full row"
    );

    // the same fixture with the abort channel latched: its row returns
    let shared = shared_with_fixture();
    *shared.alarms.lock().unwrap() = vllm_top::derive::Alarms {
        abort: true,
        ..vllm_top::derive::Alarms::default()
    };
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| draw(f, &ui_default(), &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..120 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(
        out.contains("abort/s"),
        "a fired alarm must render its full row:\n{out}"
    );
    assert!(
        out.contains("quiet: 5xx·alloc fail"),
        "the fired channel must leave the quiet line:\n{out}"
    );
}

#[test]
fn compact_layout_renders_at_quarter_screen() {
    // 96x14
    let out = render_at(96, 14, Overlay::None);
    assert!(out.contains("vllm-top"));
    assert!(out.contains("RUNNING") || out.contains("running"));
    // compact collapses the subtitle lines but keeps hero + graph + latency
    assert!(out.contains("p50"));
}

#[test]
fn overlays_render() {
    let help = render_at(120, 30, Overlay::Help);
    assert!(help.contains("keymap"));
    assert!(help.contains("pause scraping"));
    let explain = render_at(120, 40, Overlay::Explain);
    assert!(explain.contains("What am I looking at?"));
    assert!(explain.contains("Time to first token"));
    assert!(explain.contains("p50 is a typical request"));
}

#[test]
fn tiny_terminal_does_not_panic() {
    let _ = render_at(40, 8, Overlay::None);
    let _ = render_at(20, 4, Overlay::None);
}

#[test]
fn insecure_tls_hint_renders_only_when_enabled() {
    let out = render_at(120, 30, Overlay::None);
    assert!(
        !out.contains("insecure TLS"),
        "hint shown without --insecure"
    );

    let shared = shared_with_fixture();
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    let ui = Ui {
        insecure: true,
        ..ui_default()
    };
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..120 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(
        out.contains("insecure TLS"),
        "hint missing with --insecure:\n{out}"
    );
}

#[test]
fn debug_print_full() {
    let out = render_at(120, 30, Overlay::None);
    println!("=====\n{out}\n=====");
}

#[test]
fn error_banner_renders_when_stale() {
    let shared = shared_with_fixture();
    *shared.last_error.lock().unwrap() = Some("scrape failed: connection refused".into());
    *shared.last_ok.lock().unwrap() = Some(Instant::now() - std::time::Duration::from_secs(30));
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    let ui = ui_default();
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..120 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(out.contains("connection lost"), "banner missing:\n{out}");
    assert!(out.contains("30s ago"));
}

#[test]
fn narrow_full_layout_truncates_with_ellipsis() {
    let shared = shared_with_fixture();
    let backend = TestBackend::new(76, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    // force full layout below the auto threshold to exercise the panels
    let ui = Ui {
        compact: Some(false),
        ..ui_default()
    };
    terminal.draw(|f| draw(f, &ui, &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..76 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    assert!(
        out.contains('\u{2026}'),
        "expected truncated lines with ellipsis:\n{out}"
    );
    // no line may overflow its panel border (wrap would have pushed content down)
    assert!(out.contains("Generation"));
    assert!(out.contains("Health & trouble"));
}

#[test]
fn debug_print_wide() {
    let out = render_at(230, 45, Overlay::None);
    println!("=====\n{out}\n=====");
}

// A non-finite gauge reading is no data: the UI renders the missing marker,
// never a "NaN" percentage or a maximal-looking bar.
#[test]
fn non_finite_gauge_reading_renders_as_no_data() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/live.txt"
    ))
    .unwrap();
    for reading in ["NaN", "+Inf", "-Inf"] {
        // newest scrape carries a non-finite KV usage reading
        // where the fixture has a finite one
        let stale = format!(
            "{}\nvllm:kv_cache_usage_perc{{engine=\"0\",model_name=\"DeepSeek-V4-Flash-Vision-Exp\"}} {reading}\n",
            fixture
        );
        let shared = shared_with_bodies(&[fixture.clone(), stale]);
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, &ui_default(), &shared)).unwrap();
        let mut out = String::new();
        for row in 0..30 {
            for col in 0..120 {
                out.push(
                    terminal.backend().buffer()[(col, row)]
                        .symbol()
                        .chars()
                        .next()
                        .unwrap_or(' '),
                );
            }
            out.push('\n');
        }
        assert!(
            !out.contains("NaN"),
            "a {reading} reading must render as no data, got:\n{out}"
        );
    }
}

// The latency plots carry labeled y-gridlines like the rate graphs do,
// working at any magnitude: a sub-second TTFT scale labels its seconds
// (0.1s), a tok/s scale labels k-formatted rates (10 tok/s). The sparse
// 2-sample fixture leaves both plots without data, so this fixture feeds
// four samples one scrape apart with real traffic.
#[test]
fn latency_plots_carry_labeled_gridlines_at_any_magnitude() {
    let body = |i: u32| {
        format!(
            "# TYPE vllm:time_to_first_token_seconds histogram\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.1\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.2\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"0.4\"}} {}\n\
             vllm:time_to_first_token_seconds_bucket{{le=\"+Inf\"}} {}\n\
             vllm:time_to_first_token_seconds_count {}\n\
             # TYPE vllm:prompt_tokens_by_source_total counter\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {}\n\
             # TYPE vllm:num_requests_running gauge\n\
             vllm:num_requests_running 2\n",
            2.0 * f64::from(i),
            5.0 * f64::from(i),
            6.0 * f64::from(i),
            6.0 * f64::from(i),
            6.0 * f64::from(i),
            10.0 * f64::from(i)
        )
    };
    let bodies: Vec<String> = (0..4).map(body).collect();
    let shared = shared_with_bodies(&bodies);
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| draw(f, &ui_default(), &shared)).unwrap();
    let mut out = String::new();
    for row in 0..30 {
        for col in 0..120 {
            out.push(
                terminal.backend().buffer()[(col, row)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' '),
            );
        }
        out.push('\n');
    }
    // the TTFT p95 lands at 0.34s, so a 0.1-step ladder prints "0.1s":
    // the sub-second regime an integer-only scale had left unlabeled
    assert!(
        out.contains("0.1s"),
        "fractional ttft gridline label missing:\n{out}"
    );
    // the cache-miss lane runs at 10 computed tok/s: the 10-step gridline
    // labels "10 tok/s"
    assert!(
        out.contains("10 tok/s"),
        "cache-miss gridline label missing:\n{out}"
    );
}

// The very first frame, before any scrape has landed, shows a stable
// zero state: the panels and their placeholders draw without panic,
// no graph is plotted from an empty ring, and no fabricated numbers
// appear. Two consecutive frames must render identically.
#[test]
fn cold_start_frame_renders_a_stable_zero_state() {
    let shared_for = || {
        Arc::new(Shared {
            history: Mutex::new(History::default()),
            last_ok: Mutex::new(None),
            last_error: Mutex::new(None),
            paused: AtomicBool::new(false),
            interval_ms: std::sync::atomic::AtomicU64::new(1000),
            peaks: Mutex::new(vllm_top::derive::Peaks::default()),
            alarms: Mutex::new(vllm_top::derive::Alarms::default()),
        })
    };
    let render = |cols: u16, rows: u16, shared: &Arc<Shared>| {
        let backend = TestBackend::new(cols, rows);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, &ui_default(), shared)).unwrap();
        let mut out = String::new();
        for row in 0..rows {
            for col in 0..cols {
                out.push(
                    terminal.backend().buffer()[(col, row)]
                        .symbol()
                        .chars()
                        .next()
                        .unwrap_or(' '),
                );
            }
            out.push('\n');
        }
        out
    };

    let shared = shared_for();
    let first = render(120, 30, &shared);
    let again = render(120, 30, &shared_for());
    assert_eq!(
        first, again,
        "the zero state must render identically every frame"
    );

    assert!(first.contains("vllm-top"), "header missing:\n{first}");
    assert!(first.contains("Engine status"), "hero frame missing");
    assert!(first.contains("Latency"), "latency panel missing");
    assert!(first.contains("Peaks"), "peaks panel missing");
    // the peaks panel shows the missing marker for every zero value
    assert!(
        first.contains("decode   —"),
        "peaks placeholders missing:\n{first}"
    );
    // an empty ring plots no graphs and shows no hero numbers
    assert!(!first.contains("Prefill"), "a graph rendered from no data");
    assert!(!first.contains("Decode"), "a graph rendered from no data");
    assert!(
        !first.contains("RUNNING"),
        "hero numbers rendered from no data"
    );
    assert!(!first.contains("NaN"), "a NaN reached the screen:\n{first}");

    // A terminal too small for the panels' combined minimum height has
    // more than one valid layout split, and which split the solver lands
    // on is not stable across runs
    // (solver variable ids are process global, so parallel tests shift them).
    // Content is therefore pinned
    // at the size above, whose panel minimums exactly fill the screen,
    // while smaller terminals get a no-panic check only.
    let _ = render(40, 8, &shared_for());
}
