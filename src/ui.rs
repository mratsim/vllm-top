use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CLine};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::Frame;

use crate::args::Args;
use crate::derive::{Derived, WIN_LABELS};
use crate::scrape::Shared;
use crate::theme::{usage_color, Theme, THEMES};

/// Minimum terminal size (rows, cols) for the full layout; below either, the
/// compact layout takes over (subtitles collapse first, then graphs).
const FULL_MIN_ROWS: u16 = 26;
const FULL_MIN_COLS: u16 = 90;

pub struct Ui {
    pub theme_idx: usize,
    /// None = auto from terminal size, Some = forced
    pub compact: Option<bool>,
    pub graphs_on: bool,
    pub window_focus: usize,
    pub paused_since: Option<Instant>,
    /// `--insecure` is active: flagged in the header so unverified TLS is visible
    pub insecure: bool,
    pub overlay: Overlay,
    pub scroll: u16,
    pub interval: f64,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Overlay {
    None,
    Help,
    Explain,
}

impl Ui {
    pub fn new(args: &Args) -> Self {
        let theme_idx = THEMES
            .iter()
            .position(|t| t.name == args.theme)
            .unwrap_or(0);
        Self {
            theme_idx,
            compact: if args.compact { Some(true) } else { None },
            graphs_on: true,
            window_focus: 2,
            paused_since: None,
            insecure: args.insecure,
            overlay: Overlay::None,
            scroll: 0,
            interval: args.interval.clamp(0.5, 10.0),
        }
    }

    pub fn theme(&self) -> &'static Theme {
        &THEMES[self.theme_idx]
    }
}

pub fn run(
    terminal: &mut ratatui::DefaultTerminal,
    shared: Arc<Shared>,
    ui: &mut Ui,
) -> anyhow::Result<()> {
    loop {
        if event_ready()? {
            if let crossterm::event::Event::Key(k) = crossterm::event::read()? {
                if k.kind == crossterm::event::KeyEventKind::Press
                    && handle_key(ui, &shared, k.code)
                {
                    return Ok(());
                }
            }
        }

        terminal.draw(|f| draw(f, ui, &shared))?;
    }
}

fn event_ready() -> anyhow::Result<bool> {
    Ok(crossterm::event::poll(Duration::from_millis(120))?)
}

fn handle_key(ui: &mut Ui, shared: &Arc<Shared>, code: crossterm::event::KeyCode) -> bool {
    use crossterm::event::KeyCode::*;
    match code {
        Char('q') | Esc => true,
        Char(' ') => {
            let now_paused = !shared.paused.load(Ordering::Relaxed);
            shared.paused.store(now_paused, Ordering::Relaxed);
            ui.paused_since = now_paused.then_some(Instant::now());
            false
        }
        Char('c') => {
            let auto = ui.compact.unwrap_or_else(auto_compact);
            ui.compact = Some(!auto);
            false
        }
        Char('g') => {
            ui.graphs_on = !ui.graphs_on;
            false
        }
        Char('t') => {
            ui.theme_idx = (ui.theme_idx + 1) % THEMES.len();
            false
        }
        Char(c @ '1'..='3') => {
            ui.window_focus = c.to_digit(10).unwrap_or(3) as usize - 1;
            false
        }
        Char('e') => {
            ui.overlay = if ui.overlay == Overlay::Explain {
                Overlay::None
            } else {
                Overlay::Explain
            };
            ui.scroll = 0;
            false
        }
        Char('?') => {
            ui.overlay = if ui.overlay == Overlay::Help {
                Overlay::None
            } else {
                Overlay::Help
            };
            false
        }
        Char('+') | Char('=') => {
            ui.interval = (ui.interval * 2.0).min(10.0);
            shared
                .interval_ms
                .store((ui.interval * 1000.0) as u64, Ordering::Relaxed);
            false
        }
        Char('-') => {
            ui.interval = (ui.interval / 2.0).max(0.5);
            shared
                .interval_ms
                .store((ui.interval * 1000.0) as u64, Ordering::Relaxed);
            false
        }
        Down | Char('j') => {
            if ui.overlay == Overlay::Explain {
                ui.scroll = ui.scroll.saturating_add(1);
            }
            false
        }
        Up | Char('k') => {
            ui.scroll = ui.scroll.saturating_sub(1);
            false
        }
        _ => false,
    }
}

fn auto_compact() -> bool {
    match crossterm::terminal::size() {
        Ok((cols, rows)) => rows < FULL_MIN_ROWS || cols < FULL_MIN_COLS,
        Err(_) => false,
    }
}

// ---------------------------------------------------------------- rendering

pub fn draw(f: &mut Frame, ui: &Ui, shared: &Arc<Shared>) {
    let t = ui.theme();
    f.render_widget(Block::new().style(Style::new().bg(t.bg)), f.area());

    let history = shared.history.lock().unwrap();
    let d = crate::derive::derive(&history, ui.window_focus);
    let uptime = history.first_seen.map(|t| t.elapsed());
    drop(history);
    let age = shared.last_ok.lock().unwrap().map(|t| t.elapsed());
    let err = shared.last_error.lock().unwrap().clone();

    let compact = ui.compact.unwrap_or_else(|| {
        let a = f.area();
        a.height < FULL_MIN_ROWS || a.width < FULL_MIN_COLS
    });

    let mut constraints: Vec<Constraint> = vec![
        Constraint::Length(1),
        // the full hero needs four content rows: the token/s cell alone carries three
        // (prefill rate, decode rate, prompt-size context)
        // hero grows one row: the pools cell stacks one line per pool plus
        // the occupancy gloss under the label
        Constraint::Length(if compact { 3 } else { 7 }),
    ];
    let mut graph_i: Option<usize> = None;
    if ui.graphs_on {
        graph_i = Some(constraints.len());
        constraints.push(if compact {
            Constraint::Min(3)
        } else {
            Constraint::Min(7)
        });
    }
    let lat_i = constraints.len();
    constraints.push(Constraint::Length(if compact { 5 } else { 8 }));
    let det_i = constraints.len();
    constraints.push(if compact {
        Constraint::Length(3)
    } else {
        // detail sizes to its content (capped); all spare rows go to the
        // graphs row above, which is the only Min and absorbs the rest
        let alarms = *shared.alarms.lock().unwrap();
        Constraint::Length(detail_height(
            d.as_ref(),
            ui.window_focus,
            &alarms,
            &shared.peaks.lock().unwrap(),
        ))
    });

    let chunks = Layout::vertical(constraints).split(f.area());

    let peaks = *shared.peaks.lock().unwrap();
    draw_header(f, ui, t, chunks[0], d.as_ref(), uptime, age);
    draw_hero(f, ui, t, chunks[1], d.as_ref(), compact);
    if let Some(gi) = graph_i {
        if let Some(d) = &d {
            if compact {
                draw_rate_graph(f, t, chunks[gi], d, RateGraph::Decode, &peaks);
            } else {
                draw_graphs_row(f, t, chunks[gi], d, &peaks);
            }
        }
    }
    draw_latency(f, ui, t, chunks[lat_i], d.as_ref(), compact);
    draw_detail(f, ui, t, chunks[det_i], d.as_ref(), compact, shared);

    match ui.overlay {
        Overlay::None => {}
        Overlay::Help => draw_help(f, t, f.area()),
        Overlay::Explain => draw_explain(f, ui, t, f.area()),
    }

    if let Some(paused_at) = ui.paused_since {
        let secs = paused_at.elapsed().as_secs();
        banner(
            f,
            f.area(),
            format!("⏸ paused {secs}s — press space to resume"),
            t.warn,
        );
    } else if err.is_some() && (d.is_none() || age_is_stale(age, ui)) {
        let e = err.unwrap_or_default();
        let a = age.map(|a| a.as_secs()).unwrap_or(0);
        banner(
            f,
            f.area(),
            format!("⚠ connection lost — retrying… last data {a}s ago — {e}"),
            t.bad,
        );
    }
}

fn age_is_stale(age: Option<Duration>, ui: &Ui) -> bool {
    match age {
        None => true,
        Some(a) => a > Duration::from_secs_f64(ui.interval * 3.0 + 1.0),
    }
}

fn banner(f: &mut Frame, area: Rect, msg: String, color: ratatui::style::Color) {
    let area = Rect {
        x: area.x,
        y: area.y + area.height.saturating_sub(1),
        width: area.width,
        height: 1,
    };
    let line = Line::from(Span::styled(
        msg,
        Style::new().fg(color).add_modifier(Modifier::BOLD),
    ));
    f.render_widget(Paragraph::new(line), area);
}

// ---- header ---------------------------------------------------------------

fn draw_header(
    f: &mut Frame,
    ui: &Ui,
    t: &Theme,
    area: Rect,
    d: Option<&Derived>,
    uptime: Option<Duration>,
    age: Option<Duration>,
) {
    let mut spans = vec![Span::styled(
        " vllm-top",
        Style::new().fg(t.accent).add_modifier(Modifier::BOLD),
    )];
    if let Some(d) = d {
        spans.push(Span::styled(
            format!(" · {} ({})", d.model, d.engine),
            Style::new().fg(t.fg),
        ));
        if let Some(up) = uptime {
            spans.push(Span::styled(
                format!(" · up {}", fmt_dur(up)),
                Style::new().fg(t.dim),
            ));
        }
    }
    spans.push(Span::styled(
        format!(" · poll {:.1}s", ui.interval),
        Style::new().fg(t.dim),
    ));
    if ui.insecure {
        spans.push(Span::styled(
            " · \u{26a0} insecure TLS",
            Style::new().fg(t.warn),
        ));
    }
    if let Some(a) = age {
        if a > Duration::from_secs(2) {
            spans.push(Span::styled(
                format!(" · data {:.0}s old", a.as_secs_f64()),
                Style::new().fg(t.warn),
            ));
        }
    }
    spans.push(Span::styled(
        "  [? help · e explain]",
        Style::new().fg(t.dim),
    ));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ---- hero strip ------------------------------------------------------------

fn draw_hero(f: &mut Frame, ui: &Ui, t: &Theme, area: Rect, d: Option<&Derived>, compact: bool) {
    let focus = ui.window_focus;
    let block = block(t, "Engine status", None);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // the token/s cell keeps double width (its rate lines); time-to-first-
    // token is sized to its long label, and the memory-pools cell fits
    // the stacked per-pool readings, the longest pool line being 18 columns
    let cols = Layout::horizontal([
        Constraint::Percentage(12),
        Constraint::Percentage(12),
        Constraint::Percentage(27),
        Constraint::Percentage(18),
        Constraint::Percentage(12),
        Constraint::Percentage(19),
    ])
    .split(inner);

    let mut cells: Vec<(String, Vec<Line<'static>>)> = Vec::new();
    if let Some(d) = d {
        let kv = |label: &str, value: &str, sub: &str, style: Style| {
            (
                label.to_string(),
                vec![
                    Line::from(Span::styled(format!(" {}", value), style)),
                    Line::from(Span::styled(format!(" {}", sub), Style::new().fg(t.dim))),
                ],
            )
        };
        let f_run = Style::new().fg(t.fg).add_modifier(Modifier::BOLD);
        cells.push(kv(
            "RUNNING",
            &fmt_num(d.running),
            "being answered now",
            f_run,
        ));
        let q_style = match d.queue.unwrap_or(0.0) {
            q if q > 20.0 => Style::new().fg(t.bad).add_modifier(Modifier::BOLD),
            q if q > 5.0 => Style::new().fg(t.warn).add_modifier(Modifier::BOLD),
            _ => Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        };
        cells.push(kv("QUEUE", &fmt_num(d.queue), "waiting to start", q_style));
        let rate_line = |tag: &str, instant: Option<f64>, win: &[Option<f64>; 3], color| {
            Line::from(vec![
                Span::styled(format!(" {tag} "), Style::new().fg(t.dim)),
                Span::styled(
                    format!("{:.0} tok/s", instant.unwrap_or(0.0)),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  ·  {}", triple(win)), Style::new().fg(t.dim)),
            ])
        };
        let mut rates = vec![
            rate_line("prefill", d.prefill_instant, &d.prefill_rate, t.s2),
            rate_line("decode", d.decode_instant, &d.decode_rate, t.s1),
        ];
        // prompt-size context, the same pair gloss the latency row subtitle
        // carries: prompt sizes as submitted plus the cache misses' sizes
        if let Some(gloss) = prompt_computed_gloss(d) {
            rates.push(Line::from(Span::styled(
                format!(" {gloss}"),
                Style::new().fg(t.dim),
            )));
        }
        cells.push(("TOKEN/s".into(), rates));
        let ttft = d.ttft[focus].p95;
        cells.push(kv(
            "TIME TO FIRST TOKEN",
            &fmt_secs(ttft),
            "p95 · first token",
            Style::new()
                .fg(latency_color(t, ttft, &d.ttft[focus]))
                .add_modifier(Modifier::BOLD),
        ));
        let s = d.stalls[focus];
        let stall_style = if s.count > 0 {
            Style::new().fg(t.bad).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(t.good)
        };
        cells.push(kv(
            "STALLS",
            &if s.count > 0 {
                format!("{}×", s.count)
            } else {
                "0".into()
            },
            &format!("{:.1}s frozen in {}", s.seconds, WIN_LABELS[focus]),
            stall_style,
        ));
        // one pool per line: the joined single line truncates at live
        // widths. Each pool line shows the absolute reading, fullness
        // colored. The host tier stays out of the cell: it idles
        // near 100% by design (an LRU write-through cache), entries
        // recycled in place: a stacked host line would look like an alarm
        // the engine never actually raises
        let mut pool_lines: Vec<Line<'static>> = d
            .pools
            .iter()
            .filter(|p| p.name != "host")
            .map(|p| {
                let style = Style::new()
                    .fg(usage_color(t, p.usage))
                    .add_modifier(Modifier::BOLD);
                match pool_value(p) {
                    Some(v) => Line::from(Span::styled(format!(" {} {v}", p.name), style)),
                    None => Line::from(Span::styled(format!(" {}", p.name), style)),
                }
            })
            .collect();
        pool_lines.push(Line::from(Span::styled(
            " full = requests queue",
            Style::new().fg(t.dim),
        )));
        cells.push(("MEMORY POOLS".into(), pool_lines));
    }

    for (i, cell) in cells.iter().enumerate() {
        if i >= cols.len() {
            break;
        }
        let area = cols[i];
        let label = if compact {
            cell.0.to_lowercase()
        } else {
            cell.0.clone()
        };
        let mut lines = vec![truncate_line(
            Line::from(Span::styled(label, Style::new().fg(t.dim))),
            area.width as usize,
        )];
        for l in &cell.1 {
            lines.push(truncate_line(l.clone(), area.width as usize));
        }
        f.render_widget(Paragraph::new(lines), area);
    }
}

fn latency_color(
    t: &Theme,
    p95: Option<f64>,
    _q: &crate::derive::Quantiles,
) -> ratatui::style::Color {
    match p95 {
        Some(v) if v > 2.0 => t.bad,
        Some(v) if v > 0.5 => t.warn,
        Some(_) => t.good,
        None => t.dim,
    }
}

// ---- graphs -----------------------------------------------------------------

fn draw_graphs_row(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    d: &Derived,
    peaks: &crate::derive::Peaks,
) {
    let cols =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);
    draw_rate_graph(f, t, cols[0], d, RateGraph::Prefill, peaks);
    draw_rate_graph(f, t, cols[1], d, RateGraph::Decode, peaks);
}

/// Which rate series a standalone graph shows.
pub enum RateGraph {
    Prefill,
    Decode,
}

/// Calibration gridlines for an observed peak, at any magnitude: one line
/// per round step (1, 0.1, 0.01, ... below 1), the last step above the peak,
/// scale topped 10% higher. The lowest line carries the axis unit
/// (ms-scale latencies label 0.4s, rates label 10k tok/s).
fn grid_for_scale(scale: f64, t: &Theme) -> (Vec<(f64, ratatui::style::Color)>, f64) {
    if scale > 0.0 {
        let step = 10.0_f64.powf(scale.log10().floor());
        let top_line = (scale / step).ceil() * step;
        let lines: Vec<(f64, ratatui::style::Color)> = (1..=(top_line / step) as i64)
            .map(|m| (m as f64 * step, t.dim))
            .collect();
        (lines, top_line * 1.1)
    } else {
        (Vec::new(), 1.0)
    }
}

/// Bottom-axis tick marks for one graph: a red tick on every stalled
/// interval and, where `evict` is given, an orange tick in the paired color
/// marking intervals whose eviction rate advanced. Both ticks on one
/// interval render as the stall (stalls are the emergency).
#[derive(Clone, Copy)]
struct Ticks<'a> {
    stall: &'a [bool],
    evict: Option<(&'a [f64], ratatui::style::Color)>,
}

/// Dress a mini graph wears besides its fill: the axis unit on the lowest
/// gridline label, the stall/eviction ticks, an optional overlay series
/// drawn as a line over the fill, and the calibration gridlines.
struct PlotDress<'a> {
    unit: &'a str,
    ticks: Option<Ticks<'a>>,
    overlay: Option<(&'a [Option<f64>], ratatui::style::Color)>,
    ref_lines: &'a [(f64, ratatui::style::Color)],
}

fn draw_rate_graph(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    d: &Derived,
    which: RateGraph,
    peaks: &crate::derive::Peaks,
) {
    let g = &d.graphs;
    let (vals_src, instant, color, title, gloss, sub) = match which {
        RateGraph::Prefill => (
            &g.prefill.vals,
            d.prefill_instant,
            t.s2,
            "Prefill",
            "prompt processing",
            Some(Line::from(Span::styled(
                " bursts freeze streams ".to_string(),
                Style::new().fg(t.dim),
            ))),
        ),
        RateGraph::Decode => (
            &g.decode.vals,
            d.decode_instant,
            t.s1,
            "Decode",
            "token generation",
            // legend words carry their lines' colors: the aggregate fill
            // renders in s1 and the single-stream line in accent
            Some(Line::from(vec![
                Span::styled(
                    " aggregate ",
                    Style::new().fg(t.s1).add_modifier(Modifier::BOLD),
                ),
                Span::styled("\u{b7} ", Style::new().fg(t.dim)),
                Span::styled(
                    "single-stream ",
                    Style::new().fg(t.s3).add_modifier(Modifier::BOLD),
                ),
            ])),
        ),
    };
    let vals: Vec<Option<f64>> = vals_src.iter().map(|v| Some(*v)).collect();
    // scale is the larger of the in-window max and the session peak;
    // grid_for_scale adds headroom above the top gridline
    let vmax = vals.iter().filter_map(|v| *v).fold(0.0_f64, f64::max);
    let peak = match which {
        RateGraph::Prefill => peaks.prefill,
        RateGraph::Decode => peaks.decode,
    };
    let scale = vmax.max(peak.unwrap_or(0.0));
    let (ref_lines, ymax) = grid_for_scale(scale, t);
    let title = Line::from(vec![
        Span::styled(
            format!(" {title} "),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("({gloss}) "), Style::new().fg(t.dim)),
        Span::styled(
            format!(
                "\u{2014} {} {:.0} \u{b7} peak {:.0} ",
                match which {
                    RateGraph::Prefill => "now",
                    RateGraph::Decode => "aggregate now",
                },
                instant.unwrap_or(0.0),
                scale
            ),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "\u{25cf} ",
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]);
    // the decode legend line gains the window's per-stream stats:
    // mean and max of the single-stream series join the legend
    // on the bottom border, keeping the top title short on laptops
    let sub = match which {
        RateGraph::Prefill => sub,
        RateGraph::Decode => {
            let (sum, peak, n) = g
                .per_stream
                .iter()
                .filter_map(|v| *v)
                .fold((0.0_f64, 0.0_f64, 0_usize), |(s, p, n), v| {
                    (s + v, p.max(v), n + 1)
                });
            let mut spans = match sub {
                Some(line) => line.spans,
                None => Vec::new(),
            };
            if n > 0 {
                spans.push(Span::styled(
                    format!(" \u{b7} avg {:.0} \u{b7} peak {:.0} ", sum / n as f64, peak),
                    Style::new().fg(t.s3).add_modifier(Modifier::BOLD),
                ));
            }
            Some(Line::from(spans))
        }
    };
    let block = block_titled(t, title, sub);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // the decode canvas carries a second line for the per-stream speed
    // (decode rate divided by running requests); aggregate is at minimum
    // the single-stream rate, so the two share a scale
    let (ticks, overlay) = match which {
        RateGraph::Prefill => (None, None),
        RateGraph::Decode => (
            Some(Ticks {
                stall: &g.stall,
                evict: None,
            }),
            Some((g.per_stream.as_slice(), t.s3)),
        ),
    };
    mini_line_graph(
        f,
        inner,
        &vals,
        &g.dt,
        ymax,
        color,
        PlotDress {
            unit: " tok/s",
            ticks,
            overlay,
            ref_lines: &ref_lines,
        },
    );
}

/// TTFT plot (left of the latency row's two plots): the p95 series shaded,
/// in seconds. The table's row subtitle repeats the pair gloss.
fn draw_ttft_plot(f: &mut Frame, t: &Theme, area: Rect, d: &Derived) {
    let g = &d.graphs;
    let ticks = Ticks {
        stall: &g.stall,
        evict: Some((&g.evictions.vals, t.warn)),
    };
    let scale = g.ttft_p95.iter().filter_map(|v| *v).fold(0.0_f64, f64::max);
    let (ref_lines, ymax) = grid_for_scale(scale, t);
    // latency is a lower-is-better metric, so the fill color follows
    // the latest p95: green only while genuinely good, amber or red
    // as it worsens; a static green fill reads as healthy
    // at any magnitude, which is exactly wrong
    let p95_now = d.ttft[2].p95;
    let lat_color = latency_color(t, p95_now, &d.ttft[2]);
    let title = Line::from(vec![
        Span::styled(" TTFT ", Style::new().fg(t.fg).add_modifier(Modifier::BOLD)),
        // legend: the p95 series (the only series plotted)
        Span::styled(
            "\u{25cf} ",
            Style::new().fg(lat_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled("p95", Style::new().fg(t.dim)),
    ]);
    // the subtitle carries the prompt half of the size gloss: the cache-miss
    // plot carries the computed half, the table's subtitle holds the full pair
    let sub = prompt_size_span(d)
        .map(|gloss| Line::from(Span::styled(format!(" {gloss} "), Style::new().fg(t.dim))));
    let block = block_titled(t, title, sub);
    let inner = block.inner(area);
    f.render_widget(block, area);
    mini_line_graph(
        f,
        inner,
        &g.ttft_p95,
        &g.dt,
        ymax,
        lat_color,
        PlotDress {
            unit: "s",
            ticks: Some(ticks),
            overlay: None,
            ref_lines: &ref_lines,
        },
    );
}

/// Cache-misses plot (right of the latency row's two plots): computed prompt
/// tokens/s per interval, on a tok/s scale with the same stall/eviction ticks.
/// These are the prefix-cache misses, the prompt work the device chewed raw.
fn draw_cache_miss_plot(f: &mut Frame, t: &Theme, area: Rect, d: &Derived) {
    let g = &d.graphs;
    let ticks = Ticks {
        stall: &g.stall,
        evict: Some((&g.evictions.vals, t.warn)),
    };
    let vals = &g.cache_misses;
    let vmax = vals.iter().filter_map(|v| *v).fold(0.0_f64, f64::max);
    let (ref_lines, ymax) = grid_for_scale(vmax, t);
    let title = Line::from(vec![
        Span::styled(
            " Cache misses ",
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "\u{25cf}",
            Style::new().fg(t.s2).add_modifier(Modifier::BOLD),
        ),
    ]);
    let sub = computed_size_span(d)
        .map(|gloss| Line::from(Span::styled(format!(" {gloss} "), Style::new().fg(t.dim))));
    let block = block_titled(t, title, sub);
    let inner = block.inner(area);
    f.render_widget(block, area);
    mini_line_graph(
        f,
        inner,
        vals,
        &g.dt,
        ymax,
        t.s2,
        PlotDress {
            unit: " tok/s",
            ticks: Some(ticks),
            overlay: None,
            ref_lines: &ref_lines,
        },
    );
}

fn mini_line_graph(
    f: &mut Frame,
    area: Rect,
    vals: &[Option<f64>],
    dt: &[f64],
    ymax: f64,
    color: ratatui::style::Color,
    dress: PlotDress<'_>,
) {
    // every graph in the app spans the same trailing window
    let window_secs = crate::derive::WINDOWS[2].as_secs_f64();
    let unit = dress.unit;
    let ticks = dress.ticks;
    let overlay = dress.overlay;
    let ref_lines = dress.ref_lines;
    if dt.is_empty() || ymax <= 0.0 || window_secs <= 0.0 || area.width == 0 || area.height == 0 {
        return;
    }
    // coordinates are braille dots (2 per char column, 4 per char row);
    // the newest bin is held out to the last dot column so the graph
    // reaches the right border
    let w_dots = (area.width as f64) * 2.0;
    let h_dots = (area.height as f64) * 4.0;
    // fixed wall-clock axis: the window is always `window_secs` wide,
    // newest sample pinned to the right edge
    let span: f64 = dt.iter().sum::<f64>();
    let axis_x = |cum: f64| {
        ((w_dots - 1.0) - (span - cum) / window_secs * (w_dots - 1.0)).clamp(0.0, w_dots - 1.0)
    };
    // each sample holds for its whole scrape interval; fill columns
    // interpolate between interval ends to form a solid histogram. An absent value
    // (no measured data at that position) breaks the fill, so the gap is drawn, never bridged.
    let mut tops: Vec<(usize, f64, f64)> = Vec::new();
    let mut tick_dots: Vec<(f64, ratatui::style::Color)> = Vec::new();
    let mut x = 0.0;
    for (i, v) in vals.iter().enumerate() {
        let cx = axis_x(x);
        if let Some(v) = v {
            let h = ((v / ymax) * h_dots * 0.96).min(h_dots * 0.96);
            tops.push((i, cx, h));
        }
        let stalled = ticks.is_some_and(|t| t.stall.get(i).copied().unwrap_or(false));
        if stalled {
            tick_dots.push((cx, ratatui::style::Color::Red));
        } else if let Some((rates, color)) = ticks.and_then(|t| t.evict) {
            // the eviction tick rides the pair's color; a zero rate draws
            // nothing
            if rates.get(i).is_some_and(|&rate| rate > 0.0) {
                tick_dots.push((cx, color));
            }
        }
        x += dt.get(i).copied().unwrap_or(1.0);
    }
    let mut cols: Vec<(f64, f64)> = Vec::new();
    for w in tops.windows(2) {
        let (i0, x0, h0) = w[0];
        let (i1, x1, h1) = w[1];
        if i1 != i0 + 1 {
            continue;
        }
        let mut cx = x0;
        while cx < x1 {
            let frac = (cx - x0) / (x1 - x0).max(1.0);
            let h = h0 + (h1 - h0) * frac;
            cols.push((cx, h.max(1.0)));
            cx += 1.0;
        }
    }
    // the newest bin is still in progress: hold its value out to the
    // right edge so the graph always touches the present moment, but only
    // when the newest position itself has data
    if let Some(&(i, start, h)) = tops.last() {
        if i + 1 == vals.len() {
            let h = h.max(1.0);
            for cx in start as i64..=(w_dots as i64 - 1) {
                cols.push((cx as f64, h));
            }
        }
    }
    // the overlay line (single-stream speed, p50) connects its own measured
    // points and, like the fill, breaks where a position has no measured
    // value. Paint order here is load bearing: braille cells keep the last
    // shape drawn, so the overlay must come after the fill or the aggregate
    // erases it where the two coincide
    let mut overlay_tops: Vec<(usize, f64, f64)> = Vec::new();
    let mut overlay_color = color;
    if let Some((series, ocolor)) = overlay {
        overlay_color = ocolor;
        let mut x = 0.0;
        for (i, v) in series.iter().enumerate() {
            if let Some(v) = v {
                let h = ((v / ymax) * h_dots * 0.96).min(h_dots * 0.96);
                overlay_tops.push((i, axis_x(x), h));
            }
            x += dt.get(i).copied().unwrap_or(1.0);
        }
    }
    // dashed gridlines at each step multiple with faded labels; the lowest
    // line carries the unit, the rest are bare numbers. A label lands in one
    // of canvas_rows - 1 distinct slots (the canvas maps label floats that way),
    // so gridlines one step apart can share a slot when the canvas is short.
    // Keep the lowest label per slot and drop the rest, or a later label
    // punches through an earlier one and leaves stray characters behind.
    let grid: Vec<(f64, f64)> = ref_lines
        .iter()
        .map(|(v, _)| (*v, (*v / ymax * h_dots * 0.96).clamp(1.0, h_dots - 2.0)))
        .collect();
    let canvas_rows = (h_dots / 4.0) as usize;
    let slot_of = |y: f64| (((h_dots - y) * (canvas_rows.max(2) - 1) as f64) / h_dots) as usize;
    let lowest = grid.first().map(|(v, _)| *v);
    let mut labels: Vec<(f64, String)> = Vec::new();
    let mut used_slot: Option<usize> = None;
    for (v, ry) in &grid {
        let y = ry + 2.0;
        let slot = slot_of(y);
        if used_slot == Some(slot) {
            continue;
        }
        used_slot = Some(slot);
        let n = if *v >= 1.0e6 {
            format!("{:.0}M", v / 1.0e6)
        } else if *v >= 1.0e3 {
            format!("{:.0}k", v / 1.0e3)
        } else if *v < 0.1 {
            format!("{v:.2}")
        } else if *v < 1.0 {
            format!("{v:.1}")
        } else {
            format!("{v:.0}")
        };
        let text = if Some(*v) == lowest {
            format!("{n}{unit}")
        } else {
            n
        };
        labels.push((y, text));
    }
    let label_style = ref_lines.first().map(|(_, c)| *c);
    let canvas = Canvas::default()
        .marker(Marker::Braille)
        .x_bounds([0.0, w_dots])
        .y_bounds([0.0, h_dots])
        .paint(move |ctx| {
            for (_, ry) in &grid {
                let mut x = 0.0;
                while x < w_dots {
                    ctx.draw(&CLine {
                        x1: x,
                        y1: *ry,
                        x2: (x + 2.0).min(w_dots - 1.0),
                        y2: *ry,
                        color: label_style.unwrap_or(color),
                    });
                    x += 5.0;
                }
            }
            for (y, text) in &labels {
                ctx.print(
                    2.0,
                    *y,
                    Span::styled(text.clone(), Style::new().fg(label_style.unwrap_or(color))),
                );
            }
            for (cx, h) in &cols {
                ctx.draw(&CLine {
                    x1: *cx,
                    y1: 0.0,
                    x2: *cx,
                    y2: *h,
                    color,
                });
            }
            for w in overlay_tops.windows(2) {
                let (i0, x0, y0) = w[0];
                let (i1, x1, y1) = w[1];
                if i1 != i0 + 1 {
                    continue;
                }
                ctx.draw(&CLine {
                    x1: x0,
                    y1: y0,
                    x2: x1,
                    y2: y1,
                    color: overlay_color,
                });
            }
            for (cx, c) in &tick_dots {
                ctx.draw(&CLine {
                    x1: *cx,
                    y1: 0.0,
                    x2: *cx,
                    y2: (h_dots * 0.08).max(2.0),
                    color: *c,
                });
            }
        });
    f.render_widget(canvas, area);
}

// ---- latency panel -----------------------------------------------------------

fn draw_latency(f: &mut Frame, ui: &Ui, t: &Theme, area: Rect, d: Option<&Derived>, compact: bool) {
    let block = block(
        t,
        "Latency — how long users wait to see words",
        Some(&lat_subtitle(d)),
    );
    let inner = block.inner(area);
    f.render_widget(block, area);

    if compact {
        let mut lines: Vec<Line> = Vec::new();
        if let Some(d) = d {
            let w = ui.window_focus;
            for (name, lat) in [
                ("time to first token (TTFT)", &d.ttft),
                ("time between tokens (ITL)", &d.itl),
            ] {
                lines.push(lat_row_compact(t, name, &lat[w]));
            }
        }
        f.render_widget(Paragraph::new(lines), inner);
        return;
    }

    // left half: the numbers table (TTFT, ITL, queue time). The workload
    // dependent e2e row and the multi-window p95 columns are gone. Right
    // half holds the TTFT and cache-miss plots, widened by reclaimed columns.
    let halves =
        Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).split(inner);
    let mut lines: Vec<Line> = Vec::new();
    if let Some(d) = d {
        let w = ui.window_focus;
        lines.push(lat_header(t));
        for (name, lat) in [
            ("TTFT", &d.ttft),
            ("ITL", &d.itl),
            ("queue time", &d.queue_time),
        ] {
            lines.push(lat_row(t, name, &lat[w]));
        }
    }
    f.render_widget(Paragraph::new(lines), halves[0]);
    if let Some(d) = d {
        let plots =
            Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(halves[1]);
        draw_ttft_plot(f, t, plots[0], d);
        draw_cache_miss_plot(f, t, plots[1], d);
    }
}

/// Subtitle: the percentile legend, plus the prompt + computed pair gloss
/// (the prompt-length and uncached prompt-length p50 to p95 pair) when known.
fn lat_subtitle(d: Option<&Derived>) -> String {
    let mut s = String::from("p50 typical · p95 1-in-20 · p99 worst");
    if let Some(d) = d {
        if let Some(gloss) = prompt_computed_gloss(d) {
            s.push_str(&format!(" · {gloss}"));
        }
    }
    s
}

/// Prompt-size distributions behind the token rates, as p50 to p95
/// spans: the prompts as submitted, and the prefix-cache misses the device
/// chewed. Each half is None while its distribution has no data; a half-known
/// span degrades to its single p50.
fn prompt_size_span(d: &crate::derive::Derived) -> Option<String> {
    size_span(d.prompt_len_p50, d.prompt_len_p95).map(|s| format!("prompt {s} tok"))
}

fn computed_size_span(d: &crate::derive::Derived) -> Option<String> {
    size_span(d.computed_p50, d.computed_p95).map(|s| format!("computed {s} tok"))
}

fn size_span(p50: Option<f64>, p95: Option<f64>) -> Option<String> {
    match (p50, p95) {
        (Some(a), Some(b)) => Some(format!("{}–{}", fmt_num(Some(a)), fmt_num(Some(b)))),
        (Some(a), None) => Some(fmt_num(Some(a))),
        _ => None,
    }
}

/// Pair gloss in the unit-once form,
/// "prompt 27.5k–178.0k · computed 2.1k–9.0k tok". None while neither
/// distribution has data.
fn prompt_computed_gloss(d: &crate::derive::Derived) -> Option<String> {
    let prompt = size_span(d.prompt_len_p50, d.prompt_len_p95);
    let computed = size_span(d.computed_p50, d.computed_p95);
    match (prompt, computed) {
        (Some(p), Some(c)) => Some(format!("prompt {p} · computed {c} tok")),
        (Some(p), None) => Some(format!("prompt {p} tok")),
        (None, Some(c)) => Some(format!("computed {c} tok")),
        (None, None) => None,
    }
}

fn triple(v: &[Option<f64>; 3]) -> String {
    v.iter()
        .map(|x| x.map(|x| format!("{x:.0}")).unwrap_or_else(|| "—".into()))
        .collect::<Vec<_>>()
        .join("/")
}

/// Header for the half-width latency table: focused-window percentiles.
fn lat_header(t: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled(" ".repeat(11), Style::new().fg(t.dim)),
        Span::styled(
            format!("{:>7}  {:>7}  {:>7}", "p50", "p95", "p99"),
            Style::new().fg(t.dim).add_modifier(Modifier::BOLD),
        ),
    ])
}

fn lat_row(t: &Theme, name: &str, q: &crate::derive::Quantiles) -> Line<'static> {
    let fmt = |v: Option<f64>| fmt_secs(v);
    Line::from(vec![
        Span::styled(format!(" {name:<10}"), Style::new().fg(t.fg)),
        Span::styled(
            format!("{:>7}  ", fmt(q.p50)),
            Style::new().fg(latency_color(t, q.p50, q)),
        ),
        Span::styled(
            format!("{:>7}  ", fmt(q.p95)),
            Style::new().fg(latency_color(t, q.p95, q)),
        ),
        Span::styled(
            format!("{:>7}", fmt(q.p99)),
            Style::new().fg(latency_color(t, q.p99, q)),
        ),
    ])
}

fn lat_row_compact(t: &Theme, name: &str, q: &crate::derive::Quantiles) -> Line<'static> {
    let fmt = |v: Option<f64>| fmt_secs(v);
    Line::from(vec![
        Span::styled(format!(" {name:<26}"), Style::new().fg(t.fg)),
        Span::styled(
            format!("p50 {:>7}  ", fmt(q.p50)),
            Style::new().fg(latency_color(t, q.p50, q)),
        ),
        Span::styled(
            format!("p95 {:>7}  ", fmt(q.p95)),
            Style::new().fg(latency_color(t, q.p95, q)),
        ),
        Span::styled(
            format!("p99 {:>7}", fmt(q.p99)),
            Style::new().fg(latency_color(t, q.p99, q)),
        ),
    ])
}

// ---- detail ------------------------------------------------------------------

/// Generation: speculative decoding feel plus how far the running
/// requests have generated; the cache family moved to the cache panel.
/// Generation: speculative decoding feel. vLLM exposes no decode
/// sequence-length sum, so the per-request progress rows are gone;
/// the spec-decode counters stand in for "how fast it feels".
fn quality_col_kvs(d: Option<&Derived>) -> Vec<Kv> {
    let mut q: Vec<Kv> = Vec::new();
    if let Some(d) = d {
        q.push(Kv::plain(
            "spec accept",
            fmt_pct(d.spec_accept),
            "how often it guesses its own next words right — high = feels faster",
        ));
        if let Some(l) = d.spec_accept_len {
            q.push(Kv::plain(
                "spec length",
                format!("{l:.2}"),
                "words gained per guess, on average",
            ));
        }
    }
    q
}

/// Cache panel: prefix-cache hit fraction with the cached and computed
/// throughput underneath, evictions, and the L2 tier traffic, all in one place.
/// Cache panel: prefix-cache hit fraction with the cached and computed
/// throughput underneath, request preemptions, and the CPU-KV-offload
/// traffic, all in one place. vLLM reports the offload tier in bytes,
/// so the wb\u{b7}rb line reads bytes/s, not tokens/s.
fn cache_col_kvs(d: Option<&Derived>, t: &Theme) -> Vec<Kv> {
    let mut c: Vec<Kv> = Vec::new();
    if let Some(d) = d {
        let hit_color = match d.cache_hit {
            Some(v) if v < 0.5 => t.bad,
            Some(v) if v < 0.8 => t.warn,
            Some(_) => t.good,
            None => t.dim,
        };
        c.push(Kv::colored(
            "cache hit",
            fmt_pct(d.cache_hit),
            "share of each new prompt the prefix cache already holds — high = fast starts",
            hit_color,
        ));
        c.push(Kv::plain(
            "cached·computed",
            format!(
                "{} · {}",
                fmt_num(d.cache_cached),
                fmt_num(d.cache_computed)
            ),
            "tok/s served from the cache · cache misses the GPU computed",
        ));
        c.push(Kv::colored(
            "preempt/s",
            fmt_num(d.evict_rate),
            "requests preempted to free KV space",
            trouble_color(t, d.evict_rate),
        ));
        c.push(Kv::plain(
            "offload st·ld",
            format!("{} · {}", fmt_num(d.l2_wb), fmt_num(d.l2_rb)),
            "bytes/s stored to / loaded from the CPU KV cache",
        ));
        c.push(Kv::colored(
            "alloc fail/s",
            fmt_num(d.l2_drop),
            "CPU KV offload allocations that failed",
            trouble_color(t, d.l2_drop),
        ));
    }
    c
}

/// Health & trouble: the stalls row always shows. Binary alarm rows
/// (retraction, 503, l2 drop) render full only after firing once in their
/// session, collapsing into one dim line while quiet.
/// Health & trouble: the stalls row always shows. Binary alarm rows
/// (abort, 5xx, alloc fail) render full only after firing once in their
/// session, collapsing into one dim line while quiet.
fn health_col_kvs(
    d: Option<&Derived>,
    focus: usize,
    t: &Theme,
    alarms: &crate::derive::Alarms,
) -> Vec<Kv> {
    let mut h: Vec<Kv> = Vec::new();
    if let Some(d) = d {
        let s = d.stalls[focus];
        let (sc, scol) = if s.count > 0 {
            (
                format!(
                    "{} × {:.1}s in {}",
                    s.count,
                    s.seconds / s.count.max(1) as f64,
                    WIN_LABELS[focus]
                ),
                t.bad,
            )
        } else {
            ("0".into(), t.good)
        };
        h.push(Kv::colored(
            "stalls",
            sc,
            "moments when prefill work froze every stream",
            scol,
        ));
        let mut quiet: Vec<&str> = Vec::new();
        if alarms.abort {
            h.push(Kv::colored(
                "abort/s",
                fmt_num(d.retract_rate),
                "requests stopped mid-answer by the client",
                trouble_color(t, d.retract_rate),
            ));
        } else {
            quiet.push("abort");
        }
        if alarms.http_5xx {
            h.push(Kv::colored(
                "5xx/s",
                fmt_num(d.http_503_rate),
                "requests refused with a server error",
                trouble_color(t, d.http_503_rate),
            ));
        } else {
            quiet.push("5xx");
        }
        if alarms.alloc_fail {
            h.push(Kv::colored(
                "alloc fail/s",
                fmt_num(d.l2_drop),
                "CPU KV offload allocations that failed",
                trouble_color(t, d.l2_drop),
            ));
        } else {
            quiet.push("alloc fail");
        }
        if !quiet.is_empty() {
            // compact channel names: the line shares the panel with the full
            // rows it replaces
            h.push(Kv::gloss_only(format!("quiet: {}", quiet.join("·"))));
        }
        h.push(Kv::plain(
            "sched cpu/s",
            format!("{:.1}", d.cpu_scheduler.unwrap_or(0.0)),
            "",
        ));
        h.push(Kv::gloss_only("scheduler compute".to_string()));
    }
    h
}

fn peaks_col_lines(peaks: &crate::derive::Peaks, t: &Theme) -> Vec<Line<'static>> {
    // the single row carries the single-stream s3 color, matching
    // the decode canvas's own single-stream line color
    let pk = |label: &str, v: Option<f64>, color: ratatui::style::Color| {
        Line::from(vec![
            Span::styled(format!(" {label:<9}"), Style::new().fg(t.dim)),
            Span::styled(
                v.map(|v| format!("{v:.0} tok/s"))
                    .unwrap_or_else(|| "\u{2014}".into()),
                Style::new().fg(color).add_modifier(Modifier::BOLD),
            ),
        ])
    };
    let l = vec![
        pk("decode", peaks.decode, t.fg),
        pk("prefill", peaks.prefill, t.fg),
        pk("single", peaks.decode_single, t.s3),
    ];
    l
}

/// Rows the detail row needs (content plus borders), bounded to [8, 14].
/// The lower bound leaves the super-large rate graphs their height share.
/// Sized by the live session state: a latched alarm adds its full
/// row to the health column, so the quiet-state shape would clip it.
fn detail_height(
    d: Option<&Derived>,
    focus: usize,
    alarms: &crate::derive::Alarms,
    peaks: &crate::derive::Peaks,
) -> u16 {
    let t = &THEMES[0];
    let n = [
        quality_col_kvs(d).len(),
        cache_col_kvs(d, t).len(),
        health_col_kvs(d, focus, t, alarms).len(),
        peaks_col_lines(peaks, t).len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    (n as u16 + 2).clamp(8, 14)
}

fn draw_detail(
    f: &mut Frame,
    ui: &Ui,
    t: &Theme,
    area: Rect,
    d: Option<&Derived>,
    compact: bool,
    session: &crate::scrape::Shared,
) {
    // the two session-sticky structures are read here, after the history
    // guard has been dropped upstream
    let peaks = session.peaks.lock().unwrap();
    let alarms = session.alarms.lock().unwrap();
    let focus = ui.window_focus;
    if compact {
        let mut text = String::new();
        if let Some(d) = d {
            text.push_str(&format!(
                "cache hit {} · cached·computed {}·{} · spec accept {}",
                fmt_pct(d.cache_hit),
                fmt_num(d.cache_cached),
                fmt_num(d.cache_computed),
                fmt_pct(d.spec_accept)
            ));
            text.push('\n');
            let s = d.stalls[focus];
            text.push_str(&format!(
                "stalls {} × {:.1}s in {} · preempt/s {} · abort/s {} · 5xx/s {}",
                s.count,
                if s.count > 0 {
                    s.seconds / s.count.max(1) as f64
                } else {
                    0.0
                },
                WIN_LABELS[focus],
                fmt_num(d.evict_rate),
                fmt_num(d.retract_rate),
                fmt_num(d.http_503_rate)
            ));
        }
        let b = block(t, "Queues & health", None);
        f.render_widget(
            Paragraph::new(text).style(Style::new().fg(t.fg)),
            b.inner(area),
        );
        f.render_widget(b, area);
        return;
    }

    let cols = Layout::horizontal([
        Constraint::Percentage(24),
        Constraint::Percentage(24),
        Constraint::Percentage(28),
        Constraint::Percentage(24),
    ])
    .split(area);

    let b1 = block(t, "Generation", Some("in-flight request stats"));
    let q = quality_col_kvs(d);
    let inner1 = b1.inner(cols[0]);
    f.render_widget(
        Paragraph::new(kv_lines(t, q, inner1.width as usize)),
        inner1,
    );
    f.render_widget(b1, cols[0]);

    let b2 = block(
        t,
        "Cache",
        Some("prefix-cache hits, traffic and the L2 tiers"),
    );
    let c = cache_col_kvs(d, t);
    let inner2 = b2.inner(cols[1]);
    f.render_widget(
        Paragraph::new(kv_lines(t, c, inner2.width as usize)),
        inner2,
    );
    f.render_widget(b2, cols[1]);

    let b3 = block(
        t,
        "Health & trouble",
        Some("anything here that is not zero deserves a look"),
    );
    let h = health_col_kvs(d, focus, t, &alarms);
    let inner3 = b3.inner(cols[2]);
    f.render_widget(
        Paragraph::new(kv_lines(t, h, inner3.width as usize)),
        inner3,
    );
    f.render_widget(b3, cols[2]);

    let b4 = block(t, "Peaks", Some("since vllm-top started"));
    let inner4 = b4.inner(cols[3]);
    let l4 = peaks_col_lines(&peaks, t)
        .into_iter()
        .map(|l| truncate_line(l, inner4.width as usize))
        .collect::<Vec<Line>>();
    f.render_widget(Paragraph::new(l4), inner4);
    f.render_widget(b4, cols[3]);
}

/// Truncate a styled line to `width` chars, appending `…` when cut.
fn truncate_line(line: Line<'static>, width: usize) -> Line<'static> {
    if width == 0 {
        return Line::default();
    }
    let mut spans: Vec<Span> = Vec::new();
    let mut remaining = width;
    for span in line.spans {
        if remaining == 0 {
            break;
        }
        let len = span.content.chars().count();
        if len <= remaining {
            remaining -= len;
            spans.push(span);
        } else {
            let cut: String = span
                .content
                .chars()
                .take(remaining.saturating_sub(1))
                .collect();
            spans.push(Span::styled(format!("{cut}\u{2026}"), span.style));
            remaining = 0;
        }
    }
    Line::from(spans)
}

/// One aligned key-value row spec for [`kv_lines`].
struct Kv {
    k: String,
    v: String,
    gloss: String,
    style: Option<Style>,
}

impl Kv {
    fn plain(k: &str, v: String, gloss: &str) -> Self {
        Self {
            k: k.into(),
            v,
            gloss: gloss.into(),
            style: None,
        }
    }
    fn colored(k: &str, v: String, gloss: &str, style: ratatui::style::Color) -> Self {
        Self {
            k: k.into(),
            v,
            gloss: gloss.into(),
            style: Some(Style::new().fg(style)),
        }
    }
    /// A dim full-width line (used for the CPU column legend).
    fn gloss_only(gloss: String) -> Self {
        Self {
            k: String::new(),
            v: String::new(),
            gloss,
            style: None,
        }
    }
}

/// Render kv entries with the value column sized to the widest value, so
/// narrow panels don't waste columns on padding.
fn kv_lines(t: &Theme, entries: Vec<Kv>, width: usize) -> Vec<Line<'static>> {
    let kpad = entries
        .iter()
        .map(|e| e.k.chars().count())
        .max()
        .unwrap_or(0)
        .max(8);
    let vpad = entries
        .iter()
        .map(|e| e.v.chars().count())
        .max()
        .unwrap_or(1);
    entries
        .into_iter()
        .map(|e| {
            if e.k.is_empty() && e.v.is_empty() {
                return truncate_line(
                    Line::from(Span::styled(
                        format!(" {}", e.gloss),
                        Style::new().fg(t.dim),
                    )),
                    width,
                );
            }
            let value_style = e
                .style
                .unwrap_or(Style::new().fg(t.fg))
                .add_modifier(Modifier::BOLD);
            truncate_line(
                Line::from(vec![
                    Span::styled(format!(" {:<kpad$}  ", e.k), Style::new().fg(t.fg)),
                    Span::styled(format!("{:<vpad$}  ", e.v), value_style),
                    Span::styled(e.gloss, Style::new().fg(t.dim)),
                ]),
                width,
            )
        })
        .collect()
}

fn trouble_color(t: &Theme, v: Option<f64>) -> ratatui::style::Color {
    if v.unwrap_or(0.0) > 0.0 {
        t.bad
    } else {
        t.good
    }
}

// ---- help & explain overlays ---------------------------------------------

fn draw_help(f: &mut Frame, t: &Theme, area: Rect) {
    let w = 52.min(area.width);
    let h = 16.min(area.height);
    let x = area.x + (area.width - w) / 2;
    let y = area.y + (area.height - h) / 2;
    let area = Rect {
        x,
        y,
        width: w,
        height: h,
    };

    let rows = [
        ("q / Esc", "quit"),
        ("space", "pause scraping (for staring & screenshots)"),
        ("c", "toggle compact / full layout"),
        ("g", "toggle graphs"),
        ("1 2 3", "focus the 5s / 15s / 60s window"),
        ("t", "cycle theme"),
        ("e", "plain-language tour of every panel"),
        ("+ / -", "poll interval (faster / slower)"),
        ("↑ ↓", "scroll the explain screen"),
        ("?", "this keymap"),
    ];
    let mut lines: Vec<Line> = Vec::new();
    for (k, v) in rows {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:<9}", k),
                Style::new().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(v.to_string(), Style::new().fg(t.fg)),
        ]));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.accent))
        .style(Style::new().bg(t.bg))
        .title(Span::styled(
            " keymap ",
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_explain(f: &mut Frame, ui: &Ui, t: &Theme, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.accent))
        .style(Style::new().bg(t.bg))
        .title(Span::styled(
            " What am I looking at?  (e or q to close · ↑↓ scroll) ",
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(ratatui::widgets::Clear, area);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    let section = |title: &str, body: &str, lines: &mut Vec<Line>| {
        lines.push(Line::from(Span::styled(
            format!(" {title}"),
            Style::new().fg(t.accent).add_modifier(Modifier::BOLD),
        )));
        for para in body.split("\n\n") {
            for seg in wrap_text(para, (inner.width as usize).saturating_sub(4).max(20)) {
                lines.push(Line::from(Span::styled(
                    format!(" {seg}"),
                    Style::new().fg(t.fg),
                )));
            }
            lines.push(Line::default());
        }
    };
    section("A token", "A token is roughly a word-piece: the model reads your prompt and writes its answer one token at a time, several times per second. \"tok/s\" below is how many of those pieces appear per second.", &mut lines);
    section("Token rates (TOKEN/s)", "Decode is the model writing its answer — the number users experience as speed. Prefill is the model reading a new prompt; it happens in bursts and can momentarily freeze everyone else's stream (the seesaw in the graph). The prompt + computed gloss names both size distributions behind those rates: the prompts as submitted, and the prefix-cache misses the GPU computed.", &mut lines);
    section("Time to first token (TTFT)", "When you send a message, this is the pause before the first word appears. Small is good. It grows when the model is busy reading long prompts or when there is a line of requests waiting.", &mut lines);
    section("Time between tokens (ITL)", "Once words are flowing, ITL is the rhythm: the time between one word and the next. Steady and small means the text streams smoothly; spikes are why text sometimes freezes then jumps — usually a new request's prompt being processed in the middle of your answer.\n\nThe decode graph draws a second line for the single-stream speed: total writing speed divided by how many answers are being written at once. When that line dips while a prefill burst rides the same graph, everyone's stream briefly froze.", &mut lines);
    section("p50 / p95 / p99", "p50 is a typical request. p95 is the experience of the unluckiest 1-in-20. p99 is the worst moments. If p50 is fine but p95 is bad, most people are happy but some are having a bad time — that gap is the number to watch.\n\nThe latency row plots first-token wait (TTFT, its p95 shaded) next to the cache-miss rate: computed prompt tokens per second, the prompt text the prefix cache did not hold. Queue time isolates congestion — how long requests waited for their turn — while end-to-end latency also grows with the work itself, so it stays out of the plots (once-mode still prints it).", &mut lines);
    section("Waiting (QUEUE)", "QUEUE counts requests that arrived but have not started. A short, spiky line is normal; a tall, flat line means the server is overloaded and everyone's wait grows. The latency table's queue row shows how long they waited.", &mut lines);
    section("Peaks", "The Peak box holds the highest rates seen since vllm-top started: the busiest decode moment, the biggest prefill burst, and single — the fastest one stream has ever moved (total decode speed divided by how many requests were sharing it).", &mut lines);
    section("Memory pools", "The model keeps working memory for every conversation in progress (KV). At 100% a pool is full: new requests wait, and the server may throw out or preempt the ones running (see preemptions).\n\nThe host tier is the CPU-RAM cache the engine uses to offload KV blocks. It is reported as a percentage, and a high reading is normal — it is a write-through cache that recycles its own oldest entries. Only the GPU KV pool running full is the requests-will-wait warning.", &mut lines);
    section("Stalls", "A stall is a moment when every stream froze because the engine was busy with something else instead of decoding: a prefill was computing (decode starved), or requests were queued waiting to start (decode enqueued/blocked). Measured as how many times it happened and how many seconds in total over the window. Red ticks mark stalled intervals on the decode and latency plots; orange ticks there mark intervals where the eviction counter advanced (device KV cache freed to make room).", &mut lines);
    section("Cache hit & the KV offload", "How much of each new prompt the server already remembers from earlier turns. The cache-misses plot tracks the opposite side: computed prompt tokens/s, the text the GPU had to read because no cache tier held it. High cache hit = fast starts and less work. A sudden drop usually means a restart or very different traffic.\n\nThe offload st\u{b7}ld line is the CPU-KV-offload traffic in bytes/s — how much KV state is being written down to, or read back from, the CPU cache. alloc fail/s counts offload allocations that failed; it should be zero.", &mut lines);
    section("Speculative accept", "The model tries to guess several words ahead and checks them in one go. Accept rate is how often the guesses are right; accept length is how many words each lucky guess saves. High numbers make everything feel faster.", &mut lines);
    section("Preemptions & aborts", "A preemption means a running request was stopped and its KV blocks freed to make room — the user's answer stalls and may need to be restarted. Aborts are requests the client cancelled mid-answer. Preemptions are the signal of memory pressure; a steady preemption count is normal, a spike is the engine running out of KV space.", &mut lines);

    f.render_widget(Paragraph::new(lines).scroll((ui.scroll, 0)), inner);
}

fn wrap_text(s: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in s.split('\n') {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if !line.is_empty() && line.len() + word.len() + 1 > width {
                out.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        out.push(line);
    }
    out
}

// ---- shared small helpers -------------------------------------------------

fn block<'a>(t: &Theme, title: &str, subtitle: Option<&str>) -> Block<'a> {
    block_titled(
        t,
        Line::from(Span::styled(
            format!(" {title} "),
            Style::new().fg(t.fg).add_modifier(Modifier::BOLD),
        )),
        subtitle.map(|s| Line::from(Span::styled(format!(" {s} "), Style::new().fg(t.dim)))),
    )
}

fn block_titled<'a>(t: &Theme, title: Line<'static>, subtitle: Option<Line<'static>>) -> Block<'a> {
    let mut b = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(t.dim))
        .style(Style::new().bg(t.bg))
        .title(title);
    if let Some(sub) = subtitle {
        b = b.title_bottom(sub);
    }
    b
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m{}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

fn fmt_num(v: Option<f64>) -> String {
    v.map(|v| {
        if v.abs() >= 1000.0 {
            format!("{:.1}k", v / 1000.0)
        } else {
            format!("{v:.0}")
        }
    })
    .unwrap_or_else(|| "—".into())
}

fn fmt_pct(v: Option<f64>) -> String {
    v.map(|v| format!("{:.0}%", v * 100.0))
        .unwrap_or_else(|| "—".into())
}

/// 115904 -> "116k", 2843392 -> "2.8M"
pub fn fmt_tokens(v: f64) -> String {
    if v >= 1.0e6 {
        format!("{:.1}M", v / 1.0e6)
    } else if v >= 1.0e3 {
        format!("{:.0}k", v / 1.0e3)
    } else {
        format!("{v:.0}")
    }
}

/// Absolute pool occupancy for the pools text lines: "8k/10k tok" for token pools.
/// Percentages stay out of the count lines where a count's denominator
/// supports no action. vLLM's host (CPU KV offload) pool reports no absolute
/// count, so it falls back to its usage percentage — the only reading it has.
/// None only when a pool has neither counts nor a usable usage ratio.
pub(crate) fn pool_value(p: &crate::derive::Pool) -> Option<String> {
    match (p.used, p.total) {
        (Some(u), Some(t)) if t > 0.0 => {
            let unit = if p.unit == "slots" { "slots" } else { "tok" };
            Some(format!("{}/{} {}", fmt_tokens(u), fmt_tokens(t), unit))
        }
        _ => {
            // no absolute counts (vLLM host offload): fall back to the usage %
            Some(format!("{:.0}%", p.usage * 100.0))
        }
    }
}

fn fmt_secs(v: Option<f64>) -> String {
    match v {
        Some(v) if v >= 100.0 => format!("{v:.0}s"),
        Some(v) if v >= 10.0 => format!("{v:.1}s"),
        Some(v) => format!("{:.0}ms", v * 1000.0),
        None => "—".into(),
    }
}
