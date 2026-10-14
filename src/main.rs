use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;

use vllm_top::{args::Args, metrics, scrape, ui};

/// Rewrites a scrape or parse failure for the anyhow return: parser bails
/// embed label values verbatim, so the printed message must carry no raw
/// terminal control bytes (the same rule as the TUI's error banner).
fn safe_err(e: anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("{}", scrape::sanitize_for_terminal(&format!("{e:#}")))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let url = scrape::metrics_url(&args.url);

    if args.once {
        // the disclosure precedes the scrapes: a first-scrape failure
        // must not leave the operator running unverified, unwarned
        if args.insecure {
            eprintln!("\u{26a0} insecure TLS: the bearer token and metrics transit unverified");
        }
        let mut h = vllm_top::history::History::default();
        // two scrapes so windows have a base for rates/quantiles
        for (i, wait) in [0.0, args.interval.clamp(0.5, 10.0)]
            .into_iter()
            .enumerate()
        {
            if i > 0 {
                std::thread::sleep(Duration::from_secs_f64(wait));
            }
            let body = scrape::fetch_once(&url, args.api_key.as_deref(), args.insecure)
                .map_err(safe_err)?;
            let sample = metrics::parse(&body).map_err(safe_err)?;
            h.push(Instant::now(), sample);
        }
        let mut peaks = vllm_top::derive::Peaks::default();
        vllm_top::once::print_once(&h, &mut peaks)?;
        return Ok(());
    }

    let shared = scrape::Shared::new();
    shared.interval_ms.store(
        (args.interval.clamp(0.5, 10.0) * 1000.0) as u64,
        Ordering::Relaxed,
    );
    scrape::spawn_scraper(shared.clone(), url.clone(), args.api_key.clone(), args.insecure);

    if args.web {
        let exe = std::env::current_exe()?.canonicalize()?;
        let mut cmd = std::process::Command::new("ttyd");
        cmd.arg("-p").arg(args.web_port.to_string());
        cmd.arg("--writable");
        
        let mut full_args = vec![exe.to_string_lossy().into_owned()];
        let mut skip_next = false;
        for arg in std::env::args().skip(1) {
            if skip_next {
                skip_next = false;
                continue;
            }
            if arg == "--web" {
                continue;
            } else if arg == "--web-port" {
                skip_next = true;
                continue;
            }
            full_args.push(arg);
        }

        for arg in full_args {
            cmd.arg(arg);
        }

        println!("Launching TUI in browser at http://localhost:{}", args.web_port);
        let mut child = cmd.spawn()?;
        let _status = child.wait()?;
        return Ok(());
    }


    let mut terminal = ratatui::init();
    let mut ui_state = ui::Ui::new(&args);
    let result = ui::run(&mut terminal, shared, &mut ui_state);
    ratatui::restore();
    result
}
