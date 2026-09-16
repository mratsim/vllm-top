use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::history::History;
use crate::metrics;

pub struct Shared {
    pub history: Mutex<History>,
    pub last_ok: Mutex<Option<Instant>>,
    pub last_error: Mutex<Option<String>>,
    pub paused: AtomicBool,
    /// scraper poll interval, live-adjustable from the TUI
    pub interval_ms: std::sync::atomic::AtomicU64,
    /// session peak rates (see derive::Peaks)
    pub peaks: Mutex<crate::derive::Peaks>,
    /// session-sticky alarm latches (see derive::Alarms)
    pub alarms: Mutex<crate::derive::Alarms>,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            history: Mutex::new(History::default()),
            last_ok: Mutex::new(None),
            last_error: Mutex::new(None),
            paused: AtomicBool::new(false),
            interval_ms: std::sync::atomic::AtomicU64::new(1000),
            peaks: Mutex::new(crate::derive::Peaks::default()),
            alarms: Mutex::new(crate::derive::Alarms::default()),
        })
    }
}

fn agent(insecure: bool, timeout: Duration) -> ureq::Agent {
    let builder = ureq::AgentBuilder::new().timeout(timeout);
    if insecure {
        let verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            Arc::new(danger::NoVerifier);
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let cfg = rustls::ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(rustls::ALL_VERSIONS)
            .expect("tls versions")
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        builder.tls_config(Arc::new(cfg)).build()
    } else {
        builder.build()
    }
}

mod danger {
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, Error, SignatureScheme};

    /// Accepts every server certificate. Only reachable via `--insecure`,
    /// which exists for homelab endpoints behind self-signed certs.
    #[derive(Debug)]
    pub struct NoVerifier;

    impl ServerCertVerifier for NoVerifier {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            verify(message, cert, dss)
        }

        fn verify_tls13_signature(
            &self,
            message: &[u8],
            cert: &CertificateDer<'_>,
            dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            verify(message, cert, dss)
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    fn verify(
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        let algs = &rustls::crypto::ring::default_provider().signature_verification_algorithms;
        rustls::crypto::verify_tls12_signature(message, cert, dss, algs)
            .or_else(|_| rustls::crypto::verify_tls13_signature(message, cert, dss, algs))
    }
}

/// Normalize a user-supplied base URL into a full /metrics URL.
pub fn metrics_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/metrics") {
        base.to_string()
    } else {
        format!("{base}/metrics")
    }
}

/// Upper bound, in visible characters, of a display-bound error message.
const MAX_ERROR_CHARS: usize = 256;

/// Rewrites an error message so it is safe to print to a terminal.
/// The message is hard-truncated to [`MAX_ERROR_CHARS`] characters with a trailing
/// ellipsis marker; control characters become visible escape notation (C0, C1 and the delete byte all covered).
///
/// Parser bails embed `SeriesKey` label values verbatim, so a hostile
/// endpoint can inject terminal control sequences through any labeled
/// error. Every message bound for a terminal goes through this first,
/// staying debuggable: family names, labels, and counts survive.
///
/// ```
/// // an injected escape sequence renders as notation, sending no bytes
/// assert_eq!(vllm_top::scrape::sanitize_for_terminal("a\u{1b}[2Jb"), "a\\x1b[2Jb");
/// ```
pub fn sanitize_for_terminal(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len().min(MAX_ERROR_CHARS));
    for c in msg.chars() {
        let piece = match c {
            '\n' => "\\n".to_owned(),
            '\r' => "\\r".to_owned(),
            '\t' => "\\t".to_owned(),
            c if c.is_control() => format!("\\x{:02x}", u32::from(c)),
            c => c.to_string(),
        };
        // one character reserved for the truncation marker
        if out.chars().count() + piece.chars().count() > MAX_ERROR_CHARS - 1 {
            out.push('\u{2026}');
            return out;
        }
        out.push_str(&piece);
    }
    out
}

pub fn fetch_once(url: &str, api_key: Option<&str>, insecure: bool) -> Result<String> {
    let agent = agent(insecure, Duration::from_secs(5));
    fetch(&agent, url, api_key)
}

pub fn spawn_scraper(shared: Arc<Shared>, url: String, api_key: Option<String>, insecure: bool) {
    let agent = agent(insecure, Duration::from_secs(5));
    std::thread::spawn(move || loop {
        if !shared.paused.load(Ordering::Relaxed) {
            match fetch(&agent, &url, api_key.as_deref()) {
                Ok(body) => match metrics::parse(&body) {
                    // a rejected payload (e.g. series cap) leaves history and peaks
                    // untouched so the UI goes stale under the error banner
                    Ok(sample) => {
                        if let Ok(mut h) = shared.history.lock() {
                            h.push(Instant::now(), sample);
                        }
                        if let Ok(mut p) = shared.peaks.lock() {
                            if let Ok(h) = shared.history.lock() {
                                crate::derive::update_peaks(&mut p, &h);
                            }
                        }
                        if let Ok(mut a) = shared.alarms.lock() {
                            if let Ok(h) = shared.history.lock() {
                                crate::derive::update_alarms(&mut a, &h);
                            }
                        }
                        if let Ok(mut t) = shared.last_ok.lock() {
                            *t = Some(Instant::now());
                        }
                        if let Ok(mut e) = shared.last_error.lock() {
                            *e = None;
                        }
                    }
                    Err(e) => {
                        // the message may embed hostile label values
                        // (series keys render verbatim in parser bails)
                        if let Ok(mut slot) = shared.last_error.lock() {
                            *slot = Some(sanitize_for_terminal(&format!("{e:#}")));
                        }
                    }
                },
                Err(e) => {
                    if let Ok(mut slot) = shared.last_error.lock() {
                        *slot = Some(sanitize_for_terminal(&format!("{e:#}")));
                    }
                }
            }
        }
        let ms = shared
            .interval_ms
            .load(Ordering::Relaxed)
            .clamp(250, 10_000);
        std::thread::sleep(Duration::from_millis(ms));
    });
}

fn fetch(agent: &ureq::Agent, url: &str, api_key: Option<&str>) -> Result<String> {
    let mut req = agent.get(url);
    if let Some(key) = api_key {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    let resp = req.call().context("scrape failed")?;
    let mut reader = resp.into_reader().take(64 * 1024 * 1024);
    let mut body = String::new();
    reader
        .read_to_string(&mut body)
        .context("reading scrape body")?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_escapes_control_characters_into_visible_notation() {
        // terminal escape sequences, the delete byte, carriage returns,
        // and an unescaped newline all become notation: the output
        // carries no raw control byte
        let safe = sanitize_for_terminal("m:h{mode=\"\u{1b}[2J\u{7f}\r\nx\"} too large");
        assert!(!safe.chars().any(char::is_control), "message was: {safe:?}");
        for notation in ["\\x1b", "\\x7f", "\\r", "\\n"] {
            assert!(safe.contains(notation), "missing {notation}: {safe:?}");
        }
    }

    #[test]
    fn sanitize_truncates_overlong_messages_with_a_marker() {
        let safe = sanitize_for_terminal(&"x".repeat(400));
        assert_eq!(safe.chars().count(), MAX_ERROR_CHARS);
        assert!(safe.ends_with('\u{2026}'), "message was: {safe:?}");
        // a hostile payload cannot smuggle controls past the cut either
        let hostile = sanitize_for_terminal(&format!("{}\u{1b}", "x".repeat(400)));
        assert!(!hostile.chars().any(char::is_control));
        assert!(hostile.ends_with('\u{2026}'));
    }

    #[test]
    fn sanitize_leaves_benign_messages_unchanged() {
        let msg = "endpoint too large: m:h{mode=\"decode\"} has 257 buckets (cap 256)";
        assert_eq!(sanitize_for_terminal(msg), msg);
    }

    #[test]
    fn url_normalization() {
        assert_eq!(
            metrics_url("http://localhost:30000"),
            "http://localhost:30000/metrics"
        );
        assert_eq!(
            metrics_url("http://localhost:30000/"),
            "http://localhost:30000/metrics"
        );
        assert_eq!(
            metrics_url("https://example.org/metrics"),
            "https://example.org/metrics"
        );
    }
}
