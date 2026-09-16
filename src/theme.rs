use ratatui::style::Color;

#[derive(Clone, Copy)]
pub struct Theme {
    pub name: &'static str,
    pub bg: Color,
    pub fg: Color,
    pub dim: Color,
    pub accent: Color,
    pub good: Color,
    pub warn: Color,
    pub bad: Color,
    /// primary graph series (decode)
    pub s1: Color,
    /// secondary graph series (prefill)
    pub s2: Color,
    /// tertiary graph series (the single-stream overlay): must contrast
    /// strongly with s1, since the two share the decode canvas
    pub s3: Color,
}

pub const THEMES: [Theme; 3] = [GRUVBOX, CATPPUCCIN, TOKYONIGHT];

pub const GRUVBOX: Theme = Theme {
    name: "gruvbox",
    bg: Color::Rgb(0x28, 0x28, 0x28),
    fg: Color::Rgb(0xeb, 0xdb, 0xb2),
    dim: Color::Rgb(0x92, 0x83, 0x74),
    accent: Color::Rgb(0x83, 0xa5, 0x98),
    good: Color::Rgb(0xb8, 0xbb, 0x26),
    warn: Color::Rgb(0xfe, 0x80, 0x19),
    bad: Color::Rgb(0xfb, 0x49, 0x34),
    s1: Color::Rgb(0x8e, 0xc0, 0x7c),
    s2: Color::Rgb(0xd3, 0x86, 0x9b),
    s3: Color::Rgb(0xfa, 0xbd, 0x2f),
};

pub const CATPPUCCIN: Theme = Theme {
    name: "catppuccin",
    bg: Color::Rgb(0x1e, 0x1e, 0x2e),
    fg: Color::Rgb(0xcd, 0xd6, 0xf4),
    dim: Color::Rgb(0x6c, 0x70, 0x86),
    accent: Color::Rgb(0x89, 0xb4, 0xfa),
    good: Color::Rgb(0xa6, 0xe3, 0xa1),
    warn: Color::Rgb(0xfa, 0xb3, 0x87),
    bad: Color::Rgb(0xf3, 0x8b, 0xa8),
    s1: Color::Rgb(0x94, 0xe2, 0xd5),
    s2: Color::Rgb(0xf5, 0xc2, 0xe7),
    s3: Color::Rgb(0xf9, 0xe2, 0xaf),
};

pub const TOKYONIGHT: Theme = Theme {
    name: "tokyonight",
    bg: Color::Rgb(0x1a, 0x1b, 0x26),
    fg: Color::Rgb(0xc0, 0xca, 0xf5),
    dim: Color::Rgb(0x56, 0x5f, 0x89),
    accent: Color::Rgb(0x7a, 0xa2, 0xf7),
    good: Color::Rgb(0x9e, 0xce, 0x6a),
    warn: Color::Rgb(0xe0, 0xaf, 0x68),
    bad: Color::Rgb(0xf7, 0x76, 0x8e),
    s1: Color::Rgb(0x73, 0xda, 0xca),
    s2: Color::Rgb(0xbb, 0x9a, 0xf7),
    s3: Color::Rgb(0xff, 0x9e, 0x64),
};

/// Color ramp for usage-style ratios: calm when healthy, loud near full.
pub fn usage_color(t: &Theme, ratio: f64) -> Color {
    if ratio < 0.6 {
        t.s1
    } else if ratio < 0.85 {
        t.warn
    } else {
        t.bad
    }
}
