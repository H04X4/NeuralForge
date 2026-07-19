use colored::*;

pub struct Style;

impl Style {
    pub fn accent() -> ColoredString {
        "◆".truecolor(120, 200, 220).bold()
    }

    pub fn ok() -> ColoredString {
        "✓".truecolor(80, 220, 120).bold()
    }

    pub fn warn() -> ColoredString {
        "△".truecolor(240, 180, 60).bold()
    }

    pub fn err() -> ColoredString {
        "✗".truecolor(240, 80, 80).bold()
    }

    pub fn dim() -> ColoredString {
        "·".truecolor(80, 90, 100)
    }

    pub fn rule() -> String {
        "─".repeat(48).truecolor(50, 60, 70).to_string()
    }

    pub fn label(s: &str) -> ColoredString {
        s.truecolor(120, 140, 160)
    }

    pub fn value(s: &str) -> ColoredString {
        s.truecolor(210, 220, 235)
    }

    pub fn heading(s: &str) -> String {
        format!(
            "{}  {}\n{}",
            Style::accent(),
            s.truecolor(190, 210, 230).bold(),
            Style::rule()
        )
    }

    pub fn sub(s: &str) -> ColoredString {
        s.truecolor(120, 140, 160).italic()
    }

    pub fn dim_label(s: &str) -> ColoredString {
        s.truecolor(90, 100, 115)
    }

    pub fn prompt_label() -> String {
        format!("{}", "you".truecolor(120, 200, 220).bold())
    }

    pub fn model_label() -> String {
        format!("{}", "neural".truecolor(80, 220, 120).bold())
    }
}

pub fn print_json<T: serde::Serialize>(value: &T) {
    match serde_json::to_string_pretty(value) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("{} JSON serialization error: {}", Style::err(), e),
    }
}

pub fn kv(key: &str, val: impl std::fmt::Display) -> String {
    format!(
        "  {} {:22} {}",
        Style::dim(),
        Style::label(key),
        Style::value(&val.to_string())
    )
}

