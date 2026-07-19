use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "neural-forge",
    about = "universal local AI runtime",
    version,
    disable_version_flag = false,
    args_conflicts_with_subcommands = false
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Doctor {
        #[arg(long)]
        json: bool,
    },

    Plan {
        model: String,

        #[arg(long, default_value = "4096")]
        context: usize,

        #[arg(long, default_value = "quality")]
        policy: String,

        #[arg(long)]
        json: bool,
    },

    Run {
        model: String,
        prompt: Vec<String>,
        #[arg(long)]
        tokenizer: String,
        #[arg(long, default_value = "512")]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long)]
        seed: Option<u64>,
    },

    Chat {
        model: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long, default_value = "1024")]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
    },

    Serve {
        model: String,
        #[arg(long)]
        tokenizer: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long)]
        api_key: Option<String>,
    },

    Pull {
        model: String,
        #[arg(long)]
        cache_dir: Option<String>,
    },

    List {
        #[arg(long)]
        json: bool,
    },

    Inspect {
        path: String,
        #[arg(long)]
        json: bool,
    },

    Tokenize {
        model: String,
        text: Vec<String>,
        #[arg(long)]
        json: bool,
    },

    Build,
}

