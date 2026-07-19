mod cli;
mod cmd;
mod term;

use clap::Parser;
use cli::{Cli, Commands};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Doctor { json } => cmd::cmd_doctor(json),
        Commands::Plan {
            model,
            context,
            policy,
            json,
        } => cmd::cmd_plan(&model, context, &policy, json),
        Commands::Run {
            model,
            prompt,
            tokenizer,
            max_tokens,
            temperature,
            seed,
        } => cmd::cmd_run(&model, &prompt, &tokenizer, max_tokens, temperature, seed),
        Commands::Chat {
            model,
            tokenizer,
            max_tokens,
            temperature,
        } => cmd::cmd_chat(&model, &tokenizer, max_tokens, temperature),
        Commands::Serve {
            model,
            tokenizer,
            host,
            port,
            api_key,
        } => cmd::cmd_serve(&model, &tokenizer, &host, port, api_key.as_deref()),
        Commands::Pull { model, cache_dir } => cmd::cmd_pull(&model, cache_dir.as_deref()),
        Commands::List { json } => cmd::cmd_list(json),
        Commands::Inspect { path, json } => cmd::cmd_inspect(&path, json),
        Commands::Tokenize { model, text, json } => cmd::cmd_tokenize(&model, &text, json),
        Commands::Build => cmd::cmd_build(),
    }
}

