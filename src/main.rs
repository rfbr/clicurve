mod app;
mod proto;
mod store;
mod tfrecord;

use anyhow::{bail, Context, Result};
use app::App;
use clap::Parser;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::path::PathBuf;
use store::MultiStore;

#[derive(Parser)]
#[command(name = "clicurve", about = "Terminal TensorBoard viewer")]
struct Cli {
    /// Path to experiment directory (auto-detects sub-experiments with tensorboard/ folders)
    path: PathBuf,

    /// Initial filter string for metric names
    #[arg(short, long)]
    filter: Option<String>,

    /// Start with log scale enabled
    #[arg(short, long)]
    log_scale: bool,

    /// List available experiments and metrics, then exit (no TUI)
    #[arg(long)]
    list: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let dir = cli.path.canonicalize().with_context(|| {
        format!("cannot resolve path: {}", cli.path.display())
    })?;

    eprintln!("Loading events from {}...", dir.display());
    let store = MultiStore::load(&dir)?;

    if store.experiments.is_empty() {
        bail!(
            "No experiments found in {}.\n\
             Expected subdirectories with tensorboard/ folders, \
             or event files directly.",
            dir.display()
        );
    }

    let n_exp = store.experiments.len();
    let n_metrics = store.total_metrics();
    let n_files = store.total_event_files();
    eprintln!("Found {n_exp} experiment(s), {n_metrics} metrics, {n_files} event files.");

    if cli.list {
        for (name, _, exp_store) in &store.experiments {
            println!("\n=== {name} ({} metrics) ===", exp_store.metrics.len());
            for (tag, points) in &exp_store.metrics {
                let last = points.last().map(|p| p.value).unwrap_or(0.0);
                let min = points.iter().map(|p| p.value).fold(f32::INFINITY, f32::min);
                let max = points.iter().map(|p| p.value).fold(f32::NEG_INFINITY, f32::max);
                let steps = points.len();
                println!("  {tag:40} {steps:>6} pts  last={last:.4}  min={min:.4}  max={max:.4}");
            }
        }
        return Ok(());
    }

    let mut app = App::new(store, dir);
    if let Some(f) = cli.filter {
        app.filter = f;
        app.filter_mode = false;
        app.reload()?;
    }
    if cli.log_scale {
        app.log_scale = true;
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Main loop
    let result = run_loop(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        terminal.draw(|f| app.draw(f))?;
        if !app.handle_event()? {
            break;
        }
    }
    Ok(())
}
