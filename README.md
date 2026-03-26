# clicurve

A fast, interactive TensorBoard viewer that runs entirely in your terminal.
No browser, no port forwarding, no fuss — just point it at your experiment logs.

Built in Rust. Reads TFEvent files directly (protobuf parsing, no Python dependency).

## Install

```bash
cargo install clicurve
```

Or build from source:

```bash
git clone https://github.com/rfbr/clicurve
cd clicurve
cargo build --release
# binary is at target/release/clicurve
```

## Usage

```bash
# Single experiment
clicurve /path/to/experiment/tensorboard/

# Multiple experiments — auto-detects subdirectories with tensorboard/ folders
clicurve /path/to/xps/

# Quick look at available metrics
clicurve /path/to/xps/ --list

# Pre-filter metrics and start with log scale
clicurve /path/to/xps/ -f "train/ce" -l
```

## Keybindings

| Key | Action |
|-----|--------|
| `j` / `k` or arrows | Navigate metric list |
| `Space` | Toggle metric or experiment selection |
| `Tab` | Switch between Experiments and Metrics panels |
| `/` | Filter metrics by name |
| `c` | Clear filter |
| `s` / `S` | Decrease / increase EMA smoothing |
| `l` | Toggle log scale |
| `t` | Toggle X-axis between step and wall-clock time |
| `a` | Select / deselect all |
| `r` | Reload data from disk |
| `q` | Quit |

### Mouse

| Action | Effect |
|--------|--------|
| Hover over chart | Tooltip with nearest data point value |
| Scroll up / down | Zoom in / out (centered on cursor) |
| Left-click drag | Rectangle selection zoom |
| Right-click drag | Pan the viewport |
| Double-click | Reset zoom to fit all data |

## Features

- **Multi-experiment comparison** — select multiple experiments, same metric plotted with distinct colors per experiment
- **EMA smoothing** — adjustable exponential moving average with bias correction (same algorithm as TensorBoard), raw data shown faded behind the smoothed line
- **Interactive zoom & pan** — scroll to zoom, drag to select a region, right-drag to pan, double-click to reset
- **Mouse hover** — tooltip with exact metric name, step, and value
- **Live reload** — auto-refreshes every 30s, or press `r` to reload manually
- **Fast** — reads TFRecord protobuf directly in Rust, handles hundreds of event files in seconds
- **Zero dependencies outside Rust** — no Python, no TensorFlow, no browser

## How it works

clicurve reads TensorBoard's binary event files (`events.out.tfevents.*`) directly using a minimal protobuf decoder. It extracts scalar metrics and renders them as interactive braille-character line charts using [ratatui](https://github.com/ratatui/ratatui).

## License

MIT
