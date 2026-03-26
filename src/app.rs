use crate::store::MultiStore;
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Chart, Clear, Dataset, GraphType, List, ListItem, ListState,
        Paragraph,
    },
    Frame,
};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Colors assigned to experiments (one color per experiment).
const EXP_COLORS: &[Color] = &[
    Color::Cyan,
    Color::Yellow,
    Color::Green,
    Color::Magenta,
    Color::Red,
    Color::Blue,
    Color::LightCyan,
    Color::LightYellow,
    Color::LightGreen,
    Color::LightMagenta,
];

/// Which panel has focus.
#[derive(Clone, Copy, PartialEq)]
enum Panel {
    Experiments,
    Metrics,
}

/// An item in the metric tree display list.
#[derive(Clone)]
enum MetricNode {
    /// Collapsible group header (e.g. "train")
    Group(String),
    /// Leaf metric: (index into visible_tags, display name)
    Leaf(usize, String),
}

pub struct App {
    pub store: MultiStore,
    pub dir: PathBuf,

    /// Which panel is focused
    panel: Panel,
    /// Whether we have multiple experiments (controls layout)
    multi_exp: bool,

    // -- Experiment list --
    exp_names: Vec<String>,
    exp_selected: Vec<bool>,
    exp_list_state: ListState,

    // -- Metric list --
    visible_tags: Vec<String>,
    metric_selected: Vec<bool>,
    metric_list_state: ListState,
    /// Tree display list for metrics (groups + leaves)
    metric_tree: Vec<MetricNode>,
    /// Which metric groups are collapsed
    collapsed_groups: HashSet<String>,

    /// Filter string for metrics
    pub filter: String,
    pub filter_mode: bool,
    /// Log scale for Y axis
    pub log_scale: bool,
    /// Last reload time
    pub last_reload: Instant,
    /// Show wall time instead of step on X axis
    pub wall_time_axis: bool,
    /// EMA smoothing factor (0.0 = off, 0.99 = very smooth)
    pub smoothing: f64,
    /// Current mouse position in terminal coordinates
    mouse_pos: Option<(u16, u16)>,
    /// Chart plotting area from last draw (for mouse → data mapping)
    chart_area: Rect,
    /// Chart data bounds from last draw [x_min, x_max, y_min, y_max]
    chart_bounds: [f64; 4],
    /// Y-axis label width from last draw
    y_label_width: u16,
    /// Zoom bounds [x_min, x_max, y_min, y_max] overriding auto bounds
    zoom_bounds: Option<[f64; 4]>,
    /// Terminal position where mouse drag started
    drag_start: Option<(u16, u16)>,
    /// Time of last mouse click (for double-click detection)
    last_click: Instant,
    /// Pan state: (terminal start pos, zoom bounds at pan start)
    pan_start: Option<((u16, u16), [f64; 4])>,
}

impl App {
    pub fn new(store: MultiStore, dir: PathBuf) -> Self {
        let multi_exp = store.experiments.len() > 1;
        let exp_names: Vec<String> = store.experiment_names().iter().map(|s| s.to_string()).collect();
        // Select all experiments by default
        let exp_selected = vec![true; exp_names.len()];
        let mut exp_list_state = ListState::default();
        if !exp_names.is_empty() {
            exp_list_state.select(Some(0));
        }

        let visible_tags = store.all_tags();
        let metric_selected = vec![false; visible_tags.len()];
        let collapsed_groups = HashSet::new();
        let metric_tree = Self::build_metric_tree(&visible_tags, &collapsed_groups);
        let mut metric_list_state = ListState::default();
        if !metric_tree.is_empty() {
            metric_list_state.select(Some(0));
        }

        let panel = if multi_exp {
            Panel::Experiments
        } else {
            Panel::Metrics
        };

        Self {
            store,
            dir,
            panel,
            multi_exp,
            exp_names,
            exp_selected,
            exp_list_state,
            visible_tags,
            metric_selected,
            metric_list_state,
            metric_tree,
            collapsed_groups,
            filter: String::new(),
            filter_mode: false,
            log_scale: false,
            last_reload: Instant::now(),
            wall_time_axis: false,
            smoothing: 0.0,
            mouse_pos: None,
            chart_area: Rect::default(),
            chart_bounds: [0.0; 4],
            y_label_width: 8,
            zoom_bounds: None,
            drag_start: None,
            last_click: Instant::now(),
            pan_start: None,
        }
    }

    /// Build the metric tree display list from tags and collapsed state.
    fn build_metric_tree(visible_tags: &[String], collapsed_groups: &HashSet<String>) -> Vec<MetricNode> {
        let mut groups: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
        let mut ungrouped: Vec<(usize, String)> = Vec::new();

        for (i, tag) in visible_tags.iter().enumerate() {
            if let Some(slash) = tag.find('/') {
                let group = tag[..slash].to_string();
                let rest = tag[slash + 1..].to_string();
                groups.entry(group).or_default().push((i, rest));
            } else {
                ungrouped.push((i, tag.clone()));
            }
        }

        let mut tree = Vec::new();
        for (idx, name) in ungrouped {
            tree.push(MetricNode::Leaf(idx, name));
        }
        for (group, members) in &groups {
            tree.push(MetricNode::Group(group.clone()));
            if !collapsed_groups.contains(group) {
                for (idx, name) in members {
                    tree.push(MetricNode::Leaf(*idx, name.clone()));
                }
            }
        }
        tree
    }

    fn rebuild_metric_tree(&mut self) {
        self.metric_tree = Self::build_metric_tree(&self.visible_tags, &self.collapsed_groups);
        // Clamp cursor
        if let Some(idx) = self.metric_list_state.selected() {
            if idx >= self.metric_tree.len() {
                self.metric_list_state.select(if self.metric_tree.is_empty() {
                    None
                } else {
                    Some(self.metric_tree.len() - 1)
                });
            }
        }
    }

    fn update_visible_tags(&mut self) {
        let all_tags = self.store.all_tags();
        if self.filter.is_empty() {
            self.visible_tags = all_tags;
        } else {
            let lower_filter = self.filter.to_lowercase();
            self.visible_tags = all_tags
                .into_iter()
                .filter(|t| t.to_lowercase().contains(&lower_filter))
                .collect();
        }
        // Preserve selections by tag name
        let old_selected: HashSet<String> = self
            .store
            .all_tags()
            .into_iter()
            .enumerate()
            .filter(|(i, _)| self.metric_selected.get(*i).copied().unwrap_or(false))
            .map(|(_, t)| t)
            .collect();
        self.metric_selected = self
            .visible_tags
            .iter()
            .map(|t| old_selected.contains(t))
            .collect();
        self.rebuild_metric_tree();
    }

    /// Get indices of selected experiments.
    fn selected_exp_indices(&self) -> Vec<usize> {
        self.exp_selected
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get selected metric tag names.
    fn selected_metric_tags(&self) -> Vec<&str> {
        self.visible_tags
            .iter()
            .enumerate()
            .filter(|(i, _)| self.metric_selected.get(*i).copied().unwrap_or(false))
            .map(|(_, t)| t.as_str())
            .collect()
    }

    /// Map terminal coordinates to data coordinates using last frame's chart geometry.
    fn terminal_to_data(&self, tx: u16, ty: u16) -> Option<(f64, f64)> {
        let area = self.chart_area;
        let [x_min, x_max, y_min, y_max] = self.chart_bounds;
        let y_lw = self.y_label_width;

        let plot_x = area.x + 1 + y_lw + 1;
        let plot_y = area.y + 1;
        let plot_w = area.width.saturating_sub(3 + y_lw);
        let plot_h = area.height.saturating_sub(4);

        if tx < plot_x || tx >= plot_x + plot_w || ty < plot_y || ty >= plot_y + plot_h || plot_w == 0 || plot_h == 0 {
            return None;
        }

        let frac_x = (tx - plot_x) as f64 / plot_w as f64;
        let frac_y = 1.0 - (ty - plot_y) as f64 / plot_h as f64;
        Some((
            x_min + frac_x * (x_max - x_min),
            y_min + frac_y * (y_max - y_min),
        ))
    }

    /// Zoom in/out centered on a terminal position. Factor < 1 zooms in, > 1 zooms out.
    fn zoom_at(&mut self, pos: (u16, u16), factor: f64) {
        let Some((cx, cy)) = self.terminal_to_data(pos.0, pos.1) else { return };
        let [x_min, x_max, y_min, y_max] = self.zoom_bounds.unwrap_or(self.chart_bounds);
        self.zoom_bounds = Some([
            cx - (cx - x_min) * factor,
            cx + (x_max - cx) * factor,
            cy - (cy - y_min) * factor,
            cy + (y_max - cy) * factor,
        ]);
    }

    pub fn reload(&mut self) -> Result<()> {
        self.store.reload(&self.dir)?;
        self.exp_names = self.store.experiment_names().iter().map(|s| s.to_string()).collect();
        // Grow/shrink exp_selected
        self.exp_selected.resize(self.exp_names.len(), false);
        self.update_visible_tags();
        self.last_reload = Instant::now();
        Ok(())
    }

    // -- Active list helpers (to share navigation logic) --

    fn active_list_len(&self) -> usize {
        match self.panel {
            Panel::Experiments => self.exp_names.len(),
            Panel::Metrics => self.metric_tree.len(),
        }
    }

    fn active_list_state(&self) -> &ListState {
        match self.panel {
            Panel::Experiments => &self.exp_list_state,
            Panel::Metrics => &self.metric_list_state,
        }
    }

    fn active_list_state_mut(&mut self) -> &mut ListState {
        match self.panel {
            Panel::Experiments => &mut self.exp_list_state,
            Panel::Metrics => &mut self.metric_list_state,
        }
    }

    fn active_selected_mut(&mut self) -> &mut Vec<bool> {
        match self.panel {
            Panel::Experiments => &mut self.exp_selected,
            Panel::Metrics => &mut self.metric_selected,
        }
    }

    /// Returns false when the app should exit.
    pub fn handle_event(&mut self) -> Result<bool> {
        if !event::poll(Duration::from_millis(250))? {
            if self.last_reload.elapsed() > Duration::from_secs(30) {
                let _ = self.reload();
            }
            return Ok(true);
        }

        let ev = event::read()?;

        // Handle mouse events
        if let Event::Mouse(mouse) = &ev {
            match mouse.kind {
                MouseEventKind::Down(MouseButton::Left) => {
                    let now = Instant::now();
                    if now.duration_since(self.last_click) < Duration::from_millis(300) {
                        // Double-click: reset zoom
                        self.zoom_bounds = None;
                        self.drag_start = None;
                    } else {
                        self.drag_start = Some((mouse.column, mouse.row));
                    }
                    self.last_click = now;
                }
                MouseEventKind::Down(MouseButton::Right) => {
                    // Right-click: start panning
                    let zb = self.zoom_bounds.unwrap_or(self.chart_bounds);
                    self.pan_start = Some(((mouse.column, mouse.row), zb));
                }
                MouseEventKind::Up(MouseButton::Left) => {
                    if let Some(start) = self.drag_start.take() {
                        let end = (mouse.column, mouse.row);
                        let dx = (end.0 as i32 - start.0 as i32).abs();
                        let dy = (end.1 as i32 - start.1 as i32).abs();
                        if dx > 3 || dy > 3 {
                            if let (Some(d0), Some(d1)) = (
                                self.terminal_to_data(start.0, start.1),
                                self.terminal_to_data(end.0, end.1),
                            ) {
                                self.zoom_bounds = Some([
                                    d0.0.min(d1.0),
                                    d0.0.max(d1.0),
                                    d0.1.min(d1.1),
                                    d0.1.max(d1.1),
                                ]);
                            }
                        }
                    }
                }
                MouseEventKind::Up(MouseButton::Right) => {
                    self.pan_start = None;
                }
                MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                    self.mouse_pos = Some((mouse.column, mouse.row));
                    // Pan: shift zoom bounds based on drag delta
                    if let Some(((tx0, ty0), orig)) = self.pan_start {
                        let plot_w = self.chart_area.width.saturating_sub(3 + self.y_label_width) as f64;
                        let plot_h = self.chart_area.height.saturating_sub(4) as f64;
                        if plot_w > 0.0 && plot_h > 0.0 {
                            let dx = -(mouse.column as f64 - tx0 as f64) * (orig[1] - orig[0]) / plot_w;
                            let dy = (mouse.row as f64 - ty0 as f64) * (orig[3] - orig[2]) / plot_h;
                            self.zoom_bounds = Some([
                                orig[0] + dx, orig[1] + dx,
                                orig[2] + dy, orig[3] + dy,
                            ]);
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    if let Some(pos) = self.mouse_pos {
                        self.zoom_at(pos, 0.8);
                    }
                }
                MouseEventKind::ScrollDown => {
                    if let Some(pos) = self.mouse_pos {
                        self.zoom_at(pos, 1.25);
                    }
                }
                _ => {}
            }
            return Ok(true);
        }

        let Event::Key(key) = ev else {
            return Ok(true);
        };
        if key.kind != KeyEventKind::Press {
            return Ok(true);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Ok(false);
        }

        if self.filter_mode {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    self.filter_mode = false;
                }
                KeyCode::Backspace => {
                    self.filter.pop();
                    self.update_visible_tags();
                }
                KeyCode::Char(c) => {
                    self.filter.push(c);
                    self.update_visible_tags();
                }
                _ => {}
            }
            return Ok(true);
        }

        match key.code {
            KeyCode::Char('q') => return Ok(false),
            KeyCode::Tab | KeyCode::BackTab => {
                if self.multi_exp {
                    self.panel = match self.panel {
                        Panel::Experiments => Panel::Metrics,
                        Panel::Metrics => Panel::Experiments,
                    };
                }
            }
            KeyCode::Char('/') => {
                self.filter_mode = true;
                // Switch to metrics panel when filtering
                self.panel = Panel::Metrics;
            }
            KeyCode::Char('c') => {
                self.filter.clear();
                self.update_visible_tags();
            }
            KeyCode::Char('l') => {
                self.log_scale = !self.log_scale;
            }
            KeyCode::Char('r') => {
                let _ = self.reload();
            }
            KeyCode::Char('t') => {
                self.wall_time_axis = !self.wall_time_axis;
            }
            KeyCode::Char('s') => {
                // Decrease smoothing
                self.smoothing = (self.smoothing - 0.1).max(0.0);
                // Snap to 0 if close
                if self.smoothing < 0.05 { self.smoothing = 0.0; }
            }
            KeyCode::Char('S') => {
                // Increase smoothing
                if self.smoothing < 0.05 {
                    self.smoothing = 0.6;
                } else {
                    self.smoothing = (self.smoothing + 0.1).min(0.99);
                }
            }
            KeyCode::Char('a') => {
                let sel = self.active_selected_mut();
                let all = sel.iter().all(|&s| s);
                for s in sel.iter_mut() {
                    *s = !all;
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if let Some(idx) = self.active_list_state().selected() {
                    if idx > 0 {
                        self.active_list_state_mut().select(Some(idx - 1));
                    }
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let len = self.active_list_len();
                if let Some(idx) = self.active_list_state().selected() {
                    if idx + 1 < len {
                        self.active_list_state_mut().select(Some(idx + 1));
                    }
                }
            }
            KeyCode::Home => {
                if self.active_list_len() > 0 {
                    self.active_list_state_mut().select(Some(0));
                }
            }
            KeyCode::End => {
                let len = self.active_list_len();
                if len > 0 {
                    self.active_list_state_mut().select(Some(len - 1));
                }
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                match self.panel {
                    Panel::Experiments => {
                        if let Some(idx) = self.exp_list_state.selected() {
                            if let Some(s) = self.exp_selected.get_mut(idx) {
                                *s = !*s;
                            }
                        }
                    }
                    Panel::Metrics => {
                        if let Some(idx) = self.metric_list_state.selected() {
                            match self.metric_tree.get(idx).cloned() {
                                Some(MetricNode::Group(name)) => {
                                    if !self.collapsed_groups.remove(&name) {
                                        self.collapsed_groups.insert(name);
                                    }
                                    self.rebuild_metric_tree();
                                }
                                Some(MetricNode::Leaf(tag_idx, _)) => {
                                    if let Some(s) = self.metric_selected.get_mut(tag_idx) {
                                        *s = !*s;
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(true)
    }

    pub fn draw(&mut self, frame: &mut Frame) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Title bar
                Constraint::Min(0),   // Main content
                Constraint::Length(1), // Status bar
            ])
            .split(frame.area());

        // Title
        let n_exp = self.store.experiments.len();
        let n_sel_exp = self.exp_selected.iter().filter(|&&s| s).count();
        let title = Line::from(vec![
            Span::styled(" clicurve ", Style::default().fg(Color::Black).bg(Color::Cyan)),
            Span::raw(format!(
                " {} | {} xps ({} selected) | {} metrics | {} event files",
                self.dir.display(),
                n_exp,
                n_sel_exp,
                self.store.total_metrics(),
                self.store.total_event_files(),
            )),
        ]);
        frame.render_widget(Paragraph::new(title), outer[0]);

        // Main layout
        if self.multi_exp {
            let main = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(36), Constraint::Min(40)])
                .split(outer[1]);

            // Left side: split vertically for experiments + metrics
            let left = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(n_exp as u16 + 2), Constraint::Min(4)])
                .split(main[0]);

            self.draw_exp_list(frame, left[0]);
            self.draw_metric_list(frame, left[1]);
            self.draw_chart(frame, main[1]);
        } else {
            let main = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(36), Constraint::Min(40)])
                .split(outer[1]);

            self.draw_metric_list(frame, main[0]);
            self.draw_chart(frame, main[1]);
        }

        // Status bar
        let status = if self.filter_mode {
            Line::from(vec![
                Span::styled(" FILTER: ", Style::default().fg(Color::Black).bg(Color::Yellow)),
                Span::raw(&self.filter),
                Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK)),
                Span::raw("  (Enter/Esc to confirm)"),
            ])
        } else {
            let log_indicator = if self.log_scale { " LOG" } else { "" };
            let mut spans = vec![
                Span::styled(" q", Style::default().fg(Color::Cyan)),
                Span::raw(" quit "),
            ];
            if self.multi_exp {
                spans.push(Span::styled("Tab", Style::default().fg(Color::Cyan)));
                spans.push(Span::raw(" panel "));
            }
            spans.extend([
                Span::styled("Space", Style::default().fg(Color::Cyan)),
                Span::raw(" toggle "),
                Span::styled("/", Style::default().fg(Color::Cyan)),
                Span::raw(" filter "),
                Span::styled("c", Style::default().fg(Color::Cyan)),
                Span::raw(" clear "),
                Span::styled("l", Style::default().fg(Color::Cyan)),
                Span::raw(format!(" log{log_indicator} ")),
                Span::styled("r", Style::default().fg(Color::Cyan)),
                Span::raw(" reload "),
                Span::styled("a", Style::default().fg(Color::Cyan)),
                Span::raw(" all "),
                Span::styled("t", Style::default().fg(Color::Cyan)),
                Span::raw(" x-axis "),
                Span::styled("s/S", Style::default().fg(Color::Cyan)),
                Span::raw(if self.smoothing > 0.0 {
                    format!(" smooth {:.0}% ", self.smoothing * 100.0)
                } else {
                    " smooth ".to_string()
                }),
            ]);
            Line::from(spans)
        };
        frame.render_widget(Paragraph::new(status), outer[2]);
    }

    fn draw_exp_list(&mut self, frame: &mut Frame, area: Rect) {
        let focused = self.panel == Panel::Experiments;
        let items: Vec<ListItem> = self
            .exp_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let selected = self.exp_selected.get(i).copied().unwrap_or(false);
                let color = if selected {
                    EXP_COLORS[i % EXP_COLORS.len()]
                } else {
                    Color::DarkGray
                };
                let marker = if selected { "● " } else { "  " };
                let n_files = self
                    .store
                    .experiments
                    .get(i)
                    .map(|(_, _, s)| s.event_files.len())
                    .unwrap_or(0);
                ListItem::new(format!("{marker}{name} ({n_files} files)"))
                    .style(Style::default().fg(color))
            })
            .collect();

        let border_style = if focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(border_style)
                    .title(" Experiments "),
            )
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::DarkGray),
            );

        frame.render_stateful_widget(list, area, &mut self.exp_list_state);
    }

    fn draw_metric_list(&mut self, frame: &mut Frame, area: Rect) {
        let focused = self.panel == Panel::Metrics;
        let items: Vec<ListItem> = self
            .metric_tree
            .iter()
            .map(|node| match node {
                MetricNode::Group(name) => {
                    let collapsed = self.collapsed_groups.contains(name);
                    let arrow = if collapsed { "▶" } else { "▼" };
                    let prefix = format!("{}/", name);
                    let n_sel = self.visible_tags.iter().enumerate()
                        .filter(|(i, t)| t.starts_with(&prefix) && self.metric_selected.get(*i).copied().unwrap_or(false))
                        .count();
                    let n_total = self.visible_tags.iter().filter(|t| t.starts_with(&prefix)).count();
                    let style = if n_sel > 0 {
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    ListItem::new(format!("{arrow} {name}/ ({n_sel}/{n_total})")).style(style)
                }
                MetricNode::Leaf(idx, display_name) => {
                    let selected = self.metric_selected.get(*idx).copied().unwrap_or(false);
                    let in_group = self.visible_tags.get(*idx).is_some_and(|t| t.contains('/'));
                    let indent = if in_group { "  " } else { "" };
                    let marker = if selected { "● " } else { "  " };
                    let style = if selected {
                        Style::default().fg(Color::White)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    ListItem::new(format!("{indent}{marker}{display_name}")).style(style)
                }
            })
            .collect();

        let title = if self.filter.is_empty() {
            " Metrics ".to_string()
        } else {
            format!(" Metrics [{}] ", self.filter)
        };

        let border_style = if focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(border_style)
                    .title(title),
            )
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::DarkGray),
            );

        frame.render_stateful_widget(list, area, &mut self.metric_list_state);
    }

    fn draw_chart(&mut self, frame: &mut Frame, area: Rect) {
        let sel_exps = self.selected_exp_indices();
        let sel_tags = self.selected_metric_tags();

        if sel_exps.is_empty() || sel_tags.is_empty() {
            let hint = if sel_exps.is_empty() {
                "Select experiments with Space (Tab to switch panels)"
            } else {
                "Select metrics with Space to plot them"
            };
            let msg = Paragraph::new(format!("\n  {hint}"))
                .block(Block::default().borders(Borders::ALL).title(" Chart "));
            frame.render_widget(msg, area);
            return;
        }

        // Build datasets: one line per (experiment, metric) pair.
        // Color by experiment, with legend showing "exp_name / metric".
        // When smoothing is active, we add two datasets per series:
        //   1. Raw data (faded color)
        //   2. EMA-smoothed line (bright color, with legend label)
        struct SeriesData {
            label: String,
            raw: Vec<(f64, f64)>,
            smoothed: Vec<(f64, f64)>,
            color: Color,
        }
        let mut all_series: Vec<SeriesData> = Vec::new();
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let use_smoothing = self.smoothing > 0.0;

        for &exp_idx in &sel_exps {
            let color = EXP_COLORS[exp_idx % EXP_COLORS.len()];
            let (exp_name, _, store) = &self.store.experiments[exp_idx];

            for &tag in &sel_tags {
                let Some(points) = store.metrics.get(tag) else {
                    continue;
                };
                if points.is_empty() {
                    continue;
                }

                let t0 = points.first().map(|p| p.wall_time).unwrap_or(0.0);

                let raw: Vec<(f64, f64)> = points
                    .iter()
                    .filter_map(|p| {
                        let x = if self.wall_time_axis {
                            (p.wall_time - t0) / 3600.0
                        } else {
                            p.step as f64
                        };
                        let y = if self.log_scale {
                            let v = p.value as f64;
                            if v > 0.0 { v.ln() } else { return None; }
                        } else {
                            p.value as f64
                        };
                        Some((x, y))
                    })
                    .collect();

                let smoothed = if use_smoothing {
                    ema_smooth(&raw, self.smoothing)
                } else {
                    Vec::new()
                };

                // Bounds from the displayed data (smoothed if active, else raw)
                let bounds_data = if use_smoothing { &smoothed } else { &raw };
                for &(x, y) in bounds_data {
                    if y < y_min { y_min = y; }
                    if y > y_max { y_max = y; }
                    if x < x_min { x_min = x; }
                    if x > x_max { x_max = x; }
                }

                let label = if self.multi_exp && sel_exps.len() > 1 {
                    format!("{} / {}", exp_name, shorten_tag(tag))
                } else {
                    shorten_tag(tag)
                };
                all_series.push(SeriesData { label, raw, smoothed, color });
            }
        }

        if all_series.is_empty() || y_min > y_max {
            let msg = Paragraph::new("\n  No data points for selected metrics")
                .block(Block::default().borders(Borders::ALL).title(" Chart "));
            frame.render_widget(msg, area);
            return;
        }

        // Padding
        let y_range = y_max - y_min;
        let y_pad = if y_range.abs() < 1e-10 { 1.0 } else { y_range * 0.05 };
        y_min -= y_pad;
        y_max += y_pad;
        let x_range = x_max - x_min;
        let x_pad = if x_range.abs() < 1e-10 { 1.0 } else { x_range * 0.02 };
        x_min -= x_pad;
        x_max += x_pad;

        // Apply zoom if set
        if let Some([zx_min, zx_max, zy_min, zy_max]) = self.zoom_bounds {
            x_min = zx_min;
            x_max = zx_max;
            y_min = zy_min;
            y_max = zy_max;
        }

        let datasets: Vec<Dataset> = if use_smoothing {
            // When smoothing: raw data faded + smoothed line bright
            all_series
                .iter()
                .flat_map(|s| {
                    let faded = dim_color(s.color);
                    let raw_ds = Dataset::default()
                        .name(String::new())
                        .marker(symbols::Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(faded))
                        .data(&s.raw);
                    let smooth_ds = Dataset::default()
                        .name(s.label.clone())
                        .marker(symbols::Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(s.color))
                        .data(&s.smoothed);
                    [raw_ds, smooth_ds]
                })
                .collect()
        } else {
            all_series
                .iter()
                .map(|s| {
                    Dataset::default()
                        .name(s.label.clone())
                        .marker(symbols::Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(s.color))
                        .data(&s.raw)
                })
                .collect()
        };

        let x_title = if self.wall_time_axis { "Hours" } else { "Step" };
        let y_title = if self.log_scale { "ln(value)" } else { "Value" };

        let x_labels = make_axis_labels(x_min, x_max, 5);
        let y_labels = make_axis_labels(y_min, y_max, 5);

        // Compute y-axis label width for mouse mapping
        let y_lw = y_labels.iter().map(|l| l.len()).max().unwrap_or(6) as u16;
        self.chart_area = area;
        self.chart_bounds = [x_min, x_max, y_min, y_max];
        self.y_label_width = y_lw;

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(" Chart "))
            .x_axis(
                Axis::default()
                    .title(x_title)
                    .bounds([x_min, x_max])
                    .labels(x_labels),
            )
            .y_axis(
                Axis::default()
                    .title(y_title)
                    .bounds([y_min, y_max])
                    .labels(y_labels),
            );

        frame.render_widget(chart, area);

        // Selection rectangle during drag
        if let (Some(start), Some(current)) = (self.drag_start, self.mouse_pos) {
            let rx = start.0.min(current.0);
            let ry = start.1.min(current.1);
            let rw = (start.0.max(current.0) - rx).saturating_add(1);
            let rh = (start.1.max(current.1) - ry).saturating_add(1);
            if rw > 1 && rh > 1 {
                let sel_rect = Rect::new(rx, ry, rw, rh);
                frame.render_widget(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::White)),
                    sel_rect,
                );
            }
        }

        // Hover tooltip
        if let Some((mx, my)) = self.mouse_pos {
            // Compute the inner plotting region of the chart:
            // border(1) + y_labels(y_lw) + gap(1) on left, border(1) on right
            // border(1) on top, x_labels(1) + x_title(1) + border(1) on bottom
            let plot_x = area.x + 1 + y_lw + 1;
            let plot_y = area.y + 1;
            let plot_w = area.width.saturating_sub(3 + y_lw);
            let plot_h = area.height.saturating_sub(4); // top border + bottom border + labels + title

            if mx >= plot_x
                && mx < plot_x + plot_w
                && my >= plot_y
                && my < plot_y + plot_h
                && plot_w > 0
                && plot_h > 0
            {
                // Map terminal coords to data coords
                let frac_x = (mx - plot_x) as f64 / plot_w as f64;
                let frac_y = 1.0 - (my - plot_y) as f64 / plot_h as f64;
                let data_x = x_min + frac_x * (x_max - x_min);
                let data_y = y_min + frac_y * (y_max - y_min);

                // Find nearest point across all visible series (use displayed data: smoothed or raw)
                let mut best: Option<(&str, f64, f64, f64, Color)> = None; // (label, x, y, dist, color)
                let x_scale = if (x_max - x_min).abs() > 1e-15 { plot_w as f64 / (x_max - x_min) } else { 1.0 };
                let y_scale = if (y_max - y_min).abs() > 1e-15 { plot_h as f64 / (y_max - y_min) } else { 1.0 };

                for s in &all_series {
                    let data = if use_smoothing && !s.smoothed.is_empty() {
                        &s.smoothed
                    } else {
                        &s.raw
                    };
                    for &(px, py) in data {
                        // Distance in pixel space for fair comparison
                        let dx = (px - data_x) * x_scale;
                        let dy = (py - data_y) * y_scale;
                        let dist = dx * dx + dy * dy;
                        if best.is_none() || dist < best.unwrap().3 {
                            best = Some((&s.label, px, py, dist, s.color));
                        }
                    }
                }

                if let Some((label, px, py, dist, color)) = best {
                    // Only show tooltip if reasonably close (within ~5 cells)
                    if dist < 25.0 * 25.0 {
                        let val_str = if self.log_scale {
                            format!("{:.4} (raw: {:.4})", py, py.exp())
                        } else {
                            format!("{:.4}", py)
                        };
                        let x_str = if self.wall_time_axis {
                            format!("{:.2}h", px)
                        } else {
                            format!("step {}", px as i64)
                        };
                        let line1 = format!(" {} ", label);
                        let line2 = format!(" {} = {} ", x_str, val_str);
                        let w = line1.len().max(line2.len()) as u16;
                        let h = 2u16;

                        // Position tooltip near cursor, keep inside chart
                        let tx = if mx + 2 + w < area.x + area.width {
                            mx + 2
                        } else {
                            mx.saturating_sub(w + 1)
                        };
                        let ty = if my + 1 + h < area.y + area.height {
                            my + 1
                        } else {
                            my.saturating_sub(h)
                        };

                        let tooltip_area = Rect::new(tx, ty, w, h);
                        frame.render_widget(Clear, tooltip_area);
                        let tooltip = Paragraph::new(vec![
                            Line::from(Span::styled(
                                line1,
                                Style::default().fg(color).add_modifier(Modifier::BOLD),
                            )),
                            Line::from(Span::styled(
                                line2,
                                Style::default().fg(Color::White),
                            )),
                        ])
                        .style(Style::default().bg(Color::Rgb(30, 30, 30)));
                        frame.render_widget(tooltip, tooltip_area);
                    }
                }
            }
        }
    }
}

fn shorten_tag(tag: &str) -> String {
    let parts: Vec<&str> = tag.split('/').collect();
    if parts.len() <= 2 {
        tag.to_string()
    } else {
        parts[parts.len() - 2..].join("/")
    }
}

fn make_axis_labels(min: f64, max: f64, count: usize) -> Vec<String> {
    let step = (max - min) / (count - 1) as f64;
    (0..count)
        .map(|i| {
            let val = min + step * i as f64;
            format_number(val)
        })
        .collect()
}

/// Exponential moving average with bias correction (same as TensorBoard).
fn ema_smooth(data: &[(f64, f64)], alpha: f64) -> Vec<(f64, f64)> {
    if data.is_empty() || alpha <= 0.0 {
        return data.to_vec();
    }
    let mut result = Vec::with_capacity(data.len());
    let mut ema = 0.0;
    let mut correction = 1.0; // tracks alpha^n for debiasing
    for &(x, y) in data {
        ema = alpha * ema + (1.0 - alpha) * y;
        correction *= alpha;
        // Bias-corrected value
        let debiased = ema / (1.0 - correction);
        result.push((x, debiased));
    }
    result
}

/// Return a dimmed version of a color for showing raw data behind smoothed.
fn dim_color(c: Color) -> Color {
    match c {
        Color::Cyan => Color::Rgb(0, 80, 80),
        Color::Yellow => Color::Rgb(80, 80, 0),
        Color::Green => Color::Rgb(0, 80, 0),
        Color::Magenta => Color::Rgb(80, 0, 80),
        Color::Red => Color::Rgb(80, 0, 0),
        Color::Blue => Color::Rgb(0, 0, 80),
        Color::LightCyan => Color::Rgb(0, 60, 60),
        Color::LightYellow => Color::Rgb(60, 60, 0),
        Color::LightGreen => Color::Rgb(0, 60, 0),
        Color::LightMagenta => Color::Rgb(60, 0, 60),
        _ => Color::DarkGray,
    }
}

fn format_number(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs >= 1_000_000.0 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.1}k", v / 1_000.0)
    } else if abs >= 1.0 {
        format!("{:.2}", v)
    } else if abs >= 0.001 {
        format!("{:.4}", v)
    } else {
        format!("{:.2e}", v)
    }
}
