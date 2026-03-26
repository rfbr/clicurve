/// Metric data store: loads TensorBoard event files and organizes scalar metrics.
/// Supports both single-experiment and multi-experiment modes.
use crate::proto::{self, event, summary};
use crate::tfrecord::TfRecordReader;
use anyhow::{Context, Result};
use prost::Message;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ScalarPoint {
    pub step: i64,
    pub value: f32,
    pub wall_time: f64,
}

/// All scalar metrics from a single TensorBoard log directory.
#[derive(Debug, Clone)]
pub struct MetricStore {
    /// tag -> sorted list of points
    pub metrics: BTreeMap<String, Vec<ScalarPoint>>,
    pub event_files: Vec<PathBuf>,
}

impl MetricStore {
    pub fn load(dir: &Path) -> Result<Self> {
        let event_files = find_event_files(dir)?;
        let mut metrics: BTreeMap<String, Vec<ScalarPoint>> = BTreeMap::new();

        for path in &event_files {
            let file = File::open(path)
                .with_context(|| format!("opening {}", path.display()))?;
            let reader = TfRecordReader::new(BufReader::new(file));

            for record in reader {
                let data = match record {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let event = match proto::Event::decode(data.as_slice()) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                if let Some(event::What::Summary(summary)) = event.what {
                    for val in summary.value {
                        if let Some(summary::value::Kind::SimpleValue(v)) = val.value {
                            metrics.entry(val.tag).or_default().push(ScalarPoint {
                                step: event.step,
                                value: v,
                                wall_time: event.wall_time,
                            });
                        }
                    }
                }
            }
        }

        for points in metrics.values_mut() {
            points.sort_by_key(|p| p.step);
        }

        Ok(Self {
            metrics,
            event_files,
        })
    }

    pub fn tags(&self) -> Vec<&str> {
        self.metrics.keys().map(|s| s.as_str()).collect()
    }
}

/// Multiple experiments, each with its own MetricStore.
#[derive(Debug, Clone)]
pub struct MultiStore {
    /// experiment name -> store
    pub experiments: Vec<(String, PathBuf, MetricStore)>,
}

impl MultiStore {
    /// Try to detect experiments in the given directory.
    /// Looks for subdirectories that contain a `tensorboard/` folder.
    /// Falls back to treating the directory itself as a single experiment.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut experiments = Vec::new();

        // Check if subdirs contain tensorboard/ folders
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut subdirs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .collect();
            subdirs.sort_by_key(|e| e.file_name());

            for entry in &subdirs {
                let tb_dir = entry.path().join("tensorboard");
                if tb_dir.is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    match MetricStore::load(&tb_dir) {
                        Ok(store) if !store.metrics.is_empty() => {
                            experiments.push((name, tb_dir, store));
                        }
                        _ => {}
                    }
                }
            }
        }

        // Fallback: treat the directory itself as a single experiment
        if experiments.is_empty() {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "default".to_string());
            let store = MetricStore::load(dir)?;
            experiments.push((name, dir.to_path_buf(), store));
        }

        Ok(Self { experiments })
    }

    pub fn reload(&mut self, dir: &Path) -> Result<()> {
        let fresh = Self::load(dir)?;
        *self = fresh;
        Ok(())
    }

    /// Union of all metric tags across all experiments.
    pub fn all_tags(&self) -> Vec<String> {
        let mut tags: BTreeMap<&str, ()> = BTreeMap::new();
        for (_, _, store) in &self.experiments {
            for tag in store.tags() {
                tags.entry(tag).or_default();
            }
        }
        tags.keys().map(|s| s.to_string()).collect()
    }

    pub fn experiment_names(&self) -> Vec<&str> {
        self.experiments.iter().map(|(n, _, _)| n.as_str()).collect()
    }

    pub fn total_event_files(&self) -> usize {
        self.experiments.iter().map(|(_, _, s)| s.event_files.len()).sum()
    }

    pub fn total_metrics(&self) -> usize {
        self.all_tags().len()
    }
}

fn find_event_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir).follow_links(true) {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy();
        if name.starts_with("events.out.tfevents.") {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}
