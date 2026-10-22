//! Project directory I/O with atomic writes.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::SCHEMA_VERSION;
use crate::frames::FrameTable;
use crate::project::Project;
use crate::track::TrackSet;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("project schema version {found} is not supported (expected {SCHEMA_VERSION})")]
    Schema { found: u32 },
}

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Handle to a `<source>.anon/` directory.
#[derive(Debug, Clone)]
pub struct ProjectDir {
    pub root: PathBuf,
}

impl ProjectDir {
    /// Default project location for a source video: `<source>.anon/`.
    pub fn for_source(source: &Path) -> Self {
        let mut name = source.file_name().unwrap_or_default().to_os_string();
        name.push(".anon");
        Self {
            root: source.with_file_name(name),
        }
    }

    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn create(&self) -> Result<(), StoreError> {
        fs::create_dir_all(&self.root).map_err(io_err(&self.root))
    }

    pub fn path(&self, file: &str) -> PathBuf {
        self.root.join(file)
    }

    fn write_atomic(&self, file: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let path = self.path(file);
        let tmp = self.path(&format!(".{file}.tmp"));
        {
            let mut f = fs::File::create(&tmp).map_err(io_err(&tmp))?;
            f.write_all(bytes).map_err(io_err(&tmp))?;
            f.sync_all().map_err(io_err(&tmp))?;
        }
        fs::rename(&tmp, &path).map_err(io_err(&path))
    }

    fn read_json<T: serde::de::DeserializeOwned>(&self, file: &str) -> Result<T, StoreError> {
        let path = self.path(file);
        let bytes = fs::read(&path).map_err(io_err(&path))?;
        serde_json::from_slice(&bytes).map_err(|source| StoreError::Json { path, source })
    }

    fn write_json<T: serde::Serialize>(&self, file: &str, v: &T) -> Result<(), StoreError> {
        let bytes = serde_json::to_vec_pretty(v).expect("serializable");
        self.write_atomic(file, &bytes)
    }

    pub fn load_project(&self) -> Result<Project, StoreError> {
        let p: Project = self.read_json("project.json")?;
        if p.schema_version != SCHEMA_VERSION {
            return Err(StoreError::Schema {
                found: p.schema_version,
            });
        }
        Ok(p)
    }

    pub fn save_project(&self, p: &Project) -> Result<(), StoreError> {
        self.write_json("project.json", p)
    }

    pub fn load_tracks(&self) -> Result<TrackSet, StoreError> {
        if !self.path("tracks.json").exists() {
            return Ok(TrackSet::default());
        }
        self.read_json("tracks.json")
    }

    pub fn save_tracks(&self, t: &TrackSet) -> Result<(), StoreError> {
        self.write_json("tracks.json", t)
    }

    pub fn load_frames(&self) -> Result<FrameTable, StoreError> {
        let path = self.path("frames.bin");
        if !path.exists() {
            return Ok(FrameTable::default());
        }
        Ok(FrameTable::from_bytes(
            &fs::read(&path).map_err(io_err(&path))?,
        ))
    }

    pub fn save_frames(&self, f: &FrameTable) -> Result<(), StoreError> {
        self.write_atomic("frames.bin", &f.to_bytes())
    }

    /// Append-only raw detection log (one JSON object per line).
    pub fn detections_writer(&self) -> Result<std::io::BufWriter<fs::File>, StoreError> {
        let path = self.path("detections.jsonl");
        let f = fs::File::create(&path).map_err(io_err(&path))?;
        Ok(std::io::BufWriter::new(f))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_path_next_to_source() {
        let d = ProjectDir::for_source(Path::new("/x/clip.mp4"));
        assert_eq!(d.root, PathBuf::from("/x/clip.mp4.anon"));
    }
}
