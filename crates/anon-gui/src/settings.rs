//! Persistent app settings in the platform config directory
//! (`~/.config/reco-anonymize`, `~/Library/Application Support/reco-anonymize`,
//! `%APPDATA%\reco-anonymize`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// "en", "de", "fr", "es"; `None` = system language.
    pub language: Option<String>,
    pub check_updates: bool,
    pub model: Option<PathBuf>,
    pub text_model: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: None,
            check_updates: true,
            model: None,
            text_model: None,
        }
    }
}

fn dir() -> Option<PathBuf> {
    let d = dirs::config_dir()?.join("reco-anonymize");
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

fn path() -> Option<PathBuf> {
    Some(dir()?.join("settings.json"))
}

/// Copy a chosen model and its manifest (`<model>.json`) into the settings
/// directory (`slot` = "model" or "text-model") and return the copy, so the
/// choice survives the original being moved, deleted or evicted from a cloud
/// drive. Falls back to the original path if copying fails.
pub fn adopt_model(src: &Path, slot: &str) -> PathBuf {
    let copy = || -> std::io::Result<PathBuf> {
        let d = dir()
            .ok_or_else(|| std::io::Error::other("no config dir"))?
            .join("models")
            .join(slot);
        if src.starts_with(&d) {
            return Ok(src.to_path_buf());
        }
        let name = src
            .file_name()
            .ok_or_else(|| std::io::Error::other("no file name"))?;
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d)?;
        let dst = d.join(name);
        std::fs::copy(src, &dst)?;
        let manifest = PathBuf::from(format!("{}.json", src.display()));
        if manifest.exists() {
            std::fs::copy(&manifest, format!("{}.json", dst.display()))?;
        }
        Ok(dst)
    };
    copy().unwrap_or_else(|e| {
        log::warn!("keeping {} in place: {e}", src.display());
        src.to_path_buf()
    })
}

impl Settings {
    pub fn load() -> Self {
        let mut s: Settings = path()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        // forget models that were moved or deleted; older settings point at the
        // original files, take a copy of those while they are still there
        let before = (s.model.clone(), s.text_model.clone());
        s.model = s
            .model
            .filter(|p| p.exists())
            .map(|p| adopt_model(&p, "model"));
        s.text_model = s
            .text_model
            .filter(|p| p.exists())
            .map(|p| adopt_model(&p, "text-model"));
        if (s.model.clone(), s.text_model.clone()) != before {
            s.save();
        }
        s
    }

    pub fn save(&self) {
        if let Some(p) = path() {
            let _ = std::fs::write(p, serde_json::to_vec_pretty(self).expect("serializable"));
        }
    }
}
