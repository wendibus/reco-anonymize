//! Persistent app settings in the platform config directory
//! (`~/.config/reco-anonymize`, `~/Library/Application Support/reco-anonymize`,
//! `%APPDATA%\reco-anonymize`).

use std::path::PathBuf;

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

fn path() -> Option<PathBuf> {
    let d = dirs::config_dir()?.join("reco-anonymize");
    std::fs::create_dir_all(&d).ok()?;
    Some(d.join("settings.json"))
}

impl Settings {
    pub fn load() -> Self {
        let mut s: Settings = path()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        // forget models that were moved or deleted
        s.model = s.model.filter(|p| p.exists());
        s.text_model = s.text_model.filter(|p| p.exists());
        s
    }

    pub fn save(&self) {
        if let Some(p) = path() {
            let _ = std::fs::write(p, serde_json::to_vec_pretty(self).expect("serializable"));
        }
    }
}
