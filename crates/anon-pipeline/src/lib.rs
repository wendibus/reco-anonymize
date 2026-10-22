//! Orchestration: analyze → review gate → render → verify.

pub mod analyze;
pub mod audit;
pub mod dataset;
pub mod gate;

pub use PipelineError as Error;
pub use analyze::{AnalyzeOptions, AnalyzeSummary, analyze};
pub use audit::{AuditHit, AuditReport, ExportNote, acknowledge, audit, export_clean};
pub use dataset::{ExportOptions, ExportSummary, export_dataset};
pub use gate::{RenderOptions, RenderSummary, Status, approve, render, status};

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Store(#[from] anon_model::StoreError),
    #[error(transparent)]
    Io(#[from] anon_io::IoError),
    #[error("file error: {0}")]
    File(#[from] std::io::Error),
    #[error("refused: {0}")]
    Refused(String),
    #[error("verification failed: {0}")]
    Verify(String),
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
