//! Update check and self-update from GitHub releases.
//!
//! Privacy: the only request is to the GitHub releases API of this
//! repository (and, when installing, the release asset and its
//! `SHA256SUMS.txt`). No footage, paths or identifiers are sent.
//!
//! Install is fail-closed: an asset without a matching line in
//! `SHA256SUMS.txt`, or with a different hash, is refused. Files are
//! swapped by renaming the old file aside (works for running executables and
//! loaded DLLs on Windows too) and moving the new one in; leftovers are
//! removed on the next start by [`cleanup`].

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const REPO: &str = "wendibus/reco-anonymize";
const OLD_SUFFIX: &str = ".old-update";

#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("network: {0}")]
    Net(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Refused(String),
}

impl From<ureq::Error> for UpdateError {
    fn from(e: ureq::Error) -> Self {
        UpdateError::Net(e.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Release {
    pub version: String,
    pub page: String,
    pub asset: Asset,
    pub sums: Option<Asset>,
}

#[derive(Debug, Clone, Default)]
pub struct InstallReport {
    pub files: usize,
    pub dir: PathBuf,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

#[derive(Deserialize)]
struct GhRelease {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    assets: Vec<GhAsset>,
}

/// Release asset suffix for this build's platform.
pub fn platform_suffix() -> &'static str {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "windows-x86_64.zip"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "macos-arm64.tar.gz"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "linux-x86_64.tar.gz"
    } else {
        "unsupported"
    }
}

fn parse_version(v: &str) -> Option<(Vec<u64>, bool)> {
    let v = v.trim().trim_start_matches('v');
    let (core, pre) = match v.split_once('-') {
        Some((c, _)) => (c, true),
        None => (v, false),
    };
    let nums: Option<Vec<u64>> = core.split('.').map(|p| p.parse().ok()).collect();
    nums.map(|n| (n, pre))
}

/// `candidate` is a newer version than `current` (pre-releases rank below
/// the same release).
pub fn newer(candidate: &str, current: &str) -> bool {
    match (parse_version(candidate), parse_version(current)) {
        (Some((a, ap)), Some((b, bp))) => {
            let n = a.len().max(b.len());
            for i in 0..n {
                let (x, y) = (
                    a.get(i).copied().unwrap_or(0),
                    b.get(i).copied().unwrap_or(0),
                );
                if x != y {
                    return x > y;
                }
            }
            bp && !ap
        }
        _ => false,
    }
}

fn api_base() -> String {
    std::env::var("RECO_ANON_UPDATE_API").unwrap_or_else(|_| "https://api.github.com".into())
}

fn agent() -> String {
    format!("reco-anonymize/{}", env!("CARGO_PKG_VERSION"))
}

fn select(rel: GhRelease, suffix: &str) -> Option<Release> {
    if rel.draft || rel.prerelease {
        return None;
    }
    let pick = |pred: &dyn Fn(&str) -> bool| {
        rel.assets.iter().find(|a| pred(&a.name)).map(|a| Asset {
            name: a.name.clone(),
            url: a.browser_download_url.clone(),
            size: a.size,
        })
    };
    let asset = pick(&|n: &str| n.ends_with(suffix))?;
    let sums = pick(&|n: &str| n == "SHA256SUMS.txt");
    Some(Release {
        version: rel.tag_name.trim_start_matches('v').to_string(),
        page: rel.html_url.clone(),
        asset,
        sums,
    })
}

/// Latest release if it is newer than `current` and has an asset for this
/// platform.
pub fn check(current: &str) -> Result<Option<Release>, UpdateError> {
    check_for(current, platform_suffix())
}

pub fn check_for(current: &str, suffix: &str) -> Result<Option<Release>, UpdateError> {
    let url = format!("{}/repos/{REPO}/releases/latest", api_base());
    let mut resp = ureq::get(&url)
        .header("User-Agent", &agent())
        .header("Accept", "application/vnd.github+json")
        .call()?;
    let rel: GhRelease = resp
        .body_mut()
        .read_json()
        .map_err(|e| UpdateError::Net(e.to_string()))?;
    Ok(select(rel, suffix).filter(|r| newer(&r.version, current)))
}

fn download(
    url: &str,
    to: &mut impl Write,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<(), UpdateError> {
    let resp = ureq::get(url).header("User-Agent", &agent()).call()?;
    let total = resp.body().content_length();
    let mut r = resp.into_body().into_reader();
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        to.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    Ok(())
}

fn expected_sha(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        let (h, n) = (it.next()?, it.next()?);
        (n.trim_start_matches('*') == name).then(|| h.to_ascii_lowercase())
    })
}

fn extract(archive: &Path, name: &str, into: &Path) -> Result<(), UpdateError> {
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(fs::File::open(archive)?)
            .map_err(|e| UpdateError::Refused(format!("zip: {e}")))?;
        z.extract(into)
            .map_err(|e| UpdateError::Refused(format!("zip: {e}")))?;
    } else if name.ends_with(".tar.gz") {
        let gz = flate2::read::GzDecoder::new(fs::File::open(archive)?);
        let mut t = tar::Archive::new(gz);
        t.set_preserve_permissions(true);
        t.unpack(into)?;
    } else {
        return Err(UpdateError::Refused(format!(
            "unknown archive type: {name}"
        )));
    }
    Ok(())
}

/// Directory inside the extracted archive that contains `exe_name`.
fn find_root(dir: &Path, exe_name: &str) -> Option<PathBuf> {
    if dir.join(exe_name).is_file() {
        return Some(dir.to_path_buf());
    }
    for e in fs::read_dir(dir).ok()?.flatten() {
        if e.path().is_dir()
            && let Some(r) = find_root(&e.path(), exe_name)
        {
            return Some(r);
        }
    }
    None
}

fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
    }
    out
}

fn swap_in(new: &Path, target: &Path) -> Result<bool, UpdateError> {
    if target.exists() && fs::read(target)? == fs::read(new)? {
        return Ok(false);
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if target.exists() {
        let mut old = target.as_os_str().to_owned();
        old.push(OLD_SUFFIX);
        let old = PathBuf::from(old);
        let _ = fs::remove_file(&old);
        fs::rename(target, &old)?;
    }
    // copy (not rename): the temp dir may be on another filesystem
    let mut staged = target.as_os_str().to_owned();
    staged.push(".new-update");
    let staged = PathBuf::from(staged);
    fs::copy(new, &staged)?;
    fs::rename(&staged, target)?;
    Ok(true)
}

/// Download, verify and install `rel` into `dir`, whose files include
/// `exe_name` (the archive must contain it too).
pub fn install_into(
    rel: &Release,
    dir: &Path,
    exe_name: &str,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<InstallReport, UpdateError> {
    let sums_asset = rel.sums.as_ref().ok_or_else(|| {
        UpdateError::Refused("release has no SHA256SUMS.txt; refusing to install".into())
    })?;
    let mut sums = Vec::new();
    download(&sums_asset.url, &mut sums, |_, _| {})?;
    let sums = String::from_utf8_lossy(&sums);
    let want = expected_sha(&sums, &rel.asset.name).ok_or_else(|| {
        UpdateError::Refused(format!(
            "{} is not listed in SHA256SUMS.txt",
            rel.asset.name
        ))
    })?;

    let tmp = tempfile::tempdir()?;
    let archive = tmp.path().join(&rel.asset.name);
    {
        let mut f = fs::File::create(&archive)?;
        download(&rel.asset.url, &mut f, &mut progress)?;
        f.flush()?;
    }
    let got = hex::encode(Sha256::digest(fs::read(&archive)?));
    if got != want {
        return Err(UpdateError::Refused(format!(
            "checksum mismatch for {} (expected {want}, got {got})",
            rel.asset.name
        )));
    }
    let unpacked = tmp.path().join("unpacked");
    fs::create_dir_all(&unpacked)?;
    extract(&archive, &rel.asset.name, &unpacked)?;
    let root = find_root(&unpacked, exe_name)
        .ok_or_else(|| UpdateError::Refused(format!("{exe_name} not found in the update")))?;

    let mut files = 0;
    for f in files_under(&root) {
        let rel_path = f.strip_prefix(&root).expect("under root");
        if swap_in(&f, &dir.join(rel_path))? {
            files += 1;
        }
    }
    Ok(InstallReport {
        files,
        dir: dir.to_path_buf(),
    })
}

/// Install next to the running executable.
pub fn install(
    rel: &Release,
    progress: impl FnMut(u64, Option<u64>),
) -> Result<InstallReport, UpdateError> {
    let exe = std::env::current_exe()?;
    let exe = exe.canonicalize().unwrap_or(exe);
    let dir = exe
        .parent()
        .ok_or_else(|| UpdateError::Refused("no install directory".into()))?
        .to_path_buf();
    let name = exe
        .file_name()
        .ok_or_else(|| UpdateError::Refused("no executable name".into()))?
        .to_string_lossy()
        .into_owned();
    install_into(rel, &dir, &name, progress)
}

/// Remove files renamed aside by a previous update (best effort).
pub fn cleanup() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(dir) = exe.parent() else { return };
    for f in files_under(dir) {
        if f.to_string_lossy().ends_with(OLD_SUFFIX) {
            let _ = fs::remove_file(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_order() {
        assert!(newer("0.6.0", "0.5.9"));
        assert!(newer("v1.0.0", "0.99.0"));
        assert!(newer("0.5.1", "0.5"));
        assert!(!newer("0.5.0", "0.5.0"));
        assert!(!newer("0.4.9", "0.5.0"));
        assert!(newer("0.5.0", "0.5.0-rc1"));
        assert!(!newer("0.5.0-rc2", "0.5.0"));
        assert!(!newer("garbage", "0.1.0"));
    }

    #[test]
    fn sums_lookup() {
        let s = "abc123  reco-anonymize-0.5.0-linux-x86_64.tar.gz\nDEF *x.zip\n";
        assert_eq!(expected_sha(s, "x.zip").as_deref(), Some("def"));
        assert_eq!(
            expected_sha(s, "reco-anonymize-0.5.0-linux-x86_64.tar.gz").as_deref(),
            Some("abc123")
        );
        assert!(expected_sha(s, "other").is_none());
    }

    #[test]
    fn select_skips_prerelease_and_missing_platform() {
        let rel = |pre: bool, name: &str| GhRelease {
            tag_name: "v0.6.0".into(),
            html_url: "u".into(),
            draft: false,
            prerelease: pre,
            assets: vec![GhAsset {
                name: name.into(),
                browser_download_url: "d".into(),
                size: 1,
            }],
        };
        assert!(select(rel(true, "a-linux-x86_64.tar.gz"), "linux-x86_64.tar.gz").is_none());
        assert!(select(rel(false, "a-windows-x86_64.zip"), "linux-x86_64.tar.gz").is_none());
        let r = select(rel(false, "a-linux-x86_64.tar.gz"), "linux-x86_64.tar.gz").unwrap();
        assert_eq!(r.version, "0.6.0");
        assert!(r.sums.is_none());
    }
}
