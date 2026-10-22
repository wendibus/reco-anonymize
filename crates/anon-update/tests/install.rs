//! Update check + install against a local fake GitHub (plain HTTP).

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

type Routes = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

fn serve(routes: Routes) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).unwrap() <= 2 {
                    break;
                }
            }
            let body = routes
                .lock()
                .unwrap()
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, b)| b.clone());
            let mut s = s;
            match body {
                Some(b) => {
                    write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        b.len()
                    )
                    .unwrap();
                    s.write_all(&b).unwrap();
                }
                None => {
                    write!(
                        s,
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .unwrap();
                }
            }
        }
    });
    port
}

fn tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    {
        let mut t = tar::Builder::new(&mut gz);
        for (name, data) in files {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o755);
            h.set_cksum();
            t.append_data(&mut h, name, *data).unwrap();
        }
        t.finish().unwrap();
    }
    gz.finish().unwrap()
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

#[test]
fn checks_verifies_and_installs() {
    let routes: Routes = Arc::new(Mutex::new(Vec::new()));
    let port = serve(routes.clone());
    let base = format!("http://127.0.0.1:{port}");
    // SAFETY: single-threaded test setup before any other code reads it
    unsafe { std::env::set_var("RECO_ANON_UPDATE_API", &base) };

    let name = "reco-anonymize-0.6.0-linux-x86_64.tar.gz";
    let archive = tar_gz(&[
        ("reco-anonymize-0.6.0/reco-anon-review", b"new gui"),
        ("reco-anonymize-0.6.0/reco-anon", b"new cli"),
        ("reco-anonymize-0.6.0/lib/libx.so", b"same lib"),
        ("reco-anonymize-0.6.0/README.md", b"new readme"),
    ]);
    let sha = hex::encode(Sha256::digest(&archive));
    let release = serde_json::json!({
        "tag_name": "v0.6.0",
        "html_url": "https://example/release",
        "assets": [
            {"name": name, "browser_download_url": format!("{base}/dl/a"), "size": archive.len()},
            {"name": "SHA256SUMS.txt", "browser_download_url": format!("{base}/dl/sums"), "size": 1},
        ]
    });
    {
        let mut r = routes.lock().unwrap();
        r.push((
            "/repos/wendibus/reco-anonymize/releases/latest".into(),
            serde_json::to_vec(&release).unwrap(),
        ));
        r.push(("/dl/a".into(), archive.clone()));
        r.push(("/dl/sums".into(), format!("{sha}  {name}\n").into_bytes()));
    }

    // up to date / older
    assert!(
        anon_update::check_for("0.6.0", "linux-x86_64.tar.gz")
            .unwrap()
            .is_none()
    );
    assert!(
        anon_update::check_for("0.5.0", "windows-x86_64.zip")
            .unwrap()
            .is_none()
    );
    let rel = anon_update::check_for("0.5.0", "linux-x86_64.tar.gz")
        .unwrap()
        .expect("update");
    assert_eq!(rel.version, "0.6.0");

    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    std::fs::write(d.join("reco-anon-review"), b"old gui").unwrap();
    std::fs::write(d.join("reco-anon"), b"old cli").unwrap();
    std::fs::create_dir_all(d.join("lib")).unwrap();
    std::fs::write(d.join("lib/libx.so"), b"same lib").unwrap();

    // tampered checksum -> refused, nothing changed
    routes.lock().unwrap()[2].1 = format!("{}  {name}\n", "0".repeat(64)).into_bytes();
    let err = anon_update::install_into(&rel, d, "reco-anon-review", |_, _| {}).unwrap_err();
    assert!(err.to_string().contains("checksum mismatch"), "{err}");
    assert_eq!(read(&d.join("reco-anon-review")), b"old gui");

    // not listed -> refused
    routes.lock().unwrap()[2].1 = b"abc  other.tar.gz\n".to_vec();
    assert!(anon_update::install_into(&rel, d, "reco-anon-review", |_, _| {}).is_err());

    // correct -> installed
    routes.lock().unwrap()[2].1 = format!("{sha}  {name}\n").into_bytes();
    let mut last = 0;
    let rep =
        anon_update::install_into(&rel, d, "reco-anon-review", |done, _| last = done).unwrap();
    assert_eq!(last, archive.len() as u64);
    assert_eq!(rep.files, 3, "unchanged lib is skipped");
    assert_eq!(read(&d.join("reco-anon-review")), b"new gui");
    assert_eq!(read(&d.join("reco-anon")), b"new cli");
    assert_eq!(read(&d.join("README.md")), b"new readme");
    assert_eq!(read(&d.join("reco-anon-review.old-update")), b"old gui");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(d.join("reco-anon"))
            .unwrap()
            .permissions()
            .mode();
        assert!(mode & 0o111 != 0, "executable bit kept");
    }
}
