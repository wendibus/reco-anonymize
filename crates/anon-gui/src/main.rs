//! `reco-anon-review` — review app for reco-anonymize.
//!
//! The UI is a thin shell over `anon_review::ReviewSession`; long jobs
//! (analyze, preview, render, audit) run on worker threads and report back
//! through a channel drained by a UI-thread timer.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod jobs;

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use anon_model::BBox;
use anon_model::frames::flag;
use anon_review::{RegionRef, ReviewSession, Rgba, ViewMode};
use slint::{Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, SharedString, VecModel};

use jobs::{Job, JobMsg};

slint::include_modules!();

pub struct App {
    session: Option<ReviewSession>,
    model: Option<PathBuf>,
    jobs: mpsc::Sender<JobMsg>,
    job_running: bool,
    timeline_dirty: bool,
}

type Shared = Rc<RefCell<App>>;

fn to_image(img: &Rgba) -> Image {
    let buf = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&img.data, img.width, img.height);
    Image::from_rgba8(buf)
}

/// Per-frame strip: red error, orange flagged, green reviewed, grey people.
fn timeline(s: &ReviewSession, width: usize) -> Rgba {
    let n = s.frame_count().max(1) as usize;
    let w = width.min(n).max(1);
    let mut data = Vec::with_capacity(w * 4);
    for b in 0..w {
        let (a, z) = (b * n / w, ((b + 1) * n / w).max(b * n / w + 1));
        let (mut err, mut flagged, mut reviewed, mut people) = (false, false, true, 0u16);
        for f in a..z {
            if let Some(r) = s.frames.get(f as u64) {
                err |= r.flags & flag::ERROR != 0;
                if r.flags != 0 {
                    flagged = true;
                    reviewed &= s.is_reviewed(f as u64);
                }
                people = people.max(r.persons);
            }
        }
        let c = if err {
            [227, 60, 60, 255]
        } else if flagged && !reviewed {
            [230, 160, 40, 255]
        } else if flagged {
            [80, 170, 90, 255]
        } else {
            let g = if people > 0 { 90 } else { 40 };
            [g, g, g, 255]
        };
        data.extend_from_slice(&c);
    }
    Rgba {
        width: w as u32,
        height: 1,
        data,
    }
}

fn message(ui: &AppWindow, text: impl Into<SharedString>, error: bool) {
    ui.set_message(text.into());
    ui.set_message_error(error);
}

/// Push the whole session state into the UI.
fn refresh(ui: &AppWindow, app: &mut App) {
    ui.set_job_running(app.job_running);
    ui.set_model_path(
        app.model
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
            .into(),
    );
    let Some(s) = app.session.as_mut() else {
        ui.set_has_project(false);
        return;
    };
    ui.set_has_project(true);
    ui.set_project_name(
        s.project
            .source
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
            .into(),
    );
    ui.set_frame_count(s.frame_count().max(1) as i32);
    ui.set_current_frame(s.current as i32);
    ui.set_opaque_mode(s.mode == ViewMode::Opaque);
    ui.set_dirty(s.dirty);

    match s.render_current() {
        Ok(img) => ui.set_frame_image(to_image(&img)),
        Err(e) => message(ui, format!("cannot show frame: {e}"), true),
    }
    let rec = s.frames.get(s.current).copied();
    ui.set_frame_info(
        format!(
            "Frame {} / {}{}",
            s.current,
            s.frame_count().saturating_sub(1),
            rec.map_or(String::new(), |r| format!(" · {} people", r.persons))
        )
        .into(),
    );
    let flags = s.flags_at(s.current);
    ui.set_flag_info(if flags.is_empty() {
        "no flags".into()
    } else {
        format!("flagged: {}", flags.join(", ")).into()
    });

    let r = s.readiness();
    ui.set_review_progress(
        format!(
            "{} flagged frame(s), {} reviewed, {} to go",
            r.flagged,
            r.flagged - r.unreviewed,
            r.unreviewed
        )
        .into(),
    );
    let mut lines = Vec::new();
    if r.errors > 0 {
        lines.push(format!("{} frame(s) in error: re-analyze first", r.errors));
    }
    if r.unreviewed > 0 {
        lines.push(format!("{} flagged frame(s) still to review", r.unreviewed));
    }
    if r.nobody_detected {
        lines.push("nobody detected in the whole clip: check the model".into());
    }
    if lines.is_empty() {
        lines.push(if r.approval_valid {
            "approved — Render is unlocked".into()
        } else {
            "ready to approve".into()
        });
    }
    ui.set_readiness(lines.join("\n").into());
    ui.set_can_approve(r.can_approve());
    ui.set_approved(r.approval_valid);
    let export = match anon_pipeline::export_clean(&s.dir) {
        Ok((true, note)) => format!("export: CLEAN — {note}"),
        Ok((false, note)) => format!("export: not clean — {note}"),
        Err(e) => format!("export: {e}"),
    };
    ui.set_export_state(export.into());

    // masks in this frame
    let regions = s.regions_at(s.current);
    let mut rows = Vec::new();
    for reg in &regions {
        if let RegionRef::Track {
            id,
            manual,
            suppressed,
            interpolated,
            score,
        } = reg.what
        {
            let mut label = format!("#{id} {}", if manual { "manual" } else { "auto" });
            if !manual && !interpolated {
                label += &format!(" · score {score:.2}");
            }
            if interpolated {
                label += " · interpolated";
            }
            if suppressed {
                label += " · SUPPRESSED";
            }
            rows.push(TrackInfo {
                id: id as i32,
                label: label.into(),
                manual,
                suppressed,
                selected: s.selected == Some(id),
            });
        }
    }
    ui.set_tracks(ModelRc::new(VecModel::from(rows)));
    match s
        .selected
        .and_then(|id| s.tracks.tracks.iter().find(|t| t.id == id))
    {
        Some(t) => {
            let first = t.keyframes.first().map_or(0, |k| k.frame);
            let last = t.keyframes.last().map_or(0, |k| k.frame);
            let mut info = format!(
                "track #{} · {} · frames {first}–{last} · {} keyframe(s)",
                t.id,
                if t.is_manual() { "manual" } else { "auto" },
                t.keyframes.len()
            );
            if let Some(sup) = &t.suppressed {
                info += &format!("\nsuppressed: {}", sup.reason);
            }
            ui.set_selected_info(info.into());
            ui.set_selected_manual(t.is_manual());
            ui.set_selected_suppressed(t.suppressed.is_some());
            ui.set_has_selection(true);
        }
        None => ui.set_has_selection(false),
    }
    if app.timeline_dirty {
        ui.set_timeline_image(to_image(&timeline(s, 2000)));
        app.timeline_dirty = false;
    }
}

/// View coordinates (image-fit: contain) to source pixels.
fn to_source(s: &ReviewSession, x: f32, y: f32, w: f32, h: f32) -> (f32, f32) {
    let (iw, ih) = s.size();
    let scale = (w / iw as f32).min(h / ih as f32);
    let ox = (w - iw as f32 * scale) / 2.0;
    let oy = (h - ih as f32 * scale) / 2.0;
    ((x - ox) / scale, (y - oy) / scale)
}

fn open_session(ui: &AppWindow, app: &Shared, path: &Path) {
    let mut a = app.borrow_mut();
    match ReviewSession::open(path) {
        Ok(s) => {
            a.session = Some(s);
            a.timeline_dirty = true;
            message(ui, format!("opened {}", path.display()), false);
        }
        Err(e) => {
            // a video without project: offer analysis
            a.session = None;
            if path.is_file() {
                message(
                    ui,
                    format!(
                        "{} has no analysis yet ({e}). Choose a model and press Analyze.",
                        path.display()
                    ),
                    true,
                );
                ui.set_has_project(true);
                ui.set_project_name(path.display().to_string().into());
                PENDING.with(|p| *p.borrow_mut() = Some(path.to_path_buf()));
                refresh_buttons_only(ui, &a);
                return;
            }
            message(ui, format!("cannot open {}: {e}", path.display()), true);
        }
    }
    refresh(ui, &mut a);
}

fn refresh_buttons_only(ui: &AppWindow, a: &App) {
    ui.set_job_running(a.job_running);
    ui.set_model_path(
        a.model
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
            .into(),
    );
}

thread_local! {
    /// A video chosen for analysis that has no project yet.
    static PENDING: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

fn source_path(a: &App) -> Option<PathBuf> {
    a.session
        .as_ref()
        .map(|s| s.project.source.path.clone())
        .or_else(|| PENDING.with(|p| p.borrow().clone()))
}

fn edit(
    ui: &AppWindow,
    app: &Shared,
    f: impl FnOnce(&mut ReviewSession) -> Result<(), anon_review::ReviewError>,
) {
    let mut a = app.borrow_mut();
    let Some(s) = a.session.as_mut() else { return };
    match f(s).and_then(|_| s.save()) {
        Ok(()) => {}
        Err(e) => message(ui, e.to_string(), true),
    }
    a.timeline_dirty = true;
    refresh(ui, &mut a);
}

fn start_job(ui: &AppWindow, app: &Shared, job: Job) {
    let mut a = app.borrow_mut();
    if a.job_running {
        return;
    }
    if let Some(s) = a.session.as_mut()
        && let Err(e) = s.save()
    {
        message(ui, format!("save failed: {e}"), true);
        return;
    }
    a.job_running = true;
    ui.set_job_progress(-1.0);
    ui.set_job_text(job.title().into());
    jobs::spawn(job, a.jobs.clone());
    refresh_buttons_only(ui, &a);
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let ui = AppWindow::new()?;
    let (tx, rx) = mpsc::channel::<JobMsg>();
    let app: Shared = Rc::new(RefCell::new(App {
        session: None,
        model: jobs::last_model(),
        jobs: tx,
        job_running: false,
        timeline_dirty: true,
    }));

    if let Some(p) = std::env::args_os().nth(1) {
        open_session(&ui, &app, Path::new(&p));
    } else {
        refresh(&ui, &mut app.borrow_mut());
    }

    // ---- navigation ------------------------------------------------------
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_goto(move |f| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            s.goto(f.max(0) as u64);
            app.timeline_dirty = true;
        }
        refresh(&ui, &mut app);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_step(move |d| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            let f = (s.current as i64 + i64::from(d)).max(0) as u64;
            s.goto(f);
            app.timeline_dirty = true;
        }
        refresh(&ui, &mut app);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_next_flag(move |forward| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            match s
                .next_flag(forward, forward)
                .or_else(|| s.next_flag(forward, false))
            {
                Some(f) => s.goto(f),
                None => message(&ui, "no flagged frames", false),
            }
            app.timeline_dirty = true;
        }
        refresh(&ui, &mut app);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_set_opaque(move |on| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            s.mode = if on {
                ViewMode::Opaque
            } else {
                ViewMode::Outline
            };
        }
        refresh(&ui, &mut app);
    });

    // ---- selection and edits ---------------------------------------------
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_select_at(move |x, y, vw, vh| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            let (sx, sy) = to_source(s, x, y, vw, vh);
            s.selected = s.hit_test(sx, sy);
        }
        refresh(&ui, &mut app);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_add_box(move |x0, y0, x1, y1, vw, vh| {
        let ui = w.unwrap();
        edit(&ui, &a, |s| {
            let (ax, ay) = to_source(s, x0, y0, vw, vh);
            let (bx, by) = to_source(s, x1, y1, vw, vh);
            s.add_manual_box(BBox::new(ax, ay, bx, by)).map(|_| ())
        });
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_select_track(move |id| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            s.selected = Some(id as u64);
        }
        refresh(&ui, &mut app);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_deselect(move || {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            s.selected = None;
        }
        refresh(&ui, &mut app);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_suppress(move |reason| {
        let ui = w.unwrap();
        edit(&ui, &a, |s| match s.selected {
            Some(id) => s.suppress(id, &reason),
            None => Ok(()),
        });
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_restore(move || {
        let ui = w.unwrap();
        edit(&ui, &a, |s| match s.selected {
            Some(id) => s.restore(id),
            None => Ok(()),
        });
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_delete_keyframe(move || {
        let ui = w.unwrap();
        edit(&ui, &a, |s| match s.selected {
            Some(id) => s.delete_manual_keyframe(id),
            None => Ok(()),
        });
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_save(move || {
        let ui = w.unwrap();
        edit(&ui, &a, |_| Ok(()));
        message(&ui, "saved", false);
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_approve(move |note, nobody| {
        let ui = w.unwrap();
        let mut app = a.borrow_mut();
        if let Some(s) = app.session.as_mut() {
            let note = (!note.is_empty()).then(|| note.to_string());
            match s.approve(note, nobody) {
                Ok(h) => message(&ui, format!("approved ({})", &h[..12]), false),
                Err(e) => message(&ui, e.to_string(), true),
            }
        }
        refresh(&ui, &mut app);
    });

    // ---- files and jobs ----------------------------------------------------
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_open_project(move || {
        let ui = w.unwrap();
        let picked = rfd::FileDialog::new()
            .set_title("Open a video or its .anon project")
            .add_filter("Video", &["mp4", "mov", "mkv", "m4v", "avi", "mts"])
            .add_filter("Project", &["json"])
            .pick_file();
        if let Some(p) = picked {
            // project.json inside a .anon folder -> the folder
            let p = if p.file_name().is_some_and(|n| n == "project.json") {
                p.parent().map(Path::to_path_buf).unwrap_or(p)
            } else {
                p
            };
            open_session(&ui, &a, &p);
        }
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_choose_model(move || {
        let ui = w.unwrap();
        if let Some(p) = rfd::FileDialog::new()
            .set_title("RF-DETR-Seg ONNX model (with .onnx.json manifest)")
            .add_filter("ONNX", &["onnx"])
            .pick_file()
        {
            jobs::remember_model(&p);
            let mut app = a.borrow_mut();
            app.model = Some(p);
            refresh_buttons_only(&ui, &app);
        }
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_run_analyze(move || {
        let ui = w.unwrap();
        let (src, model, resume) = {
            let app = a.borrow();
            (source_path(&app), app.model.clone(), false)
        };
        if let (Some(src), Some(model)) = (src, model) {
            start_job(&ui, &a, Job::Analyze { src, model, resume });
        }
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_run_preview(move || {
        let ui = w.unwrap();
        let dir = a.borrow().session.as_ref().map(|s| s.dir.clone());
        if let Some(dir) = dir {
            start_job(&ui, &a, Job::Preview { dir });
        }
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_run_render(move || {
        let ui = w.unwrap();
        let (dir, src, model) = {
            let app = a.borrow();
            let Some(s) = app.session.as_ref() else {
                return;
            };
            (
                s.dir.clone(),
                s.project.source.path.clone(),
                app.model.clone(),
            )
        };
        let stem = src
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let out = rfd::FileDialog::new()
            .set_title("Save anonymized video")
            .set_file_name(format!("{stem}-anon.mp4"))
            .set_directory(src.parent().unwrap_or(Path::new(".")))
            .save_file();
        if let Some(out) = out {
            if out == src {
                message(&ui, "refusing to overwrite the source", true);
                return;
            }
            start_job(&ui, &a, Job::Render { dir, out, model });
        }
    });
    let (w, a) = (ui.as_weak(), app.clone());
    ui.on_run_audit(move || {
        let ui = w.unwrap();
        let (dir, model) = {
            let app = a.borrow();
            let Some(s) = app.session.as_ref() else {
                return;
            };
            (s.dir.clone(), app.model.clone())
        };
        let Some(model) = model else { return };
        let last = jobs::last_render(&dir);
        let out = rfd::FileDialog::new()
            .set_title("Rendered video to audit")
            .add_filter("Video", &["mp4", "mov", "mkv", "m4v"]);
        let out = match &last {
            Some(p) => out
                .set_directory(p.parent().unwrap_or(Path::new(".")))
                .set_file_name(
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                ),
            None => out,
        }
        .pick_file();
        if let Some(out) = out {
            start_job(&ui, &a, Job::Audit { dir, out, model });
        }
    });

    // ---- job messages --------------------------------------------------------
    let timer = slint::Timer::default();
    let (w, a) = (ui.as_weak(), app.clone());
    timer.start(
        slint::TimerMode::Repeated,
        Duration::from_millis(100),
        move || {
            let Some(ui) = w.upgrade() else { return };
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    JobMsg::Progress { fraction, text } => {
                        ui.set_job_progress(fraction);
                        ui.set_job_text(text.into());
                    }
                    JobMsg::Done { text, ok, reload } => {
                        let mut app = a.borrow_mut();
                        app.job_running = false;
                        ui.set_job_text(text.clone().into());
                        message(&ui, text, !ok);
                        if let Some(path) = reload {
                            drop(app);
                            let cur = a.borrow().session.as_ref().map(|s| s.current);
                            open_session(&ui, &a, &path);
                            let mut app = a.borrow_mut();
                            if let (Some(s), Some(c)) = (app.session.as_mut(), cur) {
                                s.goto(c);
                            }
                            app.timeline_dirty = true;
                            refresh(&ui, &mut app);
                            PENDING.with(|p| *p.borrow_mut() = None);
                        } else {
                            refresh(&ui, &mut app);
                        }
                    }
                }
            }
        },
    );

    let a = app.clone();
    ui.window().on_close_requested(move || {
        if let Some(s) = a.borrow_mut().session.as_mut() {
            let _ = s.save();
        }
        slint::CloseRequestResponse::HideWindow
    });

    ui.run()?;
    Ok(())
}
