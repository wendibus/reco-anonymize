//! Runtime strings built in Rust (the .slint strings use `@tr` and the
//! bundled .po files in `lang/`). Keyed by the English text; `{}` are
//! filled in order.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En = 0,
    De = 1,
    Fr = 2,
    Es = 3,
}

static CURRENT: AtomicU8 = AtomicU8::new(0);

impl Lang {
    pub const ALL: [Lang; 4] = [Lang::En, Lang::De, Lang::Fr, Lang::Es];

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::De => "de",
            Lang::Fr => "fr",
            Lang::Es => "es",
        }
    }

    pub fn from_code(c: &str) -> Option<Lang> {
        let c = c.to_ascii_lowercase();
        Lang::ALL.into_iter().find(|l| c.starts_with(l.code()))
    }

    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn from_index(i: i32) -> Lang {
        Lang::ALL
            .get(usize::try_from(i).unwrap_or(0))
            .copied()
            .unwrap_or(Lang::En)
    }

    /// System language if supported, else English.
    pub fn detect() -> Lang {
        sys_locale::get_locale()
            .and_then(|l| Lang::from_code(&l))
            .unwrap_or(Lang::En)
    }
}

pub fn set(l: Lang) {
    CURRENT.store(l as u8, Ordering::Relaxed);
    if let Err(e) = slint::select_bundled_translation(l.code()) {
        log::warn!("translation {}: {e:?}", l.code());
    }
}

pub fn current() -> Lang {
    Lang::from_index(i32::from(CURRENT.load(Ordering::Relaxed)))
}

/// (en, de, fr, es)
const TABLE: &[(&str, &str, &str, &str)] = &[
    (
        "cannot show frame: {}",
        "Frame kann nicht angezeigt werden: {}",
        "impossible d’afficher l’image : {}",
        "no se puede mostrar el fotograma: {}",
    ),
    (
        "Frame {} / {}",
        "Frame {} / {}",
        "Image {} / {}",
        "Fotograma {} / {}",
    ),
    (
        " · {} people",
        " · {} Personen",
        " · {} personnes",
        " · {} personas",
    ),
    (
        "no flags",
        "keine Markierungen",
        "aucun signalement",
        "sin marcas",
    ),
    ("flagged: {}", "markiert: {}", "signalé : {}", "marcado: {}"),
    (
        "low score",
        "niedrige Konfidenz",
        "score faible",
        "puntuación baja",
    ),
    (
        "count drop",
        "Personenzahl sinkt",
        "baisse du nombre",
        "caída del recuento",
    ),
    (
        "track starts/ends in view",
        "Spur beginnt/endet im Bild",
        "piste commence/finit dans l’image",
        "pista empieza/termina en imagen",
    ),
    (
        "long interpolation",
        "lange Interpolation",
        "longue interpolation",
        "interpolación larga",
    ),
    ("error", "Fehler", "erreur", "error"),
    (
        "{} flagged frame(s), {} reviewed, {} to go",
        "{} markierte Frames, {} geprüft, {} offen",
        "{} images signalées, {} vérifiées, {} restantes",
        "{} fotogramas marcados, {} revisados, {} pendientes",
    ),
    (
        "{} frame(s) in error: re-analyze first",
        "{} Frames mit Fehler: zuerst neu analysieren",
        "{} images en erreur : relancez d’abord l’analyse",
        "{} fotogramas con error: vuelve a analizar primero",
    ),
    (
        "{} flagged frame(s) still to review",
        "noch {} markierte Frames zu prüfen",
        "encore {} images signalées à vérifier",
        "quedan {} fotogramas marcados por revisar",
    ),
    (
        "nobody detected in the whole clip: check the model",
        "im ganzen Clip niemand erkannt: Modell prüfen",
        "personne détecté dans tout le clip : vérifiez le modèle",
        "no se detectó a nadie en todo el clip: revisa el modelo",
    ),
    (
        "approved — Render is unlocked",
        "freigegeben — Rendern ist möglich",
        "validé — le rendu est débloqué",
        "aprobado — el renderizado está desbloqueado",
    ),
    (
        "ready to approve",
        "bereit zur Freigabe",
        "prêt à valider",
        "listo para aprobar",
    ),
    (
        "export: CLEAN — {}",
        "Export: SAUBER — {}",
        "export : PROPRE — {}",
        "exportación: LIMPIA — {}",
    ),
    (
        "export: not clean — {}",
        "Export: nicht sauber — {}",
        "export : pas propre — {}",
        "exportación: no limpia — {}",
    ),
    (
        "not audited",
        "nicht geprüft",
        "non contrôlé",
        "sin auditar",
    ),
    (
        "output changed or missing since audit",
        "Ausgabe seit der Prüfung verändert oder fehlt",
        "sortie modifiée ou absente depuis le contrôle",
        "salida modificada o ausente desde la auditoría",
    ),
    (
        "audit passed",
        "Prüfung bestanden",
        "contrôle réussi",
        "auditoría superada",
    ),
    (
        "audit passed, {} hit(s) acknowledged",
        "Prüfung bestanden, {} Treffer bestätigt",
        "contrôle réussi, {} alerte(s) acceptée(s)",
        "auditoría superada, {} aviso(s) aceptado(s)",
    ),
    (
        "{} open audit hit(s)",
        "{} offene Prüftreffer",
        "{} alerte(s) de contrôle ouverte(s)",
        "{} aviso(s) de auditoría abiertos",
    ),
    ("manual", "manuell", "manuel", "manual"),
    ("auto", "auto", "auto", "auto"),
    (
        " · score {}",
        " · Konfidenz {}",
        " · score {}",
        " · puntuación {}",
    ),
    (
        " · interpolated",
        " · interpoliert",
        " · interpolé",
        " · interpolado",
    ),
    (
        " · SUPPRESSED",
        " · UNTERDRÜCKT",
        " · SUPPRIMÉ",
        " · SUPRIMIDO",
    ),
    (
        "track #{} · {} · frames {}–{} · {} keyframe(s)",
        "Spur #{} · {} · Frames {}–{} · {} Keyframes",
        "piste #{} · {} · images {}–{} · {} image(s) clé(s)",
        "pista #{} · {} · fotogramas {}–{} · {} fotograma(s) clave",
    ),
    (
        "suppressed: {}",
        "unterdrückt: {}",
        "supprimé : {}",
        "suprimido: {}",
    ),
    (
        "{} · frames {}–{}",
        "{} · Frames {}–{}",
        "{} · images {}–{}",
        "{} · fotogramas {}–{}",
    ),
    (
        "{} · whole clip",
        "{} · ganzer Clip",
        "{} · tout le clip",
        "{} · todo el clip",
    ),
    ("opened {}", "{} geöffnet", "{} ouvert", "{} abierto"),
    (
        "{} has no analysis yet. Choose a model and press Analyze.",
        "{} ist noch nicht analysiert. Modell wählen und Analysieren drücken.",
        "{} n’est pas encore analysé. Choisissez un modèle et cliquez sur Analyser.",
        "{} aún no está analizado. Elige un modelo y pulsa Analizar.",
    ),
    (
        "cannot open {}: {}",
        "{} kann nicht geöffnet werden: {}",
        "impossible d’ouvrir {} : {}",
        "no se puede abrir {}: {}",
    ),
    (
        "save failed: {}",
        "Speichern fehlgeschlagen: {}",
        "échec de l’enregistrement : {}",
        "error al guardar: {}",
    ),
    ("saved", "gespeichert", "enregistré", "guardado"),
    (
        "approved ({})",
        "freigegeben ({})",
        "validé ({})",
        "aprobado ({})",
    ),
    (
        "no flagged frames",
        "keine markierten Frames",
        "aucune image signalée",
        "no hay fotogramas marcados",
    ),
    (
        "frame range: enter two frame numbers",
        "Frame-Bereich: zwei Frame-Nummern eingeben",
        "plage d’images : saisissez deux numéros",
        "rango: introduce dos números de fotograma",
    ),
    (
        "refusing to overwrite the source",
        "Quelle wird nicht überschrieben",
        "refus d’écraser la source",
        "no se sobrescribe el original",
    ),
    ("analyzing…", "analysiere…", "analyse…", "analizando…"),
    (
        "rendering preview…",
        "rendere Vorschau…",
        "rendu de l’aperçu…",
        "renderizando vista previa…",
    ),
    ("rendering…", "rendere…", "rendu…", "renderizando…"),
    ("auditing…", "prüfe…", "contrôle…", "auditando…"),
    (
        "auditing the export…",
        "prüfe den Export…",
        "contrôle de l’export…",
        "auditando la exportación…",
    ),
    (
        "exporting training data…",
        "exportiere Trainingsdaten…",
        "export des données d’entraînement…",
        "exportando datos de entrenamiento…",
    ),
    (
        "downloading update…",
        "lade Update…",
        "téléchargement de la mise à jour…",
        "descargando actualización…",
    ),
    (
        "{}: frame {} / {}",
        "{}: Frame {} / {}",
        "{} : image {} / {}",
        "{}: fotograma {} / {}",
    ),
    (
        "{}: frame {}",
        "{}: Frame {}",
        "{} : image {}",
        "{}: fotograma {}",
    ),
    ("s/frame", "s/Frame", "s/image", "s/fotograma"),
    ("{} left", "noch {}", "encore {}", "quedan {}"),
    ("analyze", "Analyse", "analyse", "análisis"),
    ("audit", "Prüfung", "contrôle", "auditoría"),
    ("export", "Export", "export", "exportación"),
    (
        "analysis done: {} frames, {} head tracks, {} text tracks, {} flagged, {} errors",
        "Analyse fertig: {} Frames, {} Kopfspuren, {} Textspuren, {} markiert, {} Fehler",
        "analyse terminée : {} images, {} pistes de tête, {} pistes de texte, {} signalées, {} erreurs",
        "análisis terminado: {} fotogramas, {} pistas de cabeza, {} pistas de texto, {} marcados, {} errores",
    ),
    (
        "preview written: {}",
        "Vorschau geschrieben: {}",
        "aperçu écrit : {}",
        "vista previa guardada: {}",
    ),
    (
        "rendered {} frames via {}, timing verified",
        "{} Frames mit {} gerendert, Timing geprüft",
        "{} images rendues via {}, timing vérifié",
        "{} fotogramas renderizados con {}, tiempos verificados",
    ),
    (
        "audit PASSED: {} frames, no uncovered head",
        "Prüfung BESTANDEN: {} Frames, kein unverdeckter Kopf",
        "contrôle RÉUSSI : {} images, aucune tête visible",
        "auditoría SUPERADA: {} fotogramas, ninguna cabeza visible",
    ),
    (
        "audit FAILED: {} uncovered head(s) — see audit-report.json, fix masks and render again",
        "Prüfung FEHLGESCHLAGEN: {} unverdeckte Köpfe — siehe audit-report.json, Masken korrigieren und neu rendern",
        "contrôle ÉCHOUÉ : {} tête(s) visible(s) — voir audit-report.json, corrigez les masques et refaites le rendu",
        "auditoría FALLIDA: {} cabeza(s) visible(s) — ver audit-report.json, corrige las máscaras y vuelve a renderizar",
    ),
    (
        "training data: {} images, {} heads → {} (UN-anonymized, keep it local)",
        "Trainingsdaten: {} Bilder, {} Köpfe → {} (NICHT anonymisiert, nur lokal verwenden)",
        "données d’entraînement : {} images, {} têtes → {} (NON anonymisées, à garder en local)",
        "datos de entrenamiento: {} imágenes, {} cabezas → {} (SIN anonimizar, mantener en local)",
    ),
    (
        "updated to {} — restart the app to finish",
        "auf {} aktualisiert — App neu starten zum Abschließen",
        "mis à jour vers {} — redémarrez l’application",
        "actualizado a {} — reinicia la aplicación para terminar",
    ),
    (
        "Open a video or its .anon project",
        "Video oder dessen .anon-Projekt öffnen",
        "Ouvrir une vidéo ou son projet .anon",
        "Abrir un vídeo o su proyecto .anon",
    ),
    (
        "Person model (RF-DETR-Seg ONNX with manifest)",
        "Personenmodell (RF-DETR-Seg ONNX mit Manifest)",
        "Modèle de personnes (RF-DETR-Seg ONNX avec manifeste)",
        "Modelo de personas (RF-DETR-Seg ONNX con manifiesto)",
    ),
    (
        "Text model (PP-OCR det ONNX with manifest)",
        "Textmodell (PP-OCR det ONNX mit Manifest)",
        "Modèle de texte (PP-OCR det ONNX avec manifeste)",
        "Modelo de texto (PP-OCR det ONNX con manifiesto)",
    ),
    (
        "Save anonymized video",
        "Anonymisiertes Video speichern",
        "Enregistrer la vidéo anonymisée",
        "Guardar vídeo anonimizado",
    ),
    (
        "Rendered video to audit",
        "Zu prüfendes gerendertes Video",
        "Vidéo rendue à contrôler",
        "Vídeo renderizado a auditar",
    ),
    (
        "Folder for the training data",
        "Ordner für die Trainingsdaten",
        "Dossier pour les données d’entraînement",
        "Carpeta para los datos de entrenamiento",
    ),
];

/// Translate an English UI string.
pub fn tr(en: &'static str) -> &'static str {
    let l = current();
    if l == Lang::En {
        return en;
    }
    TABLE
        .iter()
        .find(|row| row.0 == en)
        .map_or(en, |row| match l {
            Lang::De => row.1,
            Lang::Fr => row.2,
            Lang::Es => row.3,
            Lang::En => row.0,
        })
}

/// Translate and fill `{}` placeholders in order.
pub fn trf(en: &'static str, args: &[&dyn std::fmt::Display]) -> String {
    let pattern = tr(en);
    let mut out = String::with_capacity(pattern.len() + 16);
    let mut it = args.iter();
    let mut rest = pattern;
    while let Some(i) = rest.find("{}") {
        out.push_str(&rest[..i]);
        if let Some(a) = it.next() {
            out.push_str(&a.to_string());
        }
        rest = &rest[i + 2..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_keeps_its_placeholders() {
        for row in TABLE {
            let n = row.0.matches("{}").count();
            for t in [row.1, row.2, row.3] {
                assert_eq!(t.matches("{}").count(), n, "{:?}", row.0);
            }
        }
    }

    #[test]
    fn fills_and_falls_back() {
        CURRENT.store(Lang::De as u8, Ordering::Relaxed);
        assert_eq!(trf("Frame {} / {}", &[&3, &9]), "Frame 3 / 9");
        assert_eq!(trf("opened {}", &[&"x.mp4"]), "x.mp4 geöffnet");
        assert_eq!(tr("not in table"), "not in table");
        CURRENT.store(Lang::En as u8, Ordering::Relaxed);
        assert_eq!(Lang::from_code("de_DE.UTF-8"), Some(Lang::De));
        assert_eq!(Lang::from_code("pt-BR"), None);
    }
}
