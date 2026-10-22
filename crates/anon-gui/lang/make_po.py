#!/usr/bin/env python3
"""Generates the .po files for the review app's Slint strings.
Edit the table below, then run: python3 crates/anon-gui/lang/make_po.py"""
import os

T = {
 "Analyze": ("Analysieren", "Analyser", "Analizar"),
 "App": ("App", "Application", "Aplicación"),
 "Approval": ("Freigabe", "Validation", "Aprobación"),
 "Approve": ("Freigeben", "Valider", "Aprobar"),
 "Approved": ("Freigegeben", "Validé", "Aprobado"),
 "Audit…": ("Prüfen…", "Contrôler…", "Auditar…"),
 "Cancel": ("Abbrechen", "Annuler", "Cancelar"),
 "Check for updates at start": ("Beim Start nach Updates suchen", "Rechercher des mises à jour au démarrage", "Buscar actualizaciones al iniciar"),
 "Click a mask to select it. Drag on the picture to add a head the detector missed. ←/→ step, Shift ×10, O toggles the export view.":
   ("Maske anklicken zum Auswählen. Im Bild ziehen, um einen übersehenen Kopf hinzuzufügen. ←/→ Einzelbild, Shift ×10, O schaltet die Export-Ansicht um.",
    "Cliquez sur un masque pour le sélectionner. Faites glisser sur l’image pour ajouter une tête manquée. ←/→ image par image, Maj ×10, O bascule la vue d’export.",
    "Haz clic en una máscara para seleccionarla. Arrastra sobre la imagen para añadir una cabeza no detectada. ←/→ fotograma a fotograma, Mayús ×10, O alterna la vista de exportación."),
 "Click the corners on the picture ({} so far).": ("Ecken im Bild anklicken (bisher {}).", "Cliquez les coins sur l’image ({} jusqu’ici).", "Haz clic en las esquinas de la imagen ({} hasta ahora)."),
 "Clip shows no people": ("Clip zeigt keine Personen", "Le clip ne montre personne", "El clip no muestra personas"),
 "Delete keyframe here": ("Keyframe hier löschen", "Supprimer l’image clé ici", "Eliminar fotograma clave aquí"),
 "Deselect": ("Abwählen", "Désélectionner", "Deseleccionar"),
 "Drag a box on another frame to add a keyframe to this track; frames in between are interpolated.":
   ("Auf einem anderen Frame ein Rechteck ziehen, um dieser Spur einen Keyframe hinzuzufügen; dazwischen wird interpoliert.",
    "Tracez un cadre sur une autre image pour ajouter une image clé à cette piste ; les images intermédiaires sont interpolées.",
    "Arrastra un recuadro en otro fotograma para añadir un fotograma clave a esta pista; los intermedios se interpolan."),
 "Draw zone (stands, scoreboard…)": ("Zone zeichnen (Tribüne, Anzeigetafel…)", "Dessiner une zone (tribunes, tableau…)", "Dibujar zona (gradas, marcador…)"),
 "Export training data…": ("Trainingsdaten exportieren…", "Exporter les données d’entraînement…", "Exportar datos de entrenamiento…"),
 "Export view": ("Export-Ansicht", "Vue d’export", "Vista de exportación"),
 "Finish": ("Fertig", "Terminer", "Terminar"),
 "Install update": ("Update installieren", "Installer la mise à jour", "Instalar actualización"),
 "Jersey numbers": ("Trikotnummern", "Numéros de maillot", "Dorsales"),
 "Label, e.g. stands": ("Bezeichnung, z. B. Tribüne", "Nom, p. ex. tribunes", "Nombre, p. ej. gradas"),
 "Language": ("Sprache", "Langue", "Idioma"),
 "Later": ("Später", "Plus tard", "Más tarde"),
 "Masks in this frame": ("Masken in diesem Frame", "Masques de cette image", "Máscaras en este fotograma"),
 "Model…": ("Modell…", "Modèle…", "Modelo…"),
 "Next unreviewed (n) ▶": ("Nächster ungeprüfter (n) ▶", "Suivant non vérifié (n) ▶", "Siguiente sin revisar (n) ▶"),
 "Note (optional)": ("Notiz (optional)", "Note (facultatif)", "Nota (opcional)"),
 "Only frames": ("Nur Frames", "Seulement les images", "Solo fotogramas"),
 "Open a video or a .anon project to start reviewing.": ("Video oder .anon-Projekt öffnen, um mit der Prüfung zu beginnen.", "Ouvrez une vidéo ou un projet .anon pour commencer la vérification.", "Abre un vídeo o un proyecto .anon para empezar la revisión."),
 "Open…": ("Öffnen…", "Ouvrir…", "Abrir…"),
 "Outline": ("Umriss", "Contours", "Contorno"),
 "Preview": ("Vorschau", "Aperçu", "Vista previa"),
 "Reason (required to suppress)": ("Begründung (zum Unterdrücken nötig)", "Motif (requis pour supprimer)", "Motivo (necesario para suprimir)"),
 "Remove": ("Entfernen", "Retirer", "Quitar"),
 "Render…": ("Rendern…", "Rendre…", "Renderizar…"),
 "Restore": ("Wiederherstellen", "Restaurer", "Restaurar"),
 "Review": ("Prüfung", "Vérification", "Revisión"),
 "Save": ("Speichern", "Enregistrer", "Guardar"),
 "Selected": ("Ausgewählt", "Sélection", "Seleccionado"),
 "Suppress (false positive)": ("Unterdrücken (Fehlerkennung)", "Supprimer (faux positif)", "Suprimir (falso positivo)"),
 "Text model…": ("Textmodell…", "Modèle de texte…", "Modelo de texto…"),
 "Undo point": ("Punkt zurück", "Annuler le point", "Deshacer punto"),
 "Version {} is available.": ("Version {} ist verfügbar.", "La version {} est disponible.", "La versión {} está disponible."),
 "Zones (always masked)": ("Zonen (immer maskiert)", "Zones (toujours masquées)", "Zonas (siempre enmascaradas)"),
 "auto": ("auto", "auto", "auto"),
 "manual": ("manuell", "manuel", "manual"),
 "no model selected": ("kein Modell gewählt", "aucun modèle choisi", "ningún modelo seleccionado"),
 "none": ("keine", "aucun", "ninguna"),
 "orange: flagged · green: reviewed · red: error · grey: people present": ("orange: markiert · grün: geprüft · rot: Fehler · grau: Personen", "orange : signalé · vert : vérifié · rouge : erreur · gris : personnes", "naranja: marcado · verde: revisado · rojo: error · gris: personas"),
 "selected": ("ausgewählt", "sélectionné", "seleccionado"),
 "suppressed": ("unterdrückt", "supprimé", "suprimido"),
 "to": ("bis", "à", "a"),
 "◀ Prev (p)": ("◀ Zurück (p)", "◀ Précédent (p)", "◀ Anterior (p)"),
}

def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')

here = os.path.dirname(os.path.abspath(__file__))
for i, lang in enumerate(["de", "fr", "es"]):
    d = os.path.join(here, lang, "LC_MESSAGES")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "anon-gui.po"), "w", encoding="utf-8") as f:
        f.write('msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Language: %s\\n"\n\n' % lang)
        for k in sorted(T):
            f.write(f'msgid "{esc(k)}"\nmsgstr "{esc(T[k][i])}"\n\n')
print("ok", len(T))
