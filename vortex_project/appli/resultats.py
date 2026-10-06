"""
appli/resultats.py — Zone de résultats de l'onglet de simulation.

Choix du calcul affiché dans l'historique, métriques, notes, retour sur la prédiction, unités physiques, puis
sous-onglets chargés à la demande : figures, animation interactive, vue 3D orientable, courbes, comparaison de
plusieurs calculs, vérification de convergence, données et fiche de résultats exportable (HTML).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# base64 : images intégrées à la fiche HTML ; html : échappement du texte ; math : √2 ; time : date de la fiche.
import base64
import html
import math
import time

# numpy : tests de valeurs finies ; pandas : tableaux (comparaison, convergence) ; streamlit : interface.
import numpy as np
import pandas as pd
import streamlit as st

# Modules de l'application : outils communs, aides pédagogiques, historique, tâches de calcul, vues 3D.
from appli import commun as c
from appli import pedagogie, stockage, taches, vues3d


def _recharger(widgets):
    """Fonction de bouton : remet les widgets du volet dans l'état où ils étaient au lancement du calcul."""
    # Chaque clé est celle d'un widget du volet ; la valeur s'appliquera à l'exécution suivante du script.
    for cle, valeur in widgets.items():
        st.session_state[cle] = valeur


def _nom_calcul(volet, numero):
    """Libellé d'un calcul de l'historique : numéro et description courte."""
    # Entrée correspondante de l'historique (None si elle a été retirée entre-temps).
    entree = stockage.trouver(volet.CLE, numero)
    return f"n° {numero} — {entree['etiquette']}" if entree else f"n° {numero}"


def afficher_resultats(volet, params):
    """Affiche le calcul choisi de l'historique du volet (le plus récent par défaut)."""
    # Historique du volet, du plus récent au plus ancien ; message d'invitation s'il est vide.
    entrees = stockage.historique(volet.CLE)
    if not entrees:
        st.info("Réglez les paramètres dans le panneau latéral (ou choisissez un scénario guidé), puis cliquez sur "
                "**▶️ Lancer la simulation**.", icon="👈")
        return
    # Numéros des calculs et clé du menu « Calcul affiché » (le plus récent si la valeur mémorisée n'existe plus).
    numeros = [e["id"] for e in entrees]
    cle = f"calcul_affiche_{volet.CLE}"
    if st.session_state.get(cle) not in numeros:
        st.session_state[cle] = numeros[0]
    # Menu de choix présent dès qu'il y a au moins deux calculs ; format_func : libellé lisible de chaque numéro.
    if len(entrees) > 1:
        st.selectbox("Calcul affiché", numeros, key=cle, format_func=lambda n: _nom_calcul(volet, n),
                     help="Les 8 derniers calculs du volet sont conservés tant que le serveur tourne.")
    # Entrée affichée et sortie de son calcul (métriques, figures, données…).
    entree = stockage.trouver(volet.CLE, st.session_state[cle])
    sortie = entree["sortie"]
    # Réglages actuels différents de ceux du calcul affiché : avertissement et bouton de rechargement.
    if entree["params"] != params:
        # Deux colonnes : texte large à gauche, bouton à droite, centrés verticalement.
        gauche, droite = st.columns([4, 1], vertical_alignment="center")
        gauche.warning("Les réglages actuels diffèrent de ceux de ce calcul : relancez la simulation pour les "
                       "appliquer, ou reprenez les réglages du calcul affiché.", icon="⚠️")
        # Bouton disponible si les réglages des widgets ont été enregistrés au lancement.
        if entree["meta"].get("widgets"):
            droite.button("↩️ Reprendre ses réglages", key=f"recharger_{volet.CLE}", on_click=_recharger,
                          args=(entree["meta"]["widgets"],), width="stretch")
    # Rappel : numéro, heure, durée et description du calcul affiché.
    st.caption(f"Calcul n° {entree['id']} de {entree['heure']} — durée {c.duree_lisible(sortie.get('duree', 0.0))} — "
               f"{entree['etiquette']}")
    # Retour sur la prédiction faite avant le lancement (si le scénario posait une question).
    pedagogie.retour_prediction(volet, entree)
    # Métriques en cartes, notes d'interprétation et alertes éventuelles.
    c.afficher_metriques(sortie["metriques"])
    for note in sortie.get("notes", []):
        st.markdown(note)
    for alerte in sortie.get("alertes", []):
        st.warning(alerte, icon="⚠️")
    # Transposition en unités physiques (volets Navier–Stokes) ; clé propre au calcul affiché.
    if sortie.get("unites"):
        pedagogie.convertisseur(sortie["unites"], f"{volet.CLE}_{entree['id']}")
    # Noms des sous-onglets : les onglets optionnels n'apparaissent que si le calcul fournit leurs données.
    noms = ["🖼️ Figures"]
    if sortie.get("animation"):
        noms.append("🎞️ Animation")
    if sortie.get("vue3d"):
        noms.append("🧊 Vue 3D")
    noms += ["📈 Courbes", "📊 Comparer"]
    # Convergence : seulement pour les volets qui savent raffiner leur calcul (raffinement ≠ None).
    if volet.raffinement(entree["params"], 2.0) is not None:
        noms.append("🔬 Convergence")
    noms.append("💾 Données et export")
    # Sous-onglets chargés à la demande (on_change="rerun" + key : seul l'onglet ouvert est construit) ;
    # dict(zip(...)) : nom → conteneur d'onglet.
    onglets = dict(zip(noms, st.tabs(noms, key=f"sous_onglets_{volet.CLE}", on_change="rerun")))
    # Contenu de l'onglet ouvert seulement (« .open » vaut True pour l'onglet actif) ; « with » y dirige l'affichage.
    if onglets["🖼️ Figures"].open:
        with onglets["🖼️ Figures"]:
            _figures(volet, entree)
    if "🎞️ Animation" in onglets and onglets["🎞️ Animation"].open:
        with onglets["🎞️ Animation"]:
            _animation(volet, entree)
    if "🧊 Vue 3D" in onglets and onglets["🧊 Vue 3D"].open:
        with onglets["🧊 Vue 3D"]:
            vue_3d(volet.CLE, entree["id"])
    if onglets["📈 Courbes"].open:
        with onglets["📈 Courbes"]:
            # Rappel des commandes de la souris, puis une courbe interactive par série.
            st.caption("Molette : zoom ; glisser : déplacement ; double-clic : retour à la vue initiale.")
            for courbe in sortie["courbes"]:
                c.graphique(courbe)
    if onglets["📊 Comparer"].open:
        with onglets["📊 Comparer"]:
            _comparer(volet, entrees)
    if "🔬 Convergence" in onglets and onglets["🔬 Convergence"].open:
        with onglets["🔬 Convergence"]:
            _convergence(volet, entree)
    if onglets["💾 Données et export"].open:
        with onglets["💾 Données et export"]:
            _donnees(volet, entree)


def _figures(volet, entree):
    """Figures matplotlib du calcul : choix de la figure (pastilles), image et légende."""
    # Figures du calcul et leurs titres (un titre par pastille).
    figures = entree["sortie"]["figures"]
    titres = [f["titre"] for f in figures]
    # st.pills : sélecteur à pastilles ; clé propre au calcul (les titres diffèrent d'un calcul à l'autre) ;
    # « or titres[0] » : première figure si aucune pastille n'est sélectionnée.
    choix = st.pills("Figure", titres, default=titres[0], key=f"figure_{volet.CLE}_{entree['id']}",
                     label_visibility="collapsed") or titres[0]
    # next(…) : première figure dont le titre correspond au choix.
    figure = next(f for f in figures if f["titre"] == choix)
    # Image à la largeur de la colonne, légende et texte alternatif (lecteurs d'écran).
    st.image(figure["png"], caption=figure["legende"], width="stretch", alt=figure["legende"])


def _animation(volet, entree):
    """Animation interactive (Plotly : Lecture/Pause, curseur temporel) et GIF téléchargeable éventuel."""
    sortie = entree["sortie"]
    anim = sortie["animation"]
    if anim["type"] == "navier_stokes":
        # Volets Navier–Stokes : carte de vorticité animée du plan méridien.
        st.caption("Vorticité ω_y du plan méridien (rouge : ω > 0, bleu : ω < 0) et lignes de courant ψ, dans une "
                   "fenêtre fixe. ▶ Lecture lance l'animation ; le curseur choisit un instant ; la souris zoome.")
        figure = vues3d.animation_vorticite(anim)
    else:
        # Filaments : courbes 3D animées, orientables à la souris.
        st.caption("Filaments 3D : ▶ Lecture lance l'animation ; faites tourner la vue en glissant avec la souris, "
                   "zoomez à la molette.")
        figure = vues3d.animation_filaments(anim["hist"], anim["etiquettes"], anim["suivre"], anim["fenetre"])
    # st.plotly_chart : figure Plotly interactive à la largeur de la colonne.
    st.plotly_chart(figure, width="stretch", key=f"anim_{volet.CLE}_{entree['id']}")
    # GIF téléchargeable s'il a été demandé avant le calcul (on_click="ignore" : pas de nouvelle exécution).
    if sortie.get("gif"):
        st.download_button("⬇️ Télécharger l'animation (GIF)", sortie["gif"]["octets"], file_name=sortie["gif"]["fichier"],
                           mime="image/gif", on_click="ignore", key=f"gif_{volet.CLE}_{entree['id']}")


# @st.fragment : interagir avec les réglages de la vue 3D ne réexécute que ce bloc (plus rapide).
@st.fragment
def vue_3d(cle_volet, numero):
    """Vue 3D orientable à la souris : isosurfaces (Navier–Stokes) ou tubes de vorticité (filaments)."""
    # Entrée relue à chaque exécution du fragment (elle peut avoir été retirée de l'historique).
    entree = stockage.trouver(cle_volet, numero)
    if entree is None:
        return
    vue = entree["sortie"]["vue3d"]
    if vue["type"] == "isosurface":
        # Trois colonnes de réglages : instantané (début, milieu, fin), critère et ouverture de l'écorché.
        col1, col2, col3 = st.columns([2, 1, 2])
        k = col1.radio("Instant", list(range(len(vue["snaps"]))), horizontal=True, index=len(vue["snaps"]) - 1,
                       format_func=lambda i: f"t = {vue['snaps'][i]['t']:.1f}", key=f"iso_t_{numero}")
        critere = col2.radio("Critère", ["Q", "λ₂"], horizontal=True, key=f"iso_c_{numero}")
        coupe = col3.slider("Écorché (secteur retiré, °)", 0, 180, 90, 15, key=f"iso_coupe_{numero}")
        # Isosurface de révolution de l'instantané choisi.
        figure = vues3d.isosurface(vue["snaps"][k], vue["Z"][k], vue["demi"], vue["r_max"],
                                   critere="Q" if critere == "Q" else "lambda2", coupe=coupe)
        st.caption("Isosurface reconstruite par révolution du champ axisymétrique ; couleur : norme de la vitesse.")
    else:
        # Filaments : choix de l'instant parmi les sauvegardes (curseur affichant les temps).
        hist = vue["hist"]
        n = len(hist.frames)
        k = st.select_slider("Instant", options=list(range(n)), value=n - 1, format_func=lambda i: f"t = {hist.times[i]:.2f}",
                             key=f"tubes_t_{numero}")
        # Tubes de vorticité de cet instant (rayon ∝ δ, couleur = Γ).
        figure = vues3d.tubes(hist.frames[k], hist.gammas, hist.deltas[k], hist.times[k], vue["rayon_tubes"], vue["fenetre"])
        st.caption("Tubes de vorticité : rayon proportionnel à δ, couleur = circulation du filament (rouge Γ > 0, "
                   "bleu Γ < 0). Glissez pour faire tourner la vue.")
    # Affichage de la figure Plotly (orientable et zoomable à la souris).
    st.plotly_chart(figure, width="stretch", key=f"vue3d_{numero}")


def _comparer(volet, entrees):
    """Superpose une grandeur de plusieurs calculs de l'historique et compare leurs métriques."""
    # Il faut au moins deux calculs pour comparer.
    if len(entrees) < 2:
        st.info("Lancez au moins deux calculs de ce volet avec des réglages différents (par exemple deux scénarios "
                "guidés, ou un Reynolds modifié) pour les comparer ici.", icon="📊")
        return
    numeros = [e["id"] for e in entrees]
    # st.multiselect : choix de plusieurs calculs (les trois plus récents par défaut).
    choix = st.multiselect("Calculs à comparer", numeros, default=numeros[:3], key=f"comparer_{volet.CLE}",
                           format_func=lambda n: _nom_calcul(volet, n))
    choisis = [stockage.trouver(volet.CLE, n) for n in choix]
    if len(choisis) < 2:
        st.caption("Choisissez au moins deux calculs.")
        return
    # Grandeurs disponibles dans tous les calculs choisis (dans l'ordre du premier) : all/any vérifient leur présence.
    noms = [s["nom"] for s in choisis[0]["sortie"].get("comparables", [])]
    communs = [n for n in noms if all(any(s["nom"] == n for s in e["sortie"].get("comparables", [])) for e in choisis)]
    if not communs:
        st.caption("Ces calculs n'ont pas de grandeur commune à superposer (configurations différentes).")
    else:
        grandeur = st.selectbox("Grandeur à superposer", communs, key=f"grandeur_{volet.CLE}")
        # Une série par calcul (format long : abscisse, valeur, nom de la série), concaténées par pd.concat.
        series = []
        for e in choisis:
            s = next(s for s in e["sortie"]["comparables"] if s["nom"] == grandeur)
            series.append(pd.DataFrame({"x": s["x"], "valeur": s["y"], "série": f"n° {e['id']} — {e['etiquette']}"}))
        c.graphique(c.courbe(grandeur, pd.concat(series), "valeur", s["y_label"], x="x", x_label=s["x_label"]))
    # Tableau des métriques : une colonne par calcul, une ligne par métrique (dictionnaire de dictionnaires).
    tableau = pd.DataFrame({f"n° {e['id']}": {m["label"]: m["valeur"] for m in e["sortie"]["metriques"]} for e in choisis})
    st.dataframe(tableau, width="stretch")
    # Effacement de l'historique du volet (la mémoire des résultats est conservée).
    st.button("🗑️ Vider l'historique de ce volet", key=f"vider_{volet.CLE}", on_click=stockage.effacer, args=(volet.CLE,))


def _verdict(ecart):
    """Verdict d'un écart relatif de convergence (critère du tutoriel : moins de 1 %)."""
    # Écart non défini (grandeur absente ou NaN) : pas de verdict.
    if not np.isfinite(ecart):
        return "—"
    # < 1 % : convergé ; 1 à 5 % : à surveiller ; > 5 % : résolution insuffisante.
    return "✅ < 1 %" if ecart < 0.01 else ("⚠️ 1 à 5 %" if ecart < 0.05 else "❌ > 5 %")


def _convergence(volet, entree):
    """Vérification de convergence : refait le calcul avec une résolution plus fine et compare les grandeurs clés."""
    st.markdown("Un résultat est **convergé** quand ses grandeurs clés changent de moins de 1 % lorsqu'on affine la "
                "résolution (tutoriel, « vérifier qu'un résultat est fiable »).")
    # Facteur de raffinement : √2 (coût ≈ ×3) ou 2 (coût ≈ ×8, critère du tutoriel).
    facteur = st.radio("Raffinement", [math.sqrt(2.0), 2.0], horizontal=True, key=f"facteur_{volet.CLE}",
                       format_func=lambda f: "résolution × √2 (coût ≈ ×3)" if f < 1.5 else "résolution × 2 (coût ≈ ×8)")
    # Paramètres du calcul raffiné et description du raffinement (fournis par le volet).
    p_fin, description = volet.raffinement(entree["params"], facteur)
    conv = entree.get("convergence")
    if conv:
        # Tableau : grandeur, valeur du calcul, valeur raffinée, écart relatif et verdict.
        lignes = []
        for nom, v in entree["sortie"]["grandeurs"].items():
            # Valeur raffinée (NaN si absente) ; écart relatif rapporté à la valeur raffinée.
            v2 = conv["grandeurs"].get(nom, float("nan"))
            ecart = abs(v2 - v) / abs(v2) if np.isfinite(v) and np.isfinite(v2) and v2 != 0 else float("nan")
            lignes.append({"grandeur": nom, "calcul": v, "calcul raffiné": v2,
                           "écart relatif": f"{100 * ecart:.2f} %" if np.isfinite(ecart) else "—", "verdict": _verdict(ecart)})
        # hide_index=True : pas de colonne d'indices.
        st.dataframe(pd.DataFrame(lignes), hide_index=True, width="stretch")
        st.caption(f"Calcul raffiné : {conv['description']} — durée {c.duree_lisible(conv['duree'])}.")
    # Estimation du calcul raffiné (alertes ignorées : « _ ») et bouton de lancement, désactivé pendant un calcul.
    _, est = volet.verifier(p_fin)
    st.caption(f"Calcul raffiné : {description} ; durée estimée ≈ {c.duree_lisible(est['calcul'])}.")
    if st.button("🔬 Lancer la vérification", key=f"convergence_{volet.CLE}_{entree['id']}", disabled=taches.en_cours()):
        # Calcul raffiné en arrière-plan ; nouvelle exécution du script pour afficher sa progression.
        taches.lancer_convergence(volet, entree, p_fin, description, est)
        st.rerun()


def _donnees(volet, entree):
    """Tableau des diagnostics, export CSV et fiche de résultats HTML."""
    sortie = entree["sortie"]
    # Tableau consultable (tri, défilement).
    st.dataframe(sortie["donnees"], height=320, width="stretch")
    gauche, droite = st.columns(2)
    # Export CSV (to_csv, encodé en UTF-8) ; on_click="ignore" : le téléchargement ne relance pas le script.
    gauche.download_button("⬇️ Données (CSV)", sortie["donnees"].to_csv(index=False).encode("utf-8"),
                           file_name=sortie["fichier_csv"], mime="text/csv", on_click="ignore",
                           key=f"csv_{volet.CLE}_{entree['id']}", width="stretch")
    # Fiche de résultats autonome (HTML), imprimable en PDF depuis le navigateur.
    droite.download_button("📄 Fiche de résultats (HTML)", fiche_html(volet, entree).encode("utf-8"),
                           file_name=f"fiche_{volet.CLE}_{entree['id']}.html", mime="text/html", on_click="ignore",
                           key=f"fiche_{volet.CLE}_{entree['id']}", width="stretch",
                           help="Page autonome (paramètres, métriques, notes, figures, convergence), imprimable en PDF "
                                "depuis le navigateur — utile pour rendre un TP.")


def fiche_html(volet, entree):
    """Fiche de résultats autonome (HTML, figures intégrées en base64) d'un calcul de l'historique."""
    sortie = entree["sortie"]
    # html.escape : neutralise les caractères spéciaux (<, >, &) des textes insérés.
    e = html.escape
    # Lignes de tableau HTML : paramètres, puis métriques (libellé, valeur, complément).
    parametres = "".join(f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in entree["params"].items())
    metriques = "".join(f"<tr><td>{e(m['label'])}</td><td><b>{e(m['valeur'])}</b></td><td>{e(m['delta'] or '')}</td></tr>"
                        for m in sortie["metriques"])
    # Notes en paragraphes (marques de gras Markdown « ** » retirées).
    notes = "".join(f"<p>{e(n).replace('**', '')}</p>" for n in sortie.get("notes", []))
    # Figures : PNG encodés en base64 dans des balises <img> (fiche lisible sans connexion ni fichiers annexes).
    figures = "".join(f"<figure><img src='data:image/png;base64,{base64.b64encode(f['png']).decode()}' alt='{e(f['legende'])}'>"
                      f"<figcaption><b>{e(f['titre'])}.</b> {e(f['legende'])}</figcaption></figure>"
                      for f in sortie["figures"])
    # Tableau de convergence s'il a été calculé.
    convergence = ""
    if entree.get("convergence"):
        conv = entree["convergence"]
        lignes = "".join(f"<tr><td>{e(nom)}</td><td>{v:.5g}</td><td>{conv['grandeurs'].get(nom, float('nan')):.5g}</td></tr>"
                         for nom, v in sortie["grandeurs"].items())
        convergence = (f"<h2>Vérification de convergence</h2><p>{e(conv['description'])}</p><table><tr><th>grandeur</th>"
                       f"<th>calcul</th><th>calcul raffiné</th></tr>{lignes}</table>")
    # Feuille de style minimale (police, tableaux, figures).
    style = ("body{font-family:'Segoe UI',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#0b0b0b}"
             "h1{font-size:1.6em}h2{font-size:1.2em;margin-top:1.6em}table{border-collapse:collapse;margin:8px 0}"
             "td,th{border:1px solid #e1e0d9;padding:4px 10px;text-align:left}figure{margin:18px 0}"
             "img{max-width:100%}figcaption{color:#52514e;font-size:.9em}")
    # Page complète : en-tête, paramètres, résultats, notes, convergence, figures ; time.strftime : date du jour.
    return (f"<!doctype html><html lang='fr'><head><meta charset='utf-8'><title>Fiche — {e(volet.LIBELLE)}</title>"
            f"<style>{style}</style></head><body><h1>{e(volet.LIBELLE)}</h1>"
            f"<p>Calcul n° {entree['id']} du {time.strftime('%d/%m/%Y')} à {entree['heure']} — durée "
            f"{e(c.duree_lisible(sortie.get('duree', 0.0)))} — {e(entree['etiquette'])}</p>"
            f"<h2>Paramètres</h2><table>{parametres}</table><h2>Résultats</h2><table>{metriques}</table>{notes}"
            f"{convergence}<h2>Figures</h2>{figures}"
            f"<p style='color:#898781'>Fiche produite par l'application « Vortex toroïdaux &amp; moteurs à vortex ».</p>"
            f"</body></html>")
