"""
appli/volets/anneau.py — Volet « anneau visqueux » : anneau de Lamb–Oseen ou vortex sphérique de Hill dans un
fluide visqueux, résolu par le solveur Navier–Stokes axisymétrique (src/solver_2d_axisymmetric.py).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# time : durée des tracés ; numpy : tableaux ; pandas : courbes et données ; streamlit : widgets.
import time

import numpy as np
import pandas as pd
import streamlit as st
# cumulative_trapezoid : trajectoire théorique ∫ U dt.
from scipy.integrate import cumulative_trapezoid

from appli import commun as c
from appli.glossaire import aide
from appli.volets import navier_stokes as ns
from src import solver_2d_axisymmetric as ax
from src import theory
from src import visualization as viz
from src import visualization_3d as v3

# Identité du volet, résumé (carte de l'accueil) et présentation (onglet de simulation).
CLE = "anneau"
LIBELLE = "🌀 Anneau visqueux (Navier–Stokes)"
RESUME = ("Un anneau (ou un vortex de Hill) ralentit en s'épaississant dans un fluide visqueux : comparaison aux "
          "formules de Saffman et de Fukumoto–Moffatt, bilan d'énergie, vues 3D.")
PRESENTATION = ("Anneau de Lamb–Oseen (cœur gaussien) ou vortex sphérique de Hill dans un fluide visqueux : les "
                "équations de Navier–Stokes axisymétriques sont résolues par projection sur grille décalée "
                "(`ax.run_vortex_ring`, `ax.run_hill_vortex`). La vitesse simulée est comparée aux formules "
                "théoriques, et le bilan dE/dt = −νΩ vérifie l'absence de dissipation numérique.")
# Formule clé : vitesse de Saffman corrigée par Fukumoto–Moffatt, avec un cœur qui diffuse.
FORMULE = (r"U(t) = \frac{\Gamma}{4\pi R}\left[\ln\frac{8R}{\sigma} - 0{,}558 - 0{,}918\,\frac{\sigma^2}{R^2}\right],"
           r"\qquad \sigma(t)^2 = a^2 + 4\nu t")
# Sections du rapport théorique affichées à côté du volet.
THEORIE = [("Fondements théoriques", "2.3"), ("Fondements théoriques", "2.1")]
# Hypothèses et limites du modèle.
LIMITES = [
    "Écoulement axisymétrique sans rotation autour de l'axe (pas de swirl) : les instabilités 3D des anneaux "
    "(instabilité de Widnall) ne peuvent pas apparaître.",
    "Domaine périodique en z, de longueur 10 : un anneau qui sort par le haut rentre par le bas.",
    "Paroi glissante en r = 8R ; écoulement laminaire, sans modèle de turbulence.",
    "Formules de Saffman et de Fukumoto–Moffatt établies pour un anneau mince (a/R ≲ 0,35).",
]
# Calcul non instantané (bouton de lancement).
INSTANTANE = False
# Préfixe des clés de ses widgets (réglages enregistrés avec chaque calcul, puis rechargeables).
PREFIXE = "an_"
# Scénarios guidés : réglages, « ce qu'il faut observer » et question « prédire puis vérifier ».
SCENARIOS = {
    # Scénario 1 : anneau de référence du rapport (Re = 1000).
    "reference": {
        "titre": "Anneau de référence (Re = 1000)",
        "reglages": {"an_ci": "lamb_oseen", "an_R": 1.0, "an_a": 0.25, "an_G": 1.0, "an_Re": 1000, "an_T": 8.0,
                     "an_dr": 0.04},
        "observer": "La vitesse U(t) décroît lentement : la diffusion épaissit le cœur (σ² = a² + 4νt) et l'anneau "
                    "ralentit en suivant Saffman + Fukumoto–Moffatt à quelques pour cent près. Environ 15 % de "
                    "l'énergie est dissipée.",
        "question": {"texte": "comment évolue la vitesse de l'anneau ?",
                     "choix": ["Elle reste constante", "Elle diminue lentement", "Elle augmente"],
                     "reponse": "Elle diminue lentement",
                     "explication": "La viscosité épaissit le cœur (σ croît) ; or U varie comme ln(8R/σ) : un cœur "
                                    "plus épais rend l'anneau plus lent."},
    },
    # Scénario 2 : même anneau dans un fluide cinq fois plus visqueux.
    "re_faible": {
        "titre": "Fluide très visqueux (Re = 200)",
        "reglages": {"an_ci": "lamb_oseen", "an_R": 1.0, "an_a": 0.25, "an_G": 1.0, "an_Re": 200, "an_T": 8.0,
                     "an_dr": 0.04},
        "observer": "Le cœur s'épaissit vite : l'anneau ralentit fortement et perd l'essentiel de son énergie. "
                    "Comparez avec l'anneau de référence dans l'onglet « Comparer ».",
        "question": {"texte": "par rapport à Re = 1000, l'énergie dissipée sera…",
                     "choix": ["plus faible", "comparable", "plus forte"], "reponse": "plus forte",
                     "explication": "La dissipation vaut νΩ : à circulation égale, une viscosité cinq fois plus "
                                    "grande dissipe beaucoup plus d'énergie."},
    },
    # Scénario 3 : cœur plus fin (maillage affiné pour garder a/Δr = 6).
    "coeur_fin": {
        "titre": "Cœur fin (a = 0,15)",
        "reglages": {"an_ci": "lamb_oseen", "an_R": 1.0, "an_a": 0.15, "an_G": 1.0, "an_Re": 1000, "an_T": 6.0,
                     "an_dr": 0.025},
        "observer": "Un cœur plus fin donne un anneau plus rapide, comme le prévoit U = Γ/(4πR)[ln(8R/a) − 0,558].",
        "question": {"texte": "par rapport à l'anneau de référence (a = 0,25), l'anneau ira…",
                     "choix": ["moins vite", "aussi vite", "plus vite"], "reponse": "plus vite",
                     "explication": "U croît comme ln(8R/a) : diviser a par 1,7 augmente la vitesse d'environ 15 %."},
    },
    # Scénario 4 : anneau épais, hors du domaine des formules d'anneau mince.
    "epais": {
        "titre": "Anneau épais (a/R = 0,4)",
        "reglages": {"an_ci": "lamb_oseen", "an_R": 1.0, "an_a": 0.4, "an_G": 1.0, "an_Re": 1000, "an_T": 8.0,
                     "an_dr": 0.05},
        "observer": "Hors du domaine des anneaux minces, l'écart aux formules de Saffman grandit : comparez l'écart "
                    "relatif avec celui de l'anneau de référence.",
    },
    # Scénario 5 : vortex de Hill de même circulation que l'anneau de référence (Γ = 5Ua = 1).
    "hill": {
        "titre": "Vortex sphérique de Hill (Γ = 1)",
        "reglages": {"an_ci": "hill", "an_ha": 1.0, "an_hU": 0.2, "an_Re": 1000, "an_T": 8.0, "an_dr": 0.04},
        "observer": "En fluide parfait, la sphère avancerait à U sans se déformer. La viscosité lisse la "
                    "discontinuité de vorticité à sa surface, crée un sillage et ralentit le vortex, dont la vorticité "
                    "se concentre peu à peu en anneau.",
        "question": {"texte": "la vitesse du vortex de Hill au cours du temps…",
                     "choix": ["reste égale à U", "diminue", "augmente"], "reponse": "diminue",
                     "explication": "La solution de Hill n'est exacte qu'en fluide parfait ; la viscosité dissipe son "
                                    "énergie et il ralentit, comme un anneau."},
    },
}


def controles():
    """Widgets du volet (panneau latéral) ; renvoie le dictionnaire des paramètres."""
    st.caption("Unités adimensionnelles : longueurs en unités de R₀ = 1, temps en R₀²/Γ₀.")
    p = {}
    # Condition initiale : anneau gaussien ou vortex sphérique de Hill (menu déroulant).
    p["ci"] = c.menu("Condition initiale", ["lamb_oseen", "hill"], "lamb_oseen", "an_ci",
                     format_func={"lamb_oseen": "Anneau de Lamb–Oseen", "hill": "Vortex sphérique de Hill"}.get,
                     aide=aide("Lamb–Oseen (cœur de)", "Vortex de Hill"))
    if p["ci"] == "lamb_oseen":
        # Anneau de Lamb–Oseen : rayon, rayon de cœur et circulation.
        p["R"] = c.curseur("Rayon de l'anneau R", 0.5, 1.5, 1.0, 0.05, "an_R", "Distance de l'axe au centre du cœur.")
        p["a"] = c.curseur("Rayon du cœur gaussien a", 0.10, 0.45, 0.25, 0.01, "an_a", aide("Rayon de cœur a (σ)"))
        p["Gamma"] = c.curseur("Circulation Γ", 0.25, 2.0, 1.0, 0.05, "an_G", aide("Circulation Γ"))
    else:
        # Vortex de Hill : rayon de la sphère et vitesse de propagation en fluide parfait.
        p["hill_a"] = c.curseur("Rayon de la sphère a", 0.5, 1.5, 1.0, 0.05, "an_ha")
        p["hill_U"] = c.curseur("Vitesse de Hill U", 0.1, 0.5, 0.2, 0.01, "an_hU",
                                "Γ = 5Ua : avec a = 1 et U = 0,2, Γ = 1 comme l'anneau de référence.")
    # Réglages communs : Reynolds (échelle quasi logarithmique), durée et maillage.
    p["Re"] = c.curseur_liste("Reynolds Re_Γ = Γ/ν", ns.OPTIONS_RE, 1000, "an_Re", aide("Nombre de Reynolds Re_Γ"))
    p["t_end"] = c.curseur("Durée simulée t_end", 1.0, 30.0, 8.0, 0.5, "an_T")
    p["dr"] = c.curseur_liste("Pas de maillage Δr (grossier → fin)", ns.OPTIONS_DR, 0.04, "an_dr",
                              "Garder a/Δr ≥ 5 (≥ 8 pour une précision de rapport). Le coût varie en 1/Δr³.")
    # Options repliables : figures supplémentaires et GIF téléchargeable.
    with st.expander("Diagnostics et rendus"):
        p["dissipation"] = c.interrupteur("Carte de dissipation Φ", True, "an_phi", aide("Dissipation Φ"))
        p["isosurfaces"] = c.interrupteur("Isosurfaces 3D (critères Q et λ₂)", True, "an_iso", aide("Critère Q", "Critère λ₂"))
        p["volume"] = c.interrupteur("Rendu volumique de l'enstrophie (lent)", False, "an_vol", aide("Enstrophie Ω"))
        p["gif"] = c.interrupteur("Animation GIF téléchargeable", False, "an_gif",
                                  "L'animation interactive est toujours disponible ; le GIF sert à l'exporter.")
    return p


def _geometrie(p):
    """Rayon de référence, rayon de cœur, circulation, domaine (Lr, zone uniforme) et type de coût du calcul."""
    if p["ci"] == "hill":
        # Vortex de Hill : sphère de rayon a, Γ = 5Ua ; le « cœur » est la sphère elle-même.
        a = p["hill_a"]
        return {"R": a, "a": a, "Gamma": 5.0 * p["hill_U"] * a, "Lr": 8.0 * max(a, 1.0), "r_uniforme": max(2.2, a + 1.2),
                "type_cout": "hill"}
    return {"R": p["R"], "a": p["a"], "Gamma": p["Gamma"], "Lr": 8.0 * p["R"], "r_uniforme": max(2.2, p["R"] + 1.2),
            "type_cout": "anneau"}


def verifier(p):
    """Garde-fous numériques et durée estimée."""
    # Géométrie du calcul (rayon, cœur, circulation, domaine).
    g = _geometrie(p)
    # Durée estimée des figures : tracés de base puis options (secondes, avant calibration).
    figures = 6.0 + 3.0 * p["isosurfaces"] + 4.0 * p["dissipation"] + 15.0 * p["volume"] + 12.0 * p["gif"]
    if p["ci"] == "hill":
        # Vitesse maximale du vortex de Hill ≈ 3U (2,5U au centre dans le repère du laboratoire).
        est = ns.estimation("hill", [{"R": g["R"], "a": g["a"], "Gamma": g["Gamma"]}], p["Re"], p["t_end"], p["dr"],
                            g["Lr"], g["r_uniforme"], figures, vitesse=3.0 * p["hill_U"])
        alertes = ns.alertes_resolution(g["a"], p["dr"], est["peclet"])
    else:
        # Anneau : vitesse maximale estimée à partir du cœur gaussien.
        est = ns.estimation("anneau", [g], p["Re"], p["t_end"], p["dr"], g["Lr"], g["r_uniforme"], figures)
        alertes = ns.alertes_resolution(p["a"], p["dr"], est["peclet"])
        # Domaine de validité des formules d'anneau mince.
        if p["a"] / p["R"] > 0.35:
            alertes.append("Anneau épais (a/R > 0,35) : les formules de Saffman et Fukumoto–Moffatt, établies pour un "
                           "anneau mince, deviennent approximatives.")
    return alertes, est


def etiquette(p):
    """Description courte d'un calcul (historique et comparaison)."""
    if p["ci"] == "hill":
        # Vortex de Hill : vitesse, rayon, Reynolds, maillage et durée.
        return f"Hill · U {p['hill_U']:g} · a {p['hill_a']:g} · Re {p['Re']:g} · Δr {p['dr']:g} · t {p['t_end']:g}"
    # Anneau : Reynolds, cœur, maillage et durée.
    texte = f"Re {p['Re']:g} · a {p['a']:g} · Δr {p['dr']:g} · t {p['t_end']:g}"
    # R et Γ ne sont mentionnés que s'ils diffèrent des valeurs de référence.
    if p["R"] != 1.0:
        texte += f" · R {p['R']:g}"
    if p["Gamma"] != 1.0:
        texte += f" · Γ {p['Gamma']:g}"
    return texte


def raffinement(p, facteur):
    """Paramètres du calcul de convergence : maillage affiné d'un facteur (√2 ou 2), sans figures optionnelles."""
    # dict(p, …) : copie des paramètres, Δr divisé par le facteur, figures optionnelles coupées.
    p2 = dict(p, dr=p["dr"] / facteur, dissipation=False, isosurfaces=False, volume=False, gif=False)
    return p2, f"Δr = {p['dr']:g} → {p2['dr']:.4g} (coût ≈ ×{facteur**3:.0f})"


def calculer(p, prog, complet=True):
    """Calcul complet (figures, animation, vues 3D) ou léger (``complet=False`` : grandeurs pour la convergence)."""
    # Géométrie (rayon, cœur, circulation, domaine) et paramètres numériques.
    g = _geometrie(p)
    T, Re, dr = p["t_end"], float(p["Re"]), p["dr"]
    # Rappel du solveur : progression et images légères de l'animation.
    rappel, images = ns.enregistreur_images(T, prog, "Navier–Stokes")
    # Instantanés complets au début, au milieu et à la fin (instants atteints exactement par le solveur).
    instants = (0.0, round(0.5 * T, 6), T)
    if p["ci"] == "hill":
        # Appel direct de la fonction du projet : vortex de Hill visqueux.
        res = ax.run_hill_vortex(a=g["a"], U=p["hill_U"], Re=Re, t_end=T, dr=dr, Lr=g["Lr"], r_uniform=g["r_uniforme"],
                                 snapshot_times=instants, record_profiles=True, callback=rappel)
    else:
        # Appel direct de la fonction du projet : anneau de Lamb–Oseen (comparaison à Saffman incluse).
        res = ax.run_vortex_ring(R=p["R"], a=p["a"], Gamma=p["Gamma"], Re=Re, t_end=T, dr=dr, Lr=g["Lr"],
                                 r_uniform=g["r_uniforme"], snapshot_times=instants, record_profiles=True,
                                 callback=rappel)
    # Diagnostics (séries temporelles), instantanés complets et grille du calcul.
    d, snaps, grille = res["diagnostics"], res["snapshots"], res["grid"]
    # Vitesse finale (moyenne des 10 derniers diagnostics), énergie dissipée, bilan et qualité numérique.
    U_fin = float(np.mean(res["U_sim"][-10:]))
    perte = float(100.0 * (1.0 - d["energy"][-1] / d["energy"][0]))
    E_th, residu = ns.bilan_energie(d)
    parasite = ns.vorticite_parasite(snaps[-1])
    # Grandeurs comparées par la vérification de convergence (et reprises dans la fiche de résultats).
    grandeurs = {"Vitesse finale U": U_fin, "Énergie dissipée (%)": perte, "Rayon final R": float(d["R"][-1])}
    if p["ci"] == "lamb_oseen":
        grandeurs["Rayon de cœur final σ"] = float(d["sigma"][-1])
    # Mesures pour la calibration des durées estimées.
    mesures = {"t_calcul": res["wall_time"], "n_pas": res["solver"].n_steps, "taille": grille.Nr * grille.Nz}
    # Sortie minimale ; calcul léger (convergence) : on s'arrête là.
    sortie = {"grandeurs": grandeurs, "mesures": mesures, "type_cout": g["type_cout"]}
    if not complet:
        return sortie
    # Positions aux instantanés et rayon de cœur final théorique (taille des fenêtres d'affichage).
    Z_s = [float(np.interp(s["t"], d["t"], d["Z"])) for s in snaps]
    sigma_f = float(theory.viscous_core_radius(g["a"], g["Gamma"] / Re, T)) if p["ci"] == "lamb_oseen" else 0.6 * g["a"]
    # Rayon de la fenêtre d'affichage et fenêtre axiale fixe couvrant toute la trajectoire.
    r_vue = g["R"] + 0.8 + 2.0 * sigma_f
    centre, demi = c.fenetre_axiale(d["Z"])
    # Images de l'animation complétées par l'instant initial, échelle de couleur commune.
    animation = ns.donnees_animation(images, snaps[0], centre, demi, r_vue)
    # Chronomètre des tracés (calibration de la durée estimée des figures).
    debut_figures = time.perf_counter()
    if p["ci"] == "hill":
        # Figures, métriques, courbes et données du vortex de Hill…
        sortie.update(_resultats_hill(p, res, d, snaps, Z_s, U_fin, perte, residu, parasite, grille, r_vue, prog))
    else:
        # … ou de l'anneau de Lamb–Oseen.
        sortie.update(_resultats_anneau(p, res, d, snaps, Z_s, U_fin, perte, residu, parasite, grille, sigma_f, prog))
    # GIF téléchargeable (optionnel).
    sortie["gif"] = None
    if p["gif"]:
        sortie["gif"] = {"octets": ns.gif_vorticite(animation, prog, 0.92), "fichier": "anneau_vorticite.gif"}
    # Durée des tracés et de l'animation.
    sortie["mesures"]["t_figures"] = time.perf_counter() - debut_figures
    # Données d'affichage interactif : animation (Plotly) et isosurfaces 3D des trois instantanés.
    sortie["animation"] = dict(animation, type="navier_stokes")
    sortie["vue3d"] = {"type": "isosurface", "snaps": snaps, "Z": Z_s, "demi": 0.5 + 2.5 * sigma_f,
                       "r_max": g["R"] + 0.5 + 2.0 * sigma_f}
    # Transposition en unités physiques (rayon et circulation simulés, vitesse, durée, distance parcourue).
    sortie["unites"] = {"Re": Re, "R": g["R"], "Gamma": g["Gamma"], "U": U_fin, "t_end": T,
                        "distance": float(d["Z"][-1] - d["Z"][0])}
    return sortie


def _resultats_anneau(p, res, d, snaps, Z_s, U_fin, perte, residu, parasite, grille, sigma_f, prog):
    """Figures, métriques, courbes et données de l'anneau de Lamb–Oseen."""
    # Rayon de l'anneau et vitesses théoriques finales (moyennes des 10 derniers diagnostics).
    R = p["R"]
    U_fm = float(np.mean(res["U_saffman_fm"][-10:]))
    U_sa = float(np.mean(res["U_saffman"][-10:]))
    # Trajectoire théorique Z(0) + ∫ U_Saffman+FM dt.
    Z_th = d["Z"][0] + cumulative_trapezoid(res["U_saffman_fm"], d["t"], initial=0.0)
    # Figures de base (module visualization) : (titre, légende, fonction de tracé).
    taches = [
        ("Champs finaux", "Vorticité, lignes de courant dans le repère de l'anneau (bulle transportée), vitesse et "
                          "pression à la fin du calcul.",
         lambda: viz.plot_axisymmetric_fields(snaps[-1], Z_ring=Z_s[-1], U_ring=U_fin, r_max=max(2.4, R + 0.9 + 2 * sigma_f))),
        ("Vitesse vs théorie", "Vitesse simulée dZ/dt comparée à Saffman et à Saffman + Fukumoto–Moffatt "
                               "(σ² = a² + 4νt), écart relatif, rayon de cœur et bilan d'énergie.",
         lambda: viz.plot_ring_speed_validation(res)),
        ("Hovmöller z–t", "Maximum radial de la vorticité et minimum de pression en fonction de z et t : la pente de "
                          "la trace est la vitesse ; tirets : trajectoire théorique ∫ U_Saffman+FM dt.",
         lambda: viz.plot_hovmoller(res["profiles"], tracks=[(d["t"], d["Z"], "centroïde simulé")],
                                    theory_tracks=[(d["t"], Z_th, "∫ U Saffman + F–M dt")])),
        ("Bilan d'énergie", "Énergie, impulsion et enstrophie : E(t) suit E₀ − ν∫Ω dt (pas de dissipation numérique).",
         lambda: viz.plot_energy_enstrophy_budget(res)),
    ]
    # Figures optionnelles, selon les interrupteurs du panneau latéral.
    if p["dissipation"]:
        taches.append(("Dissipation Φ", "Taux local de dissipation Φ = 2μ S:S aux trois instantanés : couronne de rayon "
                                        "≈ 1,34 a autour du cœur.",
                       lambda: viz.plot_dissipation_map(snaps, Z_s, r_max=max(2.2, R + 0.7 + 2 * sigma_f))))
    if p["isosurfaces"]:
        taches.append(("Isosurfaces 3D", "Instantané final reconstruit par révolution : isosurfaces des critères Q "
                                         "(couleur |u|) et λ₂ (couleur = dépression), en écorché.",
                       lambda: v3.plot_isosurfaces(snaps[-1], Z_s[-1], half_width=0.5 + 2.5 * sigma_f,
                                                   r_max=R + 0.5 + 2.0 * sigma_f)))
    if p["volume"]:
        taches.append(("Rendu volumique", "Enstrophie ½ω² reconstruite par révolution et rendue par lancer de rayons "
                                          "(début et fin du calcul).",
                       lambda: ns.rendu_volumique([snaps[0], snaps[-1]], [Z_s[0], Z_s[-1]], 1.6 * max(R, 1.0), R + 0.8,
                                                  "Anneau visqueux — rendu volumique de ½ω²")))
    # Tracés (barre de progression de 75 % à 92 % ou 100 %).
    figures = c.tracer(taches, prog, 0.75, 0.92 if p["gif"] else 1.0)
    # Écarts relatifs (en %) de la vitesse simulée aux deux formules.
    ecart_fm, ecart_sa = 100.0 * (U_fin / U_fm - 1.0), 100.0 * (U_fin / U_sa - 1.0)
    # Métriques en cartes (mini-courbes pour les grandeurs qui évoluent).
    metriques = [
        c.metrique("Vitesse finale U", f"{U_fin:.4f}", f"{ecart_fm:+.1f} % vs Saffman + F–M",
                   "dZ/dt moyennée sur la dernière unité de temps.", c.sous_echantillon(res["U_sim"])),
        c.metrique("Théorie : Saffman + F–M", f"{U_fm:.4f}", f"Saffman seul : {U_sa:.4f}",
                   aide("Saffman (formule de)", "Fukumoto–Moffatt (correction de)")),
        c.metrique("Énergie dissipée", f"{perte:.1f} %", None, "1 − E(t_end)/E(0).",
                   c.sous_echantillon(d["energy"] / d["energy"][0])),
        c.metrique("Résidu du bilan d'énergie", f"{residu:.1e} E₀", None,
                   "max |E − (E₀ − ∫νΩ dt)| / E₀ : mesure la dissipation numérique (≈ 0 attendu)."),
        c.metrique("Rayon de cœur σ final", f"{d['sigma'][-1]:.3f}", f"théorie : {res['sigma_theory'][-1]:.3f}",
                   aide("Rayon de cœur a (σ)"), c.sous_echantillon(d["sigma"])),
        c.metrique("Rayon de l'anneau R final", f"{d['R'][-1]:.3f}", f"initial : {d['R'][0]:.3f}"),
        c.metrique("Qualité numérique", f"{100 * parasite:.2f} %", "vorticité négative parasite",
                   "−min ω / max ω à la fin : des oscillations numériques créent de la vorticité négative ; moins de "
                   "0,2 % est négligeable. " + (aide("Nombre de Péclet de maille") or "")),
        c.metrique("Grille", f"{grille.Nr} × {grille.Nz}", f"{res['solver'].n_steps} pas — {res['wall_time']:.1f} s",
                   aide("CFL (condition de)", "Divergence (incompressibilité)")),
    ]
    # Interprétation chiffrée du calcul.
    notes = [f"En fin de calcul, la vitesse simulée s'écarte de **{ecart_fm:+.1f} %** de la formule de Saffman + "
             f"Fukumoto–Moffatt et de **{ecart_sa:+.1f} %** de celle de Saffman seule. La viscosité a dissipé "
             f"**{perte:.1f} %** de l'énergie initiale ; la divergence maximale vaut {d['max_div'].max():.1e}."]
    # Courbes interactives et grandeurs comparables entre calculs.
    # Tableaux des courbes interactives : vitesses et rayon de cœur.
    df_U = pd.DataFrame({"t": d["t"], "U simulée": res["U_sim"], "Saffman": res["U_saffman"],
                         "Saffman + F–M": res["U_saffman_fm"]})
    df_s = pd.DataFrame({"t": d["t"], "σ mesuré": d["sigma"], "√(a² + 4νt)": res["sigma_theory"]})
    courbes = [c.courbe("Vitesse de propagation U(t)", df_U, ["U simulée", "Saffman", "Saffman + F–M"], "U",
                        couleurs=viz.SERIES[:3]),
               ns.courbe_bilan(d),
               c.courbe("Rayon de cœur σ(t)", df_s, ["σ mesuré", "√(a² + 4νt)"], "σ", couleurs=viz.SERIES[:2])]
    # Séries comparables : celles de _comparables, plus le rayon de cœur.
    comparables = _comparables(d, res["U_sim"]) + [{"nom": "Rayon de cœur σ(t)", "x": d["t"], "y": d["sigma"],
                                                    "x_label": "t", "y_label": "σ"}]
    # Tableau des diagnostics (même contenu que outputs/data/axisym_diagnostics.csv).
    donnees = pd.DataFrame({"t": d["t"], "Z": d["Z"], "R": d["R"], "sigma": d["sigma"], "U_sim": res["U_sim"],
                            "U_saffman": res["U_saffman"], "U_saffman_fm": res["U_saffman_fm"],
                            "circulation": d["circulation"], "energy": d["energy"], "dissipation": d["dissipation"],
                            "max_div": d["max_div"]})
    # Éléments de la sortie propres à l'anneau.
    return {"metriques": metriques, "notes": notes, "figures": figures, "courbes": courbes,
            "comparables": comparables, "donnees": donnees, "fichier_csv": "anneau_diagnostics.csv"}


def _resultats_hill(p, res, d, snaps, Z_s, U_fin, perte, residu, parasite, grille, r_vue, prog):
    """Figures, métriques, courbes et données du vortex de Hill visqueux."""
    # Vitesse du vortex de Hill en fluide parfait (référence).
    U0 = p["hill_U"]
    # Trajectoire du vortex de Hill en fluide parfait : Z(0) + U t.
    Z_parfait = d["Z"][0] + U0 * d["t"]
    # Figures de base : champs finaux, Hovmöller (avec la trajectoire en fluide parfait), bilan.
    taches = [
        ("Champs finaux", "Vorticité, lignes de courant dans le repère du vortex, vitesse et pression à la fin : la "
                          "vorticité s'est concentrée et un sillage est apparu.",
         lambda: viz.plot_axisymmetric_fields(snaps[-1], Z_ring=Z_s[-1], U_ring=U_fin, r_max=max(2.4, r_vue))),
        ("Hovmöller z–t", "Trace du vortex dans le plan z–t ; tirets : trajectoire du vortex de Hill en fluide "
                          "parfait (vitesse U constante).",
         lambda: viz.plot_hovmoller(res["profiles"], tracks=[(d["t"], d["Z"], "centroïde simulé")],
                                    theory_tracks=[(d["t"], Z_parfait, "Hill en fluide parfait (U t)")])),
        ("Bilan d'énergie", "Énergie, impulsion et enstrophie : E(t) suit E₀ − ν∫Ω dt.",
         lambda: viz.plot_energy_enstrophy_budget(res)),
    ]
    # Figures optionnelles.
    if p["dissipation"]:
        taches.append(("Dissipation Φ", "Taux local de dissipation aux trois instantanés : d'abord concentré à la "
                                        "surface de la sphère, où la vorticité est discontinue.",
                       lambda: viz.plot_dissipation_map(snaps, Z_s, r_max=max(2.2, r_vue))))
    if p["isosurfaces"]:
        taches.append(("Isosurfaces 3D", "Isosurfaces des critères Q et λ₂ de l'instantané final.",
                       lambda: v3.plot_isosurfaces(snaps[-1], Z_s[-1], half_width=1.6 * p["hill_a"],
                                                   r_max=1.6 * p["hill_a"])))
    if p["volume"]:
        taches.append(("Rendu volumique", "Enstrophie ½ω² reconstruite par révolution (début et fin).",
                       lambda: ns.rendu_volumique([snaps[0], snaps[-1]], [Z_s[0], Z_s[-1]], 1.8 * p["hill_a"],
                                                  1.6 * p["hill_a"], "Vortex de Hill — rendu volumique de ½ω²")))
    # Tracés.
    figures = c.tracer(taches, prog, 0.75, 0.92 if p["gif"] else 1.0)
    # Ralentissement relatif par rapport à la vitesse de Hill.
    ralenti = 100.0 * (1.0 - U_fin / U0)
    # Métriques du vortex de Hill.
    metriques = [
        c.metrique("Vitesse finale U", f"{U_fin:.4f}", f"Hill (fluide parfait) : {U0:.4f}",
                   aide("Vortex de Hill"), c.sous_echantillon(res["U_sim"])),
        c.metrique("Ralentissement", f"{ralenti:.1f} %", "1 − U_final/U"),
        c.metrique("Énergie dissipée", f"{perte:.1f} %", None, "1 − E(t_end)/E(0).",
                   c.sous_echantillon(d["energy"] / d["energy"][0])),
        c.metrique("Résidu du bilan d'énergie", f"{residu:.1e} E₀", None, "Dissipation numérique (≈ 0 attendu)."),
        c.metrique("Circulation Γ = 5Ua", f"{5.0 * U0 * p['hill_a']:.3f}", f"finale : {d['circulation'][-1]:.3f}"),
        c.metrique("Rayon R final", f"{d['R'][-1]:.3f}", f"initial : {d['R'][0]:.3f}"),
        c.metrique("Qualité numérique", f"{100 * parasite:.2f} %", "vorticité négative parasite",
                   "−min ω / max ω à la fin ; moins de 0,2 % est négligeable."),
        c.metrique("Grille", f"{grille.Nr} × {grille.Nz}", f"{res['solver'].n_steps} pas — {res['wall_time']:.1f} s"),
    ]
    # Interprétation chiffrée.
    notes = [f"Le vortex de Hill, qui avancerait à U = {U0:g} en fluide parfait, a ralenti de **{ralenti:.1f} %** ; la "
             f"viscosité a dissipé **{perte:.1f} %** de son énergie. Le rayon de Saffman R = √(∬ωr²/∬ω) passe de "
             f"{d['R'][0]:.3f} à {d['R'][-1]:.3f}."]
    # Courbes interactives : vitesse simulée et vitesse de Hill, bilan d'énergie.
    df_U = pd.DataFrame({"t": d["t"], "U simulée": res["U_sim"], "U de Hill": np.full_like(d["t"], U0)})
    courbes = [c.courbe("Vitesse U(t) et vitesse de Hill en fluide parfait", df_U, ["U simulée", "U de Hill"], "U",
                        couleurs=viz.SERIES[:2]), ns.courbe_bilan(d)]
    # Tableau des diagnostics.
    donnees = pd.DataFrame({"t": d["t"], "Z": d["Z"], "R": d["R"], "U_sim": res["U_sim"], "circulation": d["circulation"],
                            "energy": d["energy"], "dissipation": d["dissipation"], "max_div": d["max_div"]})
    # Éléments de la sortie propres au vortex de Hill.
    return {"metriques": metriques, "notes": notes, "figures": figures, "courbes": courbes,
            "comparables": _comparables(d, res["U_sim"]), "donnees": donnees, "fichier_csv": "hill_diagnostics.csv"}


def _comparables(d, U_sim):
    """Séries comparables d'un calcul à l'autre (onglet « Comparer »)."""
    # Vitesse, énergie relative, distance parcourue et rayon en fonction du temps.
    return [{"nom": "Vitesse U(t)", "x": d["t"], "y": U_sim, "x_label": "t", "y_label": "U"},
            {"nom": "Énergie E(t)/E₀", "x": d["t"], "y": d["energy"] / d["energy"][0], "x_label": "t", "y_label": "E/E₀"},
            {"nom": "Distance parcourue Z(t) − Z(0)", "x": d["t"], "y": d["Z"] - d["Z"][0], "x_label": "t",
             "y_label": "Z − Z₀"},
            {"nom": "Rayon R(t)", "x": d["t"], "y": d["R"], "x_label": "t", "y_label": "R"}]
