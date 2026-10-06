"""
appli/volets/filaments.py — Volet « filaments 3D » : anneaux en fluide parfait par la méthode lagrangienne
(src/vortex_particles.py) — saute-mouton, collisions, anneau isolé, configuration sur mesure, cœur épais.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# tempfile : dossier temporaire (GIF écrit par matplotlib) ; time : durée des tracés ; Path : chemins.
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from appli import calibration
from appli import commun as c
from appli.glossaire import aide
from src import theory
from src import visualization as viz
from src import visualization_3d as v3
from src import vortex_particles as vp

# Identité du volet, résumé et présentation.
CLE = "filaments"
LIBELLE = "🧵 Filaments 3D (fluide parfait)"
RESUME = ("Méthode lagrangienne 3D : saute-mouton sans fin, collisions frontale et décalée, anneau isolé, "
          "configuration sur mesure ; vues 3D animées et orientables à la souris.")
PRESENTATION = ("Méthode lagrangienne 3D (`vp.VortexFilamentSimulation`) : chaque anneau est un filament fermé dont "
                "les nœuds sont transportés par la loi de Biot–Savart régularisée (noyau de Rosenhead–Moore), "
                "intégrée en Runge–Kutta 4 avec remaillage. En fluide parfait, impulsion et énergie se conservent.")
# Formule clé : loi de Biot–Savart régularisée.
FORMULE = (r"\mathbf{u}(\mathbf{x}) = -\frac{\Gamma}{4\pi}\oint \frac{(\mathbf{x}-\mathbf{x}')\times d\mathbf{l}'}"
           r"{\left(|\mathbf{x}-\mathbf{x}'|^2 + \delta^2\right)^{3/2}}")
THEORIE = [("Fondements théoriques", "1.6"), ("Fondements théoriques", "4.1")]
LIMITES = [
    "Fluide parfait : ni viscosité ni diffusion des cœurs ; les anneaux ne fusionnent jamais.",
    "Pas de reconnexion des filaments : en dessous d'un écart d'environ 2δ, la régularisation domine et la "
    "dynamique n'est plus physique (zone grisée des diagnostics de collision).",
    "Cœur représenté par un seul filament régularisé (sauf « cœur épais ») : pas d'instabilité de Widnall, qui "
    "exige la structure interne du cœur.",
    "Coût en O(M²) pour M nœuds (calcul numpy vectorisé, sans arbre multipolaire).",
]
INSTANTANE = False
# Configurations (type de calcul) : clé → libellé.
CONFIGURATIONS = {
    "saute_mouton": "Saute-mouton (deux anneaux de même sens)",
    "collision": "Collision frontale (sens opposés)",
    "collision_decalee": "Collision décalée (dynamique 3D)",
    "anneau_isole": "Anneau isolé (validation)",
    "sur_mesure": "Sur mesure (anneau B incliné)",
    "coeur_epais": "Anneau à cœur épais (multi-filaments)",
}
# Durée simulée par défaut de chaque configuration.
DUREES = {"saute_mouton": 10.0, "collision": 12.0, "collision_decalee": 10.0, "anneau_isole": 5.0, "sur_mesure": 8.0,
          "coeur_epais": 3.0}
# Préfixe des clés de ses widgets (réglages enregistrés avec chaque calcul, puis rechargeables).
PREFIXE = "fi_"
# Réglages communs aux scénarios guidés.
_BASE = {"fi_delta": 0.1, "fi_N_std": 64, "fi_dt": 0.05}
SCENARIOS = {
    # Scénario 1 : saute-mouton avec une légère ondulation 3D.
    "saute_mouton": {
        "titre": "Saute-mouton en fluide parfait",
        "reglages": dict(_BASE, fi_sc="saute_mouton", fi_sep_sm=0.6, fi_pert=0.02, fi_mode=5, fi_T_saute_mouton=10.0),
        "observer": "Les anneaux se dépassent périodiquement (trois traversées en 10 unités de temps), en suivant le "
                    "modèle coaxial à quelques millièmes près ; impulsion et énergie restent conservées.",
        "question": {"texte": "les anneaux vont-ils finir par fusionner, comme en fluide visqueux ?",
                     "choix": ["Oui", "Non"], "reponse": "Non",
                     "explication": "Sans viscosité, rien n'épaissit les cœurs : le saute-mouton se répète indéfiniment "
                                    "(seules des instabilités 3D pourraient l'interrompre)."},
    },
    # Scénario 2 : ondulation plus forte, calcul plus long.
    "instabilite": {
        "titre": "Ondulation amplifiée (amplitude 0,05, t = 20)",
        "reglages": dict(_BASE, fi_sc="saute_mouton", fi_sep_sm=0.6, fi_pert=0.05, fi_mode=5, fi_N_std=96,
                         fi_T_saute_mouton=20.0),
        "observer": "L'ondulation azimutale de mode 5 se déforme au fil des traversées : la dynamique devient pleinement "
                    "tridimensionnelle. Faites tourner la vue 3D pour suivre les déformations.",
    },
    # Scénario 3 : collision frontale.
    "collision": {
        "titre": "Collision frontale",
        "reglages": dict(_BASE, fi_sc="collision", fi_sep_col=3.0, fi_T_collision=12.0),
        "observer": "En approchant du plan de symétrie, les anneaux ralentissent et s'élargissent, comme un anneau qui "
                    "approche une paroi (méthode des images).",
        "question": {"texte": "que devient le rayon des anneaux en approchant l'un de l'autre ?",
                     "choix": ["Il diminue", "Il reste constant", "Il augmente"], "reponse": "Il augmente",
                     "explication": "Chaque anneau est étiré par le champ de l'autre ; comme l'impulsion πρR²Γ se "
                                    "conserve, ils s'élargissent en ralentissant."},
    },
    # Scénario 4 : collision décalée.
    "collision_decalee": {
        "titre": "Collision décalée (axes parallèles écartés de 0,9)",
        "reglages": dict(_BASE, fi_sc="collision_decalee", fi_sep_col=3.0, fi_dec=0.9, fi_T_collision_decalee=10.0),
        "observer": "Les anneaux s'inclinent et s'étirent de façon dissymétrique : aucun modèle axisymétrique ne décrit "
                    "cette dynamique 3D.",
    },
    # Scénario 5 : anneau isolé, validation de la vitesse.
    "anneau_isole": {
        "titre": "Anneau isolé : validation de la vitesse",
        "reglages": dict(_BASE, fi_sc="anneau_isole", fi_pert_iso=0.0, fi_mode_iso=3, fi_N_std=128,
                         fi_T_anneau_isole=5.0),
        "observer": "Avec 2πR/N ≤ δ/2, la vitesse mesurée doit égaler la forme close exacte de Rosenhead–Moore, "
                    "U = Γ/(2π√(4R² + δ²))[K(m) − E(m)], à moins de 1 % près.",
        "question": {"texte": "la vitesse mesurée, comparée à la formule exacte, sera juste…",
                     "choix": ["à moins de 1 % près", "à environ 10 % près", "à plus de 30 % près"],
                     "reponse": "à moins de 1 % près",
                     "explication": "La discrétisation intègre exactement le noyau sur chaque segment : l'erreur décroît "
                                    "comme (Δs/δ)² et reste inférieure au pour cent pour N = 128."},
    },
    # Scénario 6 : configuration sur mesure.
    "sur_mesure": {
        "titre": "Sur mesure : petit anneau incliné qui rattrape le grand",
        "reglages": dict(_BASE, fi_sc="sur_mesure", fi_RB=0.7, fi_GB=1.0, fi_dx=0.3, fi_dz=0.8, fi_incl=15,
                         fi_T_sur_mesure=8.0),
        "observer": "Le petit anneau incliné, plus rapide, rattrape le grand ; leur interaction n'est plus coaxiale : "
                    "les deux anneaux se déforment et basculent.",
    },
    # Scénario 7 : cœur épais à 7 filaments.
    "coeur_epais": {
        "titre": "Anneau à cœur épais (7 filaments)",
        "reglages": {"fi_sc": "coeur_epais", "fi_sigma": 0.15, "fi_cour": 1, "fi_N_ce": 48, "fi_T_coeur_epais": 3.0,
                     "fi_dt": 0.04},
        "observer": "Le cœur gaussien est représenté par 7 filaments de circulations pondérées : ils tournent les uns "
                    "autour des autres en avançant. La vue « Tubes » en écorché montre leur organisation.",
    },
}


def controles():
    """Widgets du volet ; renvoie le dictionnaire des paramètres."""
    st.caption("Anneaux de rayon 1 et de circulation ±1 ; fluide parfait (pas de viscosité).")
    p = {}
    # Configuration : détermine les paramètres affichés ensuite.
    p["configuration"] = c.menu("Configuration", list(CONFIGURATIONS), "saute_mouton", "fi_sc",
                                format_func=CONFIGURATIONS.get)
    conf = p["configuration"]
    # Paramètres propres à la configuration (clés distinctes quand les valeurs par défaut diffèrent).
    if conf == "saute_mouton":
        p["sep"] = c.curseur("Écart initial des anneaux", 0.3, 1.5, 0.6, 0.05, "fi_sep_sm")
        p["perturbation"] = c.curseur("Ondulation azimutale (amplitude)", 0.0, 0.05, 0.02, 0.005, "fi_pert",
                                      "Petite déformation qui rend la dynamique tridimensionnelle.")
        p["mode"] = c.curseur("Nombre d'ondulations (mode)", 2, 8, 5, 1, "fi_mode")
    elif conf in ("collision", "collision_decalee"):
        p["sep"] = c.curseur("Distance initiale des anneaux", 1.5, 5.0, 3.0, 0.1, "fi_sep_col")
        if conf == "collision_decalee":
            p["decalage"] = c.curseur("Décalage latéral des axes", 0.1, 1.5, 0.9, 0.05, "fi_dec")
    elif conf == "anneau_isole":
        p["perturbation"] = c.curseur("Ondulation azimutale (amplitude)", 0.0, 0.1, 0.0, 0.01, "fi_pert_iso",
                                      "Ondes de Kelvin : l'ondulation tourne autour de l'anneau sans s'amplifier.")
        p["mode"] = c.curseur("Nombre d'ondulations (mode)", 2, 8, 3, 1, "fi_mode_iso")
    elif conf == "sur_mesure":
        st.markdown("**Anneau B** (l'anneau A : R = 1, Γ = 1, centré à l'origine, axe z)")
        p["RB"] = c.curseur("Rayon R_B", 0.4, 1.6, 0.7, 0.05, "fi_RB")
        p["GB"] = c.curseur("Circulation Γ_B", -2.0, 2.0, 1.0, 0.1, "fi_GB")
        p["dx"] = c.curseur("Décalage latéral x_B", -1.0, 1.0, 0.3, 0.05, "fi_dx")
        p["dz"] = c.curseur("Position axiale z_B", -3.0, 3.0, 0.8, 0.1, "fi_dz")
        p["inclinaison"] = c.curseur("Inclinaison de l'axe de B (°)", 0, 60, 15, 1, "fi_incl")
    else:
        p["sigma"] = c.curseur("Rayon du cœur gaussien σ", 0.08, 0.25, 0.15, 0.01, "fi_sigma")
        p["couronnes"] = c.curseur_liste("Couronnes de filaments", [1, 2], 1, "fi_cour",
                                         "1 → 7 filaments ; 2 → 19 filaments (≈ 7× plus lent).")
    # Réglages communs : régularisation, nœuds, durée, pas de temps.
    if conf != "coeur_epais":
        p["delta"] = c.curseur("Rayon de régularisation δ", 0.03, 0.30, 0.10, 0.01, "fi_delta",
                               aide("Rosenhead–Moore (régularisation de)"))
    # Nœuds par filament : clé propre au cœur épais (valeur par défaut différente).
    cle_n = "fi_N_ce" if conf == "coeur_epais" else "fi_N_std"
    p["n_noeuds"] = c.curseur("Nœuds par filament N", 32, 192, 48 if conf == "coeur_epais" else 64, 16, cle_n,
                              "Précision ≈ 1 % si 2πR/N ≤ δ/2. Le coût varie en N².")
    p["t_end"] = c.curseur("Durée simulée t_end", 1.0, 40.0, DUREES[conf], 0.5, f"fi_T_{conf}")
    p["dt"] = c.curseur_liste("Pas de temps dt", [0.01, 0.02, 0.025, 0.04, 0.05, 0.08, 0.1], 0.05, "fi_dt",
                              "dt ≤ 0,05 conseillé pour δ = 0,1.")
    with st.expander("Options et rendus"):
        p["volume_coeur"] = c.interrupteur("Conserver le volume du cœur (δ²L constant)", False, "fi_cv",
                                           "Les tubes s'amincissent quand les filaments s'étirent.")
        p["energie_fine"] = c.interrupteur("Énergie calculée à chaque sauvegarde (plus lent)", False, "fi_ener",
                                           "Par défaut, l'énergie (coût en O(M²)) n'est calculée qu'une sauvegarde "
                                           "sur dix.")
        p["volume"] = c.interrupteur("Rendu volumique de l'enstrophie (lent)", False, "fi_vol")
        p["gif"] = c.interrupteur("Animation GIF téléchargeable", False, "fi_gif",
                                  "L'animation 3D interactive est toujours disponible ; le GIF sert à l'exporter.")
    return p


def nombre_noeuds(p):
    """Nombre total de nœuds M (1 + 3c(c + 1) filaments pour un cœur épais à c couronnes)."""
    if p["configuration"] == "coeur_epais":
        return (1 + 3 * p["couronnes"] * (p["couronnes"] + 1)) * p["n_noeuds"]
    return (1 if p["configuration"] == "anneau_isole" else 2) * p["n_noeuds"]


def verifier(p):
    """Garde-fous et durée estimée."""
    alertes = []
    if p["configuration"] != "coeur_epais" and 2 * np.pi / p["n_noeuds"] > p["delta"]:
        alertes.append(f"Discrétisation grossière : 2πR/N = {2 * np.pi / p['n_noeuds']:.3f} > δ = {p['delta']:.2f} ; "
                       "augmentez N pour une précision de l'ordre du pour cent.")
    # Taille du calcul (nœuds) et durée estimée des figures.
    M = nombre_noeuds(p)
    # Durée des figures : tracés de base puis options ; l'énergie à chaque sauvegarde coûte ≈ 10 % de plus.
    figures = 9.0 + 15.0 * p["volume"] + 10.0 * p["gif"]
    # Coût en O(M²) par pas de temps.
    est = calibration.estimation("filaments", M**2, p["t_end"] / p["dt"] * (1.1 if p["energie_fine"] else 1.0),
                                 figures, f"{M} nœuds")
    return alertes, est


def etiquette(p):
    """Description courte d'un calcul."""
    # Configuration, nœuds, pas de temps, durée, régularisation et écart éventuels.
    conf = p["configuration"]
    texte = f"{CONFIGURATIONS[conf].split(' (')[0]} · N {p['n_noeuds']} · dt {p['dt']:g} · t {p['t_end']:g}"
    if conf != "coeur_epais":
        texte += f" · δ {p['delta']:g}"
    if "sep" in p:
        texte += f" · écart {p['sep']:g}"
    if p.get("perturbation"):
        texte += f" · ondulation {p['perturbation']:g}"
    return texte


def raffinement(p, facteur):
    """Calcul de convergence : nœuds × facteur et pas de temps ÷ facteur (coût ≈ facteur³)."""
    # Copie des paramètres : nœuds multipliés, pas de temps divisé, rendus coupés.
    p2 = dict(p, n_noeuds=int(round(p["n_noeuds"] * facteur)), dt=p["dt"] / facteur, volume=False, gif=False)
    return p2, f"N = {p['n_noeuds']} → {p2['n_noeuds']}, dt = {p['dt']:g} → {p2['dt']:.4g} (coût ≈ ×{facteur**3:.0f})"


def _filaments(p):
    """Filaments initiaux, modèle coaxial de référence éventuel et noms des anneaux."""
    # Configuration, nœuds par filament et durée.
    conf, N, T = p["configuration"], p["n_noeuds"], p["t_end"]
    modele, noms = None, ["anneau A (Γ > 0)", "anneau B (Γ < 0)"]
    # Saute-mouton : deux anneaux de même sens, modèle coaxial de référence.
    if conf == "saute_mouton":
        fil = vp.leapfrog_scenario(delta=p["delta"], separation=p["sep"], n_nodes=N,
                                   perturbation_amplitude=p["perturbation"], perturbation_mode=p["mode"])
        modele = theory.CoaxialRingsModel([1.0, 1.0], [0.0, p["sep"]], 1.0, p["delta"]).integrate(T, 801)
        noms = ["anneau A", "anneau B"]
    # Collision frontale : circulations opposées, modèle coaxial.
    elif conf == "collision":
        fil = vp.head_on_collision_scenario(delta=p["delta"], separation=p["sep"], n_nodes=N)
        modele = theory.CoaxialRingsModel([1.0, 1.0], [-0.5 * p["sep"], 0.5 * p["sep"]], [1.0, -1.0],
                                          p["delta"]).integrate(T, 801)
    elif conf == "collision_decalee":
        fil = vp.offset_collision_scenario(delta=p["delta"], separation=p["sep"], offset=p["decalage"], n_nodes=N)
    # Anneau isolé (éventuellement ondulé).
    elif conf == "anneau_isole":
        fil = vp.single_ring_scenario(delta=p["delta"], n_nodes=N, perturbation_mode=p["mode"],
                                      perturbation_amplitude=p["perturbation"])
        noms = ["anneau"]
    elif conf == "sur_mesure":
        # Anneau B : axe incliné d'un angle θ dans le plan x–z (normale (sin θ, 0, cos θ)).
        theta = np.radians(p["inclinaison"])
        fil = (vp.make_ring(R=1.0, Gamma=1.0, n_nodes=N, delta=p["delta"], ring_id=0)
               + vp.make_ring(R=p["RB"], Gamma=p["GB"], n_nodes=N, center=(p["dx"], 0.0, p["dz"]),
                              normal=(np.sin(theta), 0.0, np.cos(theta)), delta=p["delta"], ring_id=1))
        noms = ["anneau A", "anneau B"]
    # Cœur épais : couronnes de filaments de circulations gaussiennes.
    else:
        fil = vp.make_ring(1.0, 1.0, N, core_radius=p["sigma"], n_core_layers=p["couronnes"])
        noms = None
    return fil, modele, noms


def calculer(p, prog, complet=True):
    """Calcul complet ou léger (convergence) par la méthode des filaments."""
    # Configuration, durée et pas de temps.
    conf, T, dt = p["configuration"], p["t_end"], p["dt"]
    filaments, modele, noms = _filaments(p)
    # Simulation (volume du cœur conservé ou non) et nombre de pas.
    sim = vp.VortexFilamentSimulation(filaments, conserve_core_volume=p["volume_coeur"])
    n_pas = int(round(T / dt))
    # Environ 120 sauvegardes ; énergie (coût O(M²)) à chaque sauvegarde ou une sur dix.
    sauvegarde = max(1, n_pas // 120)

    def rappel(s):
        # Progression : 70 % de la barre pour le calcul.
        prog(0.7 * s.time / T, f"Filaments : t = {s.time:.2f} / {T:g} — {s.n_nodes} nœuds")

    # Intégration en temps (fonction du projet) ; h : diagnostics en tableaux numpy.
    hist = sim.run(T, dt, save_every=sauvegarde, energy_every=1 if p["energie_fine"] else 10, callback=rappel)
    h = hist.as_arrays()
    # Invariants : dérive de l'impulsion (rapportée à πR²|Γ| si l'impulsion totale est nulle) et de l'énergie.
    # Dérive de l'impulsion (norme de P(t) − P(0)).
    P = h["impulse"]
    derive_P = float(np.linalg.norm(P - P[0], axis=1).max() / max(float(np.linalg.norm(P[0])), np.pi))
    # Dérive de l'énergie sur les valeurs calculées (np.isfinite écarte les NaN).
    E = h["energy"][np.isfinite(h["energy"])]
    derive_E = float(np.abs(E - E[0]).max() / abs(E[0])) if E.size > 1 else float("nan")
    # Grandeurs, notes et métriques propres à la configuration.
    grandeurs, notes, metriques_spec = _analyse(conf, p, h, modele)
    # Mesures de calibration ; calcul léger (convergence) : on s'arrête là.
    mesures = {"t_calcul": hist.wall_time, "n_pas": n_pas, "taille": nombre_noeuds(p) ** 2}
    sortie = {"grandeurs": grandeurs, "mesures": mesures, "type_cout": CLE}
    if not complet:
        return sortie
    # Vues : suivi des anneaux (saute-mouton, cœur épais, anneau isolé) ou boîte englobant tout le mouvement.
    suivre = conf in ("saute_mouton", "coeur_epais", "anneau_isole")
    if suivre:
        fenetre = 2.8 if conf == "coeur_epais" else 3.2
    else:
        # viz._frame_bounds : boîte cubique englobant toutes les sauvegardes ; plus grande dimension.
        bas, haut = viz._frame_bounds(hist.frames)
        fenetre = float((haut - bas).max())
    # Rayon des tubes (demi-δ pour le cœur épais), instants affichés et titre.
    rayon_tubes = 0.5 if conf == "coeur_epais" else 1.0
    idx = [0, len(hist.frames) // 2, len(hist.frames) - 1]
    titre = CONFIGURATIONS[conf]
    # Chronomètre des tracés.
    debut_figures = time.perf_counter()
    # Figures de base : instantanés 3D et tubes de vorticité.
    taches = [
        ("Instantanés 3D", "Filaments à quatre instants régulièrement espacés.",
         lambda: viz.plot_filament_snapshots(hist, n=4, follow=suivre, window=fenetre, labels=noms,
                                             title=f"{titre} — instantanés 3D")),
        ("Tubes de vorticité", "Tubes lissés autour des filaments : rayon ∝ δ, couleur = circulation Γ"
                               + (" ; écorché qui révèle les filaments du cœur." if conf == "coeur_epais" else "."),
         lambda: v3.plot_vortex_tubes([hist.frames[i] for i in idx], hist.gammas, [hist.deltas[i] for i in idx],
                                      [hist.times[i] for i in idx], radius_scale=rayon_tubes, follow=suivre,
                                      window=fenetre, cutaway=110.0 if conf == "coeur_epais" else None,
                                      title=f"{titre} — tubes de vorticité")),
    ]
    # Diagnostics comparés au modèle coaxial (configurations axisymétriques).
    if conf == "saute_mouton":
        taches.append(("Diagnostics", "Positions et rayons des anneaux (trait plein) comparés au modèle coaxial "
                                      "régularisé (tirets) ; dérive des invariants.",
                       lambda: viz.plot_leapfrog_diagnostics(hist, modele)))
    elif conf == "collision":
        taches.append(("Diagnostics", "Rayons et positions comparés au modèle coaxial ; zone grisée : écart < 2δ, "
                                      "où la régularisation domine.",
                       lambda: viz.plot_collision_diagnostics(hist, modele, delta=p["delta"])))
    # Hovmöller des filaments : densité axiale de vorticité λ(z, t), une ligne par sauvegarde.
    z_grille = np.linspace(np.nanmin(h["axial"]) - 1.2, np.nanmax(h["axial"]) + 1.2, 300)
    densite = np.array([vp.vorticity_line_density(fr, hist.gammas, dl, z_grille) for fr, dl in zip(hist.frames, hist.deltas)])
    pistes = [(h["t"], h["axial"][:, i], f"anneau {'AB'[i] if i < 2 else i + 1}") for i in range(h["axial"].shape[1])]
    taches.append(("Hovmöller z–t", "Densité axiale de vorticité λ(z, t) et positions des anneaux.",
                   lambda: viz.plot_hovmoller_density(z_grille, h["t"], densite, tracks=pistes, title=f"{titre} — Hovmöller z–t")))
    if p["volume"]:
        def rendu_volumique():
            # « Splatting » des segments sur une grille 3D puis lancer de rayons, à trois instants.
            images, titres = [], []
            for i in idx:
                vol, origine, pas = v3.filament_enstrophy_volume(hist.frames[i], hist.gammas, hist.deltas[i], resolution=96)
                images.append(v3.volume_render(vol, origine, pas, elev=15, azim=-35, image_size=(300, 240), density=10.0))
                titres.append(f"t = {hist.times[i]:.1f}")
            return v3.plot_volume_renderings(images, titres, title=f"{titre} — rendu volumique de ½|ω|²")
        taches.append(("Rendu volumique", "Enstrophie ½|ω|² reconstruite à partir des filaments.", rendu_volumique))
    # Tracés (barre de 72 % à 90 % ou 100 %).
    figures = c.tracer(taches, prog, 0.72, 0.9 if p["gif"] else 1.0)
    # GIF téléchargeable (optionnel).
    gif = None
    if p["gif"]:
        prog(0.9, "Animation GIF…")
        # matplotlib écrit l'animation dans un fichier : dossier temporaire supprimé à la sortie du « with ».
        with tempfile.TemporaryDirectory() as dossier, c.VERROU:
            chemin = viz.animate_filaments_3d(hist, str(Path(dossier) / "filaments.gif"), fps=15, dpi=80, follow=suivre,
                                              window=fenetre if suivre else None, title=titre,
                                              stride=max(1, len(hist.frames) // 40),
                                              rotate_deg=40.0 if conf == "collision_decalee" else 0.0)
            gif = {"octets": Path(chemin).read_bytes(), "fichier": f"filaments_{conf}.gif"}
    # Durée des tracés.
    mesures["t_figures"] = time.perf_counter() - debut_figures
    # Métriques communes (durée, nœuds, invariants) puis propres à la configuration.
    metriques = [
        c.metrique("Durée du calcul", f"{hist.wall_time:.1f} s", f"{n_pas} pas de RK4"),
        c.metrique("Nœuds", f"{hist.n_nodes[0]} → {hist.n_nodes[-1]}", None,
                   "Le remaillage ajoute des nœuds quand les filaments s'étirent."),
        c.metrique("Dérive de l'impulsion", f"{derive_P:.1e}", None,
                   "max |P(t) − P(0)| rapporté à |P(0)| (ou à πR²Γ si l'impulsion totale est nulle). "
                   + (aide("Impulsion I") or ""), c.sous_echantillon(np.linalg.norm(P - P[0], axis=1))),
        c.metrique("Dérive de l'énergie", f"{derive_E:.1e}", None,
                   "max |E(t) − E(0)| / E(0) ; énergie calculée " + ("à chaque sauvegarde." if p["energie_fine"]
                                                                    else "une sauvegarde sur dix.")),
    ] + metriques_spec
    # Courbes interactives (format long) et séries comparables.
    # Noms des anneaux (A, B…) et courbes interactives au format long.
    noms_anneaux = [f"anneau {'AB'[i] if i < 2 else i + 1}" for i in range(h["axial"].shape[1])]
    df_z = pd.concat([pd.DataFrame({"t": h["t"], "valeur": h["axial"][:, i], "série": f"z {nom}"})
                      for i, nom in enumerate(noms_anneaux)])
    df_R = pd.concat([pd.DataFrame({"t": h["t"], "valeur": h["radius"][:, i], "série": f"R {nom}"})
                      for i, nom in enumerate(noms_anneaux)])
    if modele is not None:
        df_z = pd.concat([df_z] + [pd.DataFrame({"t": modele["t"], "valeur": modele["Z"][i], "série": f"z {nom} (modèle)"})
                                   for i, nom in enumerate(noms_anneaux)])
        df_R = pd.concat([df_R] + [pd.DataFrame({"t": modele["t"], "valeur": modele["R"][i], "série": f"R {nom} (modèle)"})
                                   for i, nom in enumerate(noms_anneaux)])
    # Courbes interactives et séries comparables entre calculs.
    courbes = [c.courbe("Positions axiales z(t)", df_z, "valeur", "z"), c.courbe("Rayons R(t)", df_R, "valeur", "R")]
    comparables = [{"nom": "Rayon de l'anneau A", "x": h["t"], "y": h["radius"][:, 0], "x_label": "t", "y_label": "R"},
                   {"nom": "Position axiale de l'anneau A", "x": h["t"], "y": h["axial"][:, 0], "x_label": "t",
                    "y_label": "z"}]
    if h["axial"].shape[1] == 2:
        comparables.insert(0, {"nom": "Écart axial z_A − z_B", "x": h["t"], "y": h["axial"][:, 0] - h["axial"][:, 1],
                               "x_label": "t", "y_label": "z_A − z_B"})
    # Tableau des diagnostics : une colonne par grandeur et par anneau.
    donnees = pd.DataFrame({"t": h["t"], "n_noeuds": h["n_nodes"], "energie": h["energy"], "P_z": P[:, 2]})
    for i, nom in enumerate(noms_anneaux):
        donnees[f"z_{nom.split()[-1]}"] = h["axial"][:, i]
        donnees[f"R_{nom.split()[-1]}"] = h["radius"][:, i]
    # Éléments de la sortie ; données des vues interactives (Plotly).
    sortie.update({"metriques": metriques, "notes": notes, "figures": figures, "gif": gif, "courbes": courbes,
                   "comparables": comparables, "donnees": donnees, "fichier_csv": f"filaments_{conf}.csv",
                   "animation": {"type": "filaments", "hist": hist, "etiquettes": noms, "suivre": suivre,
                                 "fenetre": fenetre},
                   "vue3d": {"type": "tubes", "hist": hist, "rayon_tubes": rayon_tubes, "fenetre": fenetre}})
    return sortie


def _analyse(conf, p, h, modele):
    """Grandeurs (convergence), notes et métriques propres à la configuration."""
    # Valeurs par défaut (configurations sans analyse propre).
    grandeurs, notes, metriques = {}, [], []
    if conf == "saute_mouton":
        # Traversées simulées et prévues par le modèle coaxial ; rayons extrêmes.
        tr = c.croisements(h["t"], h["axial"][:, 0] - h["axial"][:, 1])
        tr_m = c.croisements(modele["t"], modele["Z"][0] - modele["Z"][1])
        grandeurs = {"1ʳᵉ traversée (t)": tr[0] if tr else float("nan"), "Rayon minimal": float(np.nanmin(h["radius"])),
                     "Rayon maximal": float(np.nanmax(h["radius"]))}
        metriques = [c.metrique("Traversées (filaments)", str(len(tr)), f"t = {c.liste_instants(tr)}", aide("Saute-mouton")),
                     c.metrique("Traversées (modèle coaxial)", str(len(tr_m)), f"t = {c.liste_instants(tr_m)}",
                                aide("Modèle coaxial")),
                     c.metrique("Rayon min / max", f"{np.nanmin(h['radius']):.3f} / {np.nanmax(h['radius']):.3f}",
                                f"modèle : {modele['R'].min():.3f} / {modele['R'].max():.3f}")]
        notes.append(f"Traversées simulées : **{c.liste_instants(tr)}** ; modèle coaxial : **{c.liste_instants(tr_m)}**.")
    # Collision : rayon final comparé au modèle coaxial.
    elif conf == "collision":
        grandeurs = {"Rayon final": float(h["radius"][-1, 0])}
        metriques = [c.metrique("Rayon final", f"{h['radius'][-1, 0]:.3f}", f"modèle coaxial : {modele['R'][0, -1]:.3f}",
                                aide("Modèle coaxial"))]
    elif conf == "anneau_isole":
        # Vitesse mesurée : pente de la position axiale (np.polyfit de degré 1) comparée à la forme close exacte.
        U_mes = float(np.polyfit(h["t"], h["axial"][:, 0], 1)[0])
        U_rm = float(theory.rosenhead_moore_ring_speed(1.0, 1.0, p["delta"], exact=True))
        ecart = 100.0 * (U_mes / U_rm - 1.0)
        grandeurs = {"Vitesse mesurée U": U_mes}
        metriques = [c.metrique("Vitesse mesurée U", f"{U_mes:.5f}", f"{ecart:+.2f} % vs forme close exacte",
                                aide("Rosenhead–Moore (régularisation de)")),
                     c.metrique("Rosenhead–Moore exacte", f"{U_rm:.5f}", "Γ/(2π√(4R²+δ²))[K(m) − E(m)]")]
        notes.append(f"La vitesse mesurée s'écarte de **{ecart:+.2f} %** de la forme close exacte de Rosenhead–Moore "
                     f"(2πR/N = {2 * np.pi / p['n_noeuds']:.3f} pour δ = {p['delta']:g}).")
    # Autres configurations : rayon final moyen.
    else:
        grandeurs = {"Rayon final moyen": float(np.nanmean(h["radius"][-1]))}
    return grandeurs, notes, metriques
