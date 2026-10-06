"""
appli/volets/saute_mouton.py — Volet « deux anneaux coaxiaux » (Navier–Stokes axisymétrique) : saute-mouton
visqueux et fusion des cœurs (circulations de même signe) ou collision frontale (circulations opposées).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# time : durée des tracés ; matplotlib (rendu « Agg ») : figure du critère de fusion ; numpy, pandas ; streamlit.
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from appli import commun as c  # noqa: E402
from appli.glossaire import aide  # noqa: E402
from appli.volets import navier_stokes as ns  # noqa: E402
from src import solver_2d_axisymmetric as ax  # noqa: E402
from src import theory  # noqa: E402
from src import visualization as viz  # noqa: E402
from src import visualization_3d as v3  # noqa: E402

# Identité du volet, résumé et présentation.
CLE = "saute_mouton"
LIBELLE = "🔁 Deux anneaux coaxiaux (Navier–Stokes)"
RESUME = ("Saute-mouton visqueux : l'anneau arrière traverse l'anneau avant, jusqu'à la fusion des cœurs ; ou "
          "collision frontale de deux anneaux de sens opposés.")
PRESENTATION = ("Deux anneaux coaxiaux de Lamb–Oseen (`ax.run_coaxial_rings`), suivis individuellement. Avec des "
                "circulations de même signe, l'anneau arrière, contracté et accéléré, traverse l'anneau avant "
                "(saute-mouton) ; la viscosité épaissit les cœurs jusqu'à leur fusion. Avec des circulations "
                "opposées, les anneaux se percutent. Le modèle coaxial en fluide parfait sert de référence.")
# Critère de fusion de deux tourbillons co-rotatifs (Meunier et al. 2002), évalué à chaque traversée.
FORMULE = (r"\text{fusion si}\quad \frac{b}{\sigma} \lesssim 4{,}2\text{–}4{,}5,\qquad b \approx |R_A - R_B|"
           r"\ \text{à la traversée},\qquad \sigma(t)^2 = a^2 + 4\nu t")
THEORIE = [("Fondements théoriques", "5.4"), ("Fondements théoriques", "2.4")]
LIMITES = [
    "Écoulement axisymétrique (anneaux coaxiaux, sans swirl) : les instabilités 3D sont absentes.",
    "Suivi des cœurs par maxima de vorticité : après une fusion ou un contact, un seul pic subsiste et les courbes "
    "de suivi s'arrêtent.",
    "Domaine périodique en z (longueur 10) et paroi glissante en r = 8R ; écoulement laminaire.",
    "Seuil de fusion b/σ ≈ 4,2–4,5 établi pour deux tourbillons rectilignes identiques : indicatif pour des "
    "anneaux, et moins précis pour des cœurs de tailles différentes.",
]
INSTANTANE = False
# Préfixe des clés de ses widgets (réglages enregistrés avec chaque calcul, puis rechargeables).
PREFIXE = "sm_"
# Réglages de départ communs aux scénarios (deux anneaux identiques).
_BASE = {"sm_RA": 1.0, "sm_aA": 0.15, "sm_GA": 1.0, "sm_RB": 1.0, "sm_aB": 0.15, "sm_GB": 1.0, "sm_sep": 0.6,
         "sm_Re": 2000, "sm_T": 5.0, "sm_dr": 0.03}
SCENARIOS = {
    # Scénario 1 : réglages par défaut (cœurs épais, fusion dès la première traversée).
    "rapide": {
        "titre": "Saute-mouton rapide : fusion à la 1ʳᵉ traversée",
        "reglages": dict(_BASE),
        "observer": "Une seule traversée vers t ≈ 1,5, puis fusion vers t ≈ 2,2 : à la traversée, l'écart des cœurs ne "
                    "vaut qu'environ 3 rayons de cœur, sous le seuil de fusion (≈ 4). Voir la figure « Critère de "
                    "fusion ».",
        "question": {"texte": "combien de fois les anneaux vont-ils se dépasser ?",
                     "choix": ["Plusieurs fois", "Une fois, puis ils fusionnent", "Jamais"],
                     "reponse": "Une fois, puis ils fusionnent",
                     "explication": "À la traversée, les deux cœurs de même signe forment une paire co-rotative trop "
                                    "serrée (b/σ ≈ 3 < 4) : ils fusionnent au lieu de se séparer."},
    },
    # Scénario 2 : cas du § 5.4 du rapport (cœurs fins, Re = 3000, maillage fin).
    "rapport": {
        "titre": "Cas du rapport : deux traversées puis fusion (≈ 4 min)",
        "reglages": dict(_BASE, sm_aA=0.10, sm_aB=0.10, sm_Re=3000, sm_T=12.0, sm_dr=0.02),
        "observer": "Deux traversées (t ≈ 1,45 et 3,92) puis fusion vers t ≈ 5,4 : b/σ ≈ 4,3 à la première traversée "
                    "(les anneaux survivent), ≈ 3,3 à la seconde (fusion). Cas étudié au § 5.4 du rapport.",
        "question": {"texte": "les anneaux survivront-ils à la première traversée ?",
                     "choix": ["Oui", "Non"], "reponse": "Oui",
                     "explication": "Avec des cœurs plus fins (a = 0,10), b/σ ≈ 4,3 dépasse le seuil à la première "
                                    "traversée ; la viscosité épaissit ensuite les cœurs et la seconde leur est fatale."},
    },
    # Scénario 3 : effet de la viscosité.
    "reynolds": {
        "titre": "Effet de la viscosité (Re = 1000)",
        "reglages": dict(_BASE, sm_Re=1000),
        "observer": "Plus visqueux : les cœurs s'épaississent plus vite, la fusion arrive plus tôt et l'énergie dissipée "
                    "augmente. Comparez avec le saute-mouton rapide dans l'onglet « Comparer ».",
    },
    # Scénario 4 : petit anneau rapide derrière un grand anneau.
    "tailles": {
        "titre": "Anneaux de tailles différentes (R = 0,8 derrière 1,2)",
        "reglages": dict(_BASE, sm_RA=0.8, sm_RB=1.2, sm_sep=0.8, sm_T=6.0),
        "observer": "Le petit anneau, plus rapide, rattrape le grand et passe à travers lui en se contractant, tandis que "
                    "le grand s'élargit et ralentit.",
    },
    # Scénario 5 : circulations opposées (collision frontale).
    "collision": {
        "titre": "Collision frontale (circulations opposées)",
        "reglages": dict(_BASE, sm_GB=-1.0, sm_sep=3.0, sm_T=8.0),
        "observer": "En approchant du plan de symétrie, les anneaux ralentissent et s'élargissent, comme un anneau qui "
                    "approche une paroi ; la vorticité de signes opposés s'annule ensuite par diffusion.",
        "question": {"texte": "que devient le rayon des anneaux à l'approche ?",
                     "choix": ["Il diminue", "Il reste constant", "Il augmente"], "reponse": "Il augmente",
                     "explication": "Chaque anneau est entraîné vers l'extérieur par le champ de son partenaire (méthode "
                                    "des images) : il s'élargit et ralentit, l'impulsion πρR²Γ restant constante."},
    },
}


def controles():
    """Widgets du volet ; renvoie le dictionnaire des paramètres."""
    st.caption("Anneau A en z = 2, anneau B devant lui. Unités : R₀ = 1, temps en R₀²/Γ₀.")
    p = {}
    # Deux colonnes : un anneau par colonne (curseurs compacts).
    col_a, col_b = st.columns(2)
    for col, nom, cle in ((col_a, "A (arrière)", "A"), (col_b, "B (avant)", "B")):
        with col:
            st.markdown(f"**Anneau {nom}**")
            p[f"R{cle}"] = c.curseur("R", 0.5, 1.5, 1.0, 0.05, f"sm_R{cle}", "Rayon de l'anneau.")
            p[f"a{cle}"] = c.curseur("a", 0.06, 0.35, 0.15, 0.01, f"sm_a{cle}", aide("Rayon de cœur a (σ)"))
            p[f"G{cle}"] = c.curseur("Γ", -2.0, 2.0, 1.0, 0.1, f"sm_G{cle}",
                                     "Même signe : saute-mouton ; signes opposés : collision frontale.")
    # Écart initial, Reynolds, durée et maillage.
    p["sep"] = c.curseur("Écart initial z_B − z_A", 0.3, 3.0, 0.6, 0.05, "sm_sep")
    p["Re"] = c.curseur_liste("Reynolds Re_Γ = max|Γ|/ν", ns.OPTIONS_RE[3:], 2000, "sm_Re", aide("Nombre de Reynolds Re_Γ"))
    p["t_end"] = c.curseur("Durée simulée t_end", 1.0, 20.0, 5.0, 0.5, "sm_T")
    p["dr"] = c.curseur_liste("Pas de maillage Δr (grossier → fin)", ns.OPTIONS_DR, 0.03, "sm_dr",
                              "Garder a/Δr ≥ 5 pour le plus fin des deux cœurs.")
    with st.expander("Diagnostics et rendus"):
        p["isosurfaces"] = c.interrupteur("Isosurfaces 3D (critère Q)", True, "sm_iso", aide("Critère Q"))
        p["dissipation"] = c.interrupteur("Carte de dissipation Φ", True, "sm_phi", aide("Dissipation Φ"))
        p["volume"] = c.interrupteur("Rendu volumique de l'enstrophie (lent)", False, "sm_vol")
        p["gif"] = c.interrupteur("Animation GIF téléchargeable", False, "sm_gif")
    return p


def anneaux(p):
    """Les deux anneaux au format de run_coaxial_rings (R, a, Gamma, z0)."""
    return [{"R": p["RA"], "a": p["aA"], "Gamma": p["GA"], "z0": 2.0},
            {"R": p["RB"], "a": p["aB"], "Gamma": p["GB"], "z0": 2.0 + p["sep"]}]


def verifier(p):
    """Garde-fous numériques et durée estimée."""
    # Cœur le plus fin (résolution) et plus grand rayon (taille du domaine).
    a_min, R_max = min(p["aA"], p["aB"]), max(p["RA"], p["RB"])
    figures = 8.0 + 3.0 * p["isosurfaces"] + 4.0 * p["dissipation"] + 15.0 * p["volume"] + 12.0 * p["gif"]
    # Estimation : domaine radial de 8 R_max, zone uniforme jusqu'à R_max + 1,2.
    est = ns.estimation(CLE, anneaux(p), p["Re"], p["t_end"], p["dr"], 8.0 * R_max, max(2.4, R_max + 1.2), figures)
    # Garde-fous de résolution, puis vérifications propres aux deux anneaux.
    alertes = ns.alertes_resolution(a_min, p["dr"], est["peclet"])
    if min(abs(p["GA"]), abs(p["GB"])) < 0.05:
        alertes.append("Circulation quasi nulle pour l'un des anneaux : choisissez |Γ| ≥ 0,1.")
    if p["sep"] < p["aA"] + p["aB"]:
        alertes.append("Les deux cœurs se recouvrent dès le départ (écart < a_A + a_B).")
    return alertes, est


def etiquette(p):
    """Description courte d'un calcul."""
    # Cœurs, écart, Reynolds, maillage, durée ; rayons et circulations s'ils diffèrent des valeurs par défaut.
    texte = f"a {p['aA']:g}/{p['aB']:g} · écart {p['sep']:g} · Re {p['Re']:g} · Δr {p['dr']:g} · t {p['t_end']:g}"
    if p["RA"] != 1.0 or p["RB"] != 1.0:
        texte += f" · R {p['RA']:g}/{p['RB']:g}"
    if p["GA"] != 1.0 or p["GB"] != 1.0:
        texte += f" · Γ {p['GA']:g}/{p['GB']:g}"
    return texte


def raffinement(p, facteur):
    """Paramètres du calcul de convergence (maillage affiné, sans figures optionnelles)."""
    # Copie des paramètres avec Δr affiné et figures optionnelles coupées.
    p2 = dict(p, dr=p["dr"] / facteur, isosurfaces=False, dissipation=False, volume=False, gif=False)
    return p2, f"Δr = {p['dr']:g} → {p2['dr']:.4g} (coût ≈ ×{facteur**3:.0f})"


def _fin_du_suivi(rg):
    """Instant où un seul pic de vorticité subsiste pendant trois diagnostics consécutifs (fusion ou contact) et
    masque des instants antérieurs (courbes de suivi tronquées)."""
    # Instants où un seul pic de vorticité est détecté.
    seul = rg["n_peaks"] < 2
    # next(générateur, None) : premier indice qui convient, ou None.
    i = next((k for k in range(len(seul) - 2) if seul[k] and seul[k + 1] and seul[k + 2]), None)
    # Aucune fusion : tous les instants sont conservés.
    if i is None:
        return None, np.ones_like(rg["t"], dtype=bool)
    return float(rg["t"][i]), rg["t"] < rg["t"][i]


def figure_critere(t, b_sur_s, traversees, rapports, t_fusion):
    """Figure du critère de fusion : écart des cœurs rapporté au rayon de cœur, b/σ(t), et seuil de Meunier."""
    # Figure d'un seul panneau ; constrained_layout ajuste les marges.
    fig, axe = plt.subplots(figsize=(8.8, 4.0), constrained_layout=True)
    # Bande du seuil de fusion des tourbillons co-rotatifs (b/σ ≈ 4,2–4,5) et courbe b/σ(t).
    axe.axhspan(4.2, 4.5, color=viz.INK["neutral"], zorder=0, label="seuil de fusion b/σ ≈ 4,2–4,5")
    axe.plot(t, b_sur_s, color=viz.SERIES[0], label="b/σ : distance des cœurs / rayon de cœur")
    # Traversées : points et valeurs de b/σ ; fusion : trait vertical.
    for tc, r in zip(traversees, rapports):
        axe.plot(tc, r, "o", color=viz.SERIES[1], ms=7, zorder=3)
        axe.annotate(f"traversée\nb/σ = {r:.1f}", (tc, r), textcoords="offset points", xytext=(8, -28), fontsize=8.5,
                     color=viz.INK["secondary"])
    if t_fusion is not None:
        axe.axvline(t_fusion, color=viz.SERIES[7], ls="--", lw=1.2, label=f"fusion des cœurs (t ≈ {t_fusion:.1f})")
    # Libellés, ordonnée à partir de 0, titre et légende.
    axe.set_xlabel("t")
    axe.set_ylabel("b/σ")
    axe.set_ylim(bottom=0.0)
    axe.set_title("Critère de fusion : les cœurs fusionnent si b/σ passe sous le seuil pendant une traversée")
    axe.legend(loc="upper right")
    return fig


def calculer(p, prog, complet=True):
    """Calcul complet ou léger (convergence) de deux anneaux coaxiaux."""
    # Anneaux, durée, Reynolds et plus grand rayon.
    liste = anneaux(p)
    T, Re, R_max = p["t_end"], float(p["Re"]), max(p["RA"], p["RB"])
    rappel, images = ns.enregistreur_images(T, prog, "Navier–Stokes, deux anneaux")
    # Appel direct de la fonction du projet (profils de Hovmöller et suivi individuel des cœurs inclus ; suivi
    # « signé » automatique si une circulation est négative).
    res = ax.run_coaxial_rings(liste, Re=Re, t_end=T, dr=p["dr"], Lr=8.0 * R_max, r_uniform=max(2.4, R_max + 1.2),
                               snapshot_times=(0.0, round(0.5 * T, 6), T), callback=rappel)
    # Diagnostics globaux, suivi individuel des cœurs, instantanés et grille.
    d, rg, snaps, grille = res["diagnostics"], res["rings"], res["snapshots"], res["grid"]
    # Même signe : saute-mouton ; signes opposés : collision frontale.
    meme_signe = np.sign(p["GA"]) == np.sign(p["GB"])
    # Fin du suivi (fusion des cœurs de même signe, ou contact des cœurs opposés) et instants conservés.
    t_fin, garde = _fin_du_suivi(rg)
    # Instants conservés, rayons et positions axiales des deux anneaux.
    tg, RA_, RB_, ZA, ZB = rg["t"][garde], rg["R"][garde, 0], rg["R"][garde, 1], rg["Z"][garde, 0], rg["Z"][garde, 1]
    # Référence en fluide parfait : modèle coaxial régularisé, δ équivalent à chaque cœur gaussien.
    deltas = [float(theory.rosenhead_moore_delta(an["a"], "gaussian")) for an in liste]
    modele = theory.CoaxialRingsModel([p["RA"], p["RB"]], [0.0, p["sep"]], [p["GA"], p["GB"]], deltas).integrate(T, 801)
    # Énergie dissipée, résidu du bilan et mesures pour la calibration.
    perte = float(100.0 * (1.0 - d["energy"][-1] / d["energy"][0]))
    _, residu = ns.bilan_energie(d)
    mesures = {"t_calcul": res["wall_time"], "n_pas": res["solver"].n_steps, "taille": grille.Nr * grille.Nz}
    # Analyse du saute-mouton (traversées, critère de fusion)…
    if meme_signe:
        # Traversées simulées et prévues ; écart des cœurs b(t) et rayon de cœur moyen σ(t) = √(a² + 4νt).
        traversees = c.croisements(tg, ZA - ZB)
        traversees_modele = c.croisements(modele["t"], modele["Z"][0] - modele["Z"][1])
        nu = max(abs(p["GA"]), abs(p["GB"])) / Re
        sigma = 0.5 * (theory.viscous_core_radius(p["aA"], nu, tg) + theory.viscous_core_radius(p["aB"], nu, tg))
        b_sur_s = np.hypot(RA_ - RB_, ZA - ZB) / sigma
        # b/σ à chaque traversée (interpolation linéaire, np.interp).
        rapports = [float(np.interp(tc, tg, b_sur_s)) for tc in traversees]
        grandeurs = {"Énergie dissipée (%)": perte,
                     "1ʳᵉ traversée (t)": traversees[0] if traversees else float("nan"),
                     "b/σ à la 1ʳᵉ traversée": rapports[0] if rapports else float("nan"),
                     "Fusion des cœurs (t)": t_fin if t_fin is not None else float("nan")}
    # … ou de la collision frontale (rayon maximal, contact des cœurs).
    else:
        grandeurs = {"Énergie dissipée (%)": perte, "Rayon maximal": float(np.nanmax(rg["R"][garde])) if garde.any() else float("nan"),
                     "Contact des cœurs (t)": t_fin if t_fin is not None else float("nan")}
    # Sortie minimale ; calcul léger (convergence) : on s'arrête là.
    sortie = {"grandeurs": grandeurs, "mesures": mesures, "type_cout": CLE}
    if not complet:
        return sortie
    # Fenêtre des vues : centroïde global (même signe) ou milieu fixe (signes opposés : centroïde mal défini).
    if meme_signe:
        Zc = [float(np.interp(s["t"], d["t"], d["Z"])) for s in snaps]
        centre, demi = c.fenetre_axiale(d["Z"], marge=2.3)
    else:
        centre, demi = 2.0 + 0.5 * p["sep"], min(0.5 * p["sep"] + 2.3, 5.0)
        Zc = [centre] * len(snaps)
    # Animation : images, échelle commune et fenêtre fixe.
    animation = ns.donnees_animation(images, snaps[0], centre, demi, R_max + 1.0)
    # Trajectoires suivies superposées au diagramme de Hovmöller (aucune si le suivi est vide).
    pistes = [(tg, ZA, "anneau A"), (tg, ZB, "anneau B")] if garde.any() else None
    # Chronomètre des tracés.
    debut_figures = time.perf_counter()
    # Figures de base : Hovmöller et bilan d'énergie.
    taches = [
        ("Hovmöller z–t", "Maximum radial de la vorticité et minimum de pression en fonction de z et t, avec les "
                          "trajectoires suivies des deux cœurs.",
         lambda: viz.plot_hovmoller(res["profiles"], tracks=pistes, title="Deux anneaux — Hovmöller z–t")),
        ("Bilan d'énergie", "Énergie, impulsion et enstrophie : l'énergie suit E₀ − ν∫Ω dt.",
         lambda: viz.plot_energy_enstrophy_budget(res, title="Deux anneaux — bilan d'énergie et d'enstrophie")),
    ]
    if meme_signe and garde.sum() >= 3:
        taches.insert(0, ("Critère de fusion", "Distance entre les cœurs rapportée à leur rayon, b/σ(t) : à chaque "
                                               "traversée (points), les cœurs fusionnent si b/σ passe sous le seuil "
                                               "≈ 4,2–4,5 (Meunier et al. 2002).",
                          lambda: figure_critere(tg, b_sur_s, traversees, rapports, t_fin)))
        # Portrait de phase (R_A − R_B, z_A − z_B) : cycle fermé en fluide parfait, spirale en visqueux.
        courbes_phase = [{"label": "modèle coaxial (fluide parfait)", "t": modele["t"], "R": modele["R"].T,
                          "Z": modele["Z"].T, "ls": "--"},
                         {"label": f"Navier–Stokes, Re_Γ = {p['Re']:g}", "t": tg, "R": rg["R"][garde], "Z": rg["Z"][garde],
                          "ls": "-"}]
        taches.append(("Portrait de phase", "Plan (R_A − R_B, z_A − z_B) : cycle fermé en fluide parfait, spirale qui "
                                            "s'enroule sous l'effet de la viscosité.",
                       lambda: viz.plot_phase_diagram(courbes_phase)))
    # Figures optionnelles.
    if p["isosurfaces"]:
        taches.append(("Isosurfaces 3D", "Isosurfaces du critère Q reconstruites par révolution aux trois instantanés "
                                         "(couleur |u|).",
                       lambda: v3.plot_isosurface_sequence(snaps, Zc, ncols=3, half_width=1.7, r_max=R_max + 0.7)))
    if p["dissipation"]:
        taches.append(("Dissipation Φ", "Taux local de dissipation Φ = 2μ S:S aux trois instantanés.",
                       lambda: viz.plot_dissipation_map(snaps, Zc, r_max=R_max + 1.2)))
    if p["volume"]:
        taches.append(("Rendu volumique", "Enstrophie ½ω² reconstruite par révolution (milieu et fin du calcul).",
                       lambda: ns.rendu_volumique(snaps[1:], Zc[1:], 1.8, R_max + 0.8,
                                                  "Deux anneaux — rendu volumique de ½ω²")))
    # Tracés (barre de 75 % à 92 % ou 100 %).
    figures = c.tracer(taches, prog, 0.75, 0.92 if p["gif"] else 1.0)
    # GIF téléchargeable (optionnel).
    sortie["gif"] = None
    if p["gif"]:
        sortie["gif"] = {"octets": ns.gif_vorticite(animation, prog, 0.92), "fichier": "deux_anneaux_vorticite.gif"}
    # Durée des tracés.
    mesures["t_figures"] = time.perf_counter() - debut_figures
    if meme_signe:
        sortie.update(_resultats_saute_mouton(p, d, res, garde, tg, RA_, RB_, ZA, ZB, b_sur_s, traversees,
                                              traversees_modele, rapports, t_fin, perte, residu, snaps, modele, grille))
    else:
        sortie.update(_resultats_collision(p, d, res, garde, tg, RA_, RB_, ZA, ZB, t_fin, perte, residu, modele, grille))
    # Données d'affichage interactif et transposition en unités physiques.
    # Figures, animation et vue 3D interactives, unités physiques.
    sortie.update({"figures": figures, "animation": dict(animation, type="navier_stokes"),
                   "vue3d": {"type": "isosurface", "snaps": snaps, "Z": Zc, "demi": 1.7, "r_max": R_max + 0.7},
                   "unites": {"Re": Re, "R": p["RA"], "Gamma": max(abs(p["GA"]), abs(p["GB"])),
                              "U": float(np.mean(np.gradient(d["Z"], d["t"])[-10:])) if meme_signe else 0.0,
                              "t_end": T, "distance": float(d["Z"][-1] - d["Z"][0]) if meme_signe else 0.0}})
    return sortie


def _resultats_saute_mouton(p, d, res, garde, tg, RA_, RB_, ZA, ZB, b_sur_s, traversees, traversees_modele,
                            rapports, t_fin, perte, residu, snaps, modele, grille):
    """Métriques, notes, courbes et données du saute-mouton (circulations de même signe)."""
    # Qualité numérique : vorticité de signe opposé à celle des anneaux (−ω si les circulations sont négatives).
    signe = 1.0 if p["GA"] > 0 else -1.0
    parasite = ns.vorticite_parasite({"omega": signe * snaps[-1]["omega"]})
    # Métriques du saute-mouton.
    metriques = [
        c.metrique("Traversées (Navier–Stokes)", str(len(traversees)), f"t = {c.liste_instants(traversees)}",
                   aide("Saute-mouton")),
        c.metrique("Traversées (fluide parfait)", str(len(traversees_modele)), f"t = {c.liste_instants(traversees_modele)}",
                   aide("Modèle coaxial")),
        c.metrique("b/σ à la 1ʳᵉ traversée", f"{rapports[0]:.2f}" if rapports else "—", "seuil de fusion ≈ 4,2–4,5",
                   aide("Fusion des cœurs"), c.sous_echantillon(b_sur_s)),
        c.metrique("Fusion des cœurs", f"t ≈ {t_fin:.1f}" if t_fin is not None else "non",
                   "un seul pic subsiste" if t_fin is not None else None, aide("Fusion des cœurs")),
        c.metrique("Énergie dissipée", f"{perte:.1f} %", f"résidu du bilan {residu:.0e} E₀",
                   aide("Enstrophie Ω"), c.sous_echantillon(d["energy"] / d["energy"][0])),
        c.metrique("Qualité numérique", f"{100 * parasite:.2f} %", "vorticité négative parasite",
                   "Moins de 0,2 % est négligeable. " + (aide("Nombre de Péclet de maille") or "")),
        c.metrique("Divergence maximale", f"{d['max_div'].max():.1e}", None, aide("Divergence (incompressibilité)")),
        c.metrique("Grille", f"{grille.Nr} × {grille.Nz}", f"{res['solver'].n_steps} pas — {res['wall_time']:.1f} s"),
    ]
    # Interprétation : verdict du critère de fusion à chaque traversée.
    # Interprétation : verdict du critère de fusion à chaque traversée.
    notes = []
    for k, (tc, r) in enumerate(zip(traversees, rapports), start=1):
        verdict = ("sous le seuil de fusion (≈ 4,2–4,5) : la paire co-rotative fusionne" if r < 4.2 else
                   "au-dessus du seuil de fusion (≈ 4,2–4,5) : les anneaux survivent à cette traversée")
        notes.append(f"**Traversée {k}** (t ≈ {tc:.2f}) : les cœurs sont distants de b ≈ {r:.1f} σ, {verdict}.")
    if t_fin is not None:
        notes.append(f"Les cœurs **fusionnent vers t ≈ {t_fin:.1f}** ; en fluide parfait, le modèle coaxial prévoit des "
                     f"traversées à t = {c.liste_instants(traversees_modele)}.")
    elif not traversees:
        notes.append("Aucune traversée pendant la durée simulée : allongez t_end.")
    notes.append(f"Énergie dissipée : **{perte:.1f} %**.")
    # Courbes interactives (format long) : positions et rayons suivis, comparés au modèle coaxial.
    # Courbes interactives.
    df_Z = pd.concat([pd.DataFrame({"t": tg, "valeur": ZA, "série": "z anneau A"}),
                      pd.DataFrame({"t": tg, "valeur": ZB, "série": "z anneau B"})])
    df_R = pd.concat([pd.DataFrame({"t": tg, "valeur": RA_, "série": "R anneau A"}),
                      pd.DataFrame({"t": tg, "valeur": RB_, "série": "R anneau B"})]
                     + [pd.DataFrame({"t": modele["t"], "valeur": modele["R"][i], "série": f"R anneau {'AB'[i]} (fluide parfait)"})
                        for i in range(2)])
    df_b = pd.DataFrame({"t": tg, "b/σ": b_sur_s, "seuil de fusion": np.full_like(tg, 4.2)})
    courbes = [c.courbe("Critère de fusion b/σ(t)", df_b, ["b/σ", "seuil de fusion"], "b/σ", couleurs=viz.SERIES[:2]),
               c.courbe("Positions axiales suivies z(t)", df_Z, "valeur", "z"),
               c.courbe("Rayons R(t) : Navier–Stokes et fluide parfait", df_R, "valeur", "R"),
               ns.courbe_bilan(d)]
    # Séries comparables entre calculs.
    comparables = [{"nom": "Écart axial z_A − z_B", "x": tg, "y": ZA - ZB, "x_label": "t", "y_label": "z_A − z_B"},
                   {"nom": "Critère b/σ(t)", "x": tg, "y": b_sur_s, "x_label": "t", "y_label": "b/σ"},
                   {"nom": "Énergie E(t)/E₀", "x": d["t"], "y": d["energy"] / d["energy"][0], "x_label": "t",
                    "y_label": "E/E₀"}]
    # Tableau des diagnostics globaux.
    donnees = pd.DataFrame({"t": d["t"], "Z_global": d["Z"], "energy": d["energy"], "dissipation": d["dissipation"],
                            "circulation": d["circulation"], "max_div": d["max_div"]})
    # Éléments de la sortie propres au saute-mouton.
    return {"metriques": metriques, "notes": notes, "courbes": courbes, "comparables": comparables,
            "donnees": donnees, "fichier_csv": "deux_anneaux_diagnostics.csv"}


def _resultats_collision(p, d, res, garde, tg, RA_, RB_, ZA, ZB, t_fin, perte, residu, modele, grille):
    """Métriques, notes, courbes et données de la collision frontale (circulations opposées)."""
    # Écart axial minimal atteint et rayon maximal (suivi) comparé au modèle coaxial en fluide parfait.
    ecart_min = float(np.nanmin(np.abs(ZB - ZA))) if garde.any() else float("nan")
    R_max = float(np.nanmax(np.concatenate([RA_, RB_]))) if garde.any() else float("nan")
    R_max_modele = float(modele["R"].max())
    # Métriques de la collision.
    metriques = [
        c.metrique("Écart minimal entre anneaux", f"{ecart_min:.3f}", f"initial : {p['sep']:g}"),
        c.metrique("Rayon maximal atteint", f"{R_max:.3f}", f"fluide parfait : {R_max_modele:.3f}", aide("Modèle coaxial"),
                   c.sous_echantillon(RA_)),
        c.metrique("Contact des cœurs", f"t ≈ {t_fin:.1f}" if t_fin is not None else "non",
                   "les pics de vorticité se confondent" if t_fin is not None else None),
        c.metrique("Énergie dissipée", f"{perte:.1f} %", f"résidu du bilan {residu:.0e} E₀",
                   aide("Enstrophie Ω"), c.sous_echantillon(d["energy"] / d["energy"][0])),
        c.metrique("Divergence maximale", f"{d['max_div'].max():.1e}", None, aide("Divergence (incompressibilité)")),
        c.metrique("Grille", f"{grille.Nr} × {grille.Nz}", f"{res['solver'].n_steps} pas — {res['wall_time']:.1f} s"),
    ]
    # Interprétation chiffrée.
    notes = [f"En approchant du plan de symétrie, les anneaux s'élargissent jusqu'à R ≈ {R_max:.2f} (fluide parfait : "
             f"{R_max_modele:.2f}) et ralentissent ; leur écart minimal vaut {ecart_min:.2f}. Énergie dissipée : "
             f"**{perte:.1f} %**."]
    # Courbes interactives : positions et rayons, comparés au modèle en fluide parfait.
    df_Z = pd.concat([pd.DataFrame({"t": tg, "valeur": ZA, "série": "z anneau A"}),
                      pd.DataFrame({"t": tg, "valeur": ZB, "série": "z anneau B"})]
                     + [pd.DataFrame({"t": modele["t"], "valeur": 2.0 + modele["Z"][i], "série": f"z anneau {'AB'[i]} (fluide parfait)"})
                        for i in range(2)])
    df_R = pd.concat([pd.DataFrame({"t": tg, "valeur": RA_, "série": "R anneau A"}),
                      pd.DataFrame({"t": tg, "valeur": RB_, "série": "R anneau B"})]
                     + [pd.DataFrame({"t": modele["t"], "valeur": modele["R"][i], "série": f"R anneau {'AB'[i]} (fluide parfait)"})
                        for i in range(2)])
    courbes = [c.courbe("Positions axiales z(t) : Navier–Stokes et fluide parfait", df_Z, "valeur", "z"),
               c.courbe("Rayons R(t) : Navier–Stokes et fluide parfait", df_R, "valeur", "R"), ns.courbe_bilan(d)]
    # Séries comparables entre calculs.
    comparables = [{"nom": "Écart axial z_B − z_A", "x": tg, "y": ZB - ZA, "x_label": "t", "y_label": "z_B − z_A"},
                   {"nom": "Rayon de l'anneau A", "x": tg, "y": RA_, "x_label": "t", "y_label": "R_A"},
                   {"nom": "Énergie E(t)/E₀", "x": d["t"], "y": d["energy"] / d["energy"][0], "x_label": "t",
                    "y_label": "E/E₀"}]
    # Tableau des diagnostics globaux.
    donnees = pd.DataFrame({"t": d["t"], "energy": d["energy"], "dissipation": d["dissipation"], "max_div": d["max_div"]})
    # Éléments de la sortie propres à la collision.
    return {"metriques": metriques, "notes": notes, "courbes": courbes, "comparables": comparables,
            "donnees": donnees, "fichier_csv": "collision_diagnostics.csv"}
