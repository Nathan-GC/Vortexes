"""
appli/volets/theorie.py — Volet « formules analytiques » (src/theory.py), recalculé à chaque mouvement d'un
curseur, sans bouton : vitesse d'un anneau mince, vortex sphérique de Hill, nombre de formation.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from appli import commun as c
from appli.glossaire import aide
from src import theory
from src import visualization as viz

CLE = "theorie"
LIBELLE = "📐 Formules analytiques"
RESUME = ("Calcul instantané des formules du rapport : vitesse d'un anneau mince selon le type de cœur, vortex de "
          "Hill, nombre de formation F*.")
PRESENTATION = ("Formules analytiques du module `src/theory.py`, recalculées à chaque mouvement d'un curseur : vitesse "
                "d'un anneau mince (Kelvin, Saffman, Fukumoto–Moffatt, Rosenhead–Moore), vortex sphérique de Hill et "
                "nombre de formation.")
FORMULE = r"U = \frac{\Gamma}{4\pi R}\left[\ln\frac{8R}{a} - C\right]"
THEORIE = [("Fondements théoriques", "2.2"), ("Fondements théoriques", "3.2")]
LIMITES = ["Formules d'anneau mince : erreur en O((a/R)² ln(a/R)), acceptable pour a/R ≲ 0,35.",
           "Vortex de Hill : solution exacte des équations d'Euler (fluide parfait)."]
# Calcul instantané : pas de bouton ni d'historique.
INSTANTANE = True
SCENARIOS = {}
# Préfixe des clés de ses widgets (réglages enregistrés avec chaque calcul, puis rechargeables).
PREFIXE = "th_"
# Types de noyau (clé du module theory → libellé avec la constante C).
NOYAUX = {"uniform": "uniforme (Kelvin, C = 1/4)", "gaussian": "gaussien (Saffman, C ≈ 0.558)",
          "hollow": "creux (Hicks, C = 1/2)", "rosenhead_moore": "Rosenhead–Moore (C = 1)"}


def controles():
    """Widgets du volet ; renvoie le dictionnaire des paramètres."""
    p = {}
    st.markdown("**Anneau mince**")
    # Anneau mince : circulation, rayon, épaisseur relative du cœur, type de noyau.
    p["Gamma"] = c.curseur("Circulation Γ", 0.1, 5.0, 1.0, 0.1, "th_G", aide("Circulation Γ"))
    p["R"] = c.curseur("Rayon R", 0.2, 3.0, 1.0, 0.05, "th_R")
    p["aR"] = c.curseur("Épaisseur relative du cœur a/R", 0.01, 0.50, 0.10, 0.01, "th_aR", aide("Rayon de cœur a (σ)"))
    p["noyau"] = c.menu("Type de noyau", list(NOYAUX), "uniform", "th_noyau", format_func=NOYAUX.get)
    st.markdown("**Vortex sphérique de Hill**")
    # Vortex de Hill : rayon de la sphère et vitesse.
    p["hill_a"] = c.curseur("Rayon de la sphère a", 0.5, 2.0, 1.0, 0.1, "th_ha", aide("Vortex de Hill"))
    p["hill_U"] = c.curseur("Vitesse de propagation U", 0.5, 2.0, 1.0, 0.1, "th_hU")
    st.markdown("**Formation des anneaux**")
    # Énergie adimensionnelle critique du détachement.
    p["alpha"] = c.curseur("Énergie adimensionnelle critique α*", 0.20, 0.50, 0.33, 0.01, "th_alpha",
                           aide("Énergie adimensionnelle α"))
    return p


# Figure du vortex de Hill tracée une seule fois (a = U = 1) : la solution est autosemblable.
@st.cache_data(show_spinner="Tracé du vortex de Hill…")
def figure_hill():
    """Image PNG de la figure adimensionnelle du vortex de Hill (lignes de courant, repères, pression)."""
    with c.VERROU:
        return c.png(viz.plot_hill_vortex(a=1.0, U=1.0, n=201))


def afficher(p):
    """Résultats instantanés des formules analytiques (zone principale)."""
    # Circulation, rayon et rayon de cœur a = (a/R) × R.
    G, R = p["Gamma"], p["R"]
    a = p["aR"] * R
    # Vitesses : formule de l'anneau mince pour le noyau choisi, Saffman + Fukumoto–Moffatt (cœur gaussien σ = a),
    # filament de Rosenhead–Moore équivalent (δ = a e^{C−1}) en forme close exacte.
    U = float(theory.ring_speed(G, R, a, p["noyau"]))
    U_fm = float(theory.saffman_speed(G, R, a, second_order=True))
    delta = float(theory.rosenhead_moore_delta(a, p["noyau"]))
    U_rm = float(theory.rosenhead_moore_ring_speed(G, R, delta, exact=True))
    # Section 1 : vitesse d'un anneau mince.
    st.markdown("#### Vitesse d'autopropulsion d'un anneau mince")
    c.afficher_metriques([
        c.metrique("U (noyau choisi)", f"{U:.4f}", f"C = {theory.core_constant(p['noyau']):.4f}"),
        c.metrique("Saffman + Fukumoto–Moffatt", f"{U_fm:.4f}", "cœur gaussien σ = a",
                   aide("Fukumoto–Moffatt (correction de)")),
        c.metrique("Filament de Rosenhead–Moore", f"{U_rm:.4f}", f"δ équivalent = {delta:.4f}",
                   "Forme close exacte U = Γ/(2π√(4R²+δ²)) [K(m) − E(m)] avec δ = a·e^{C−1}."),
        c.metrique("Énergie E (ρ = 1)", f"{float(theory.ring_energy(G, R, a, p['noyau'])):.4f}",
                   f"impulsion I = πρR²Γ = {float(theory.ring_impulse(G, R)):.4f}", aide("Impulsion I")),
    ])
    # Vitesse réduite U·4πR/Γ (fonction de a/R seulement) pour les différents noyaux.
    # Courbes U·4πR/Γ(a/R) des différents noyaux (150 valeurs de a/R).
    aR = np.linspace(0.01, 0.5, 150)
    df = pd.DataFrame({"a/R": aR,
                       "uniforme (Kelvin)": theory.ring_speed(1.0, 1.0, aR, "uniform") * 4 * np.pi,
                       "gaussien (Saffman)": theory.saffman_speed(1.0, 1.0, aR) * 4 * np.pi,
                       "gaussien + Fukumoto–Moffatt": theory.saffman_speed(1.0, 1.0, aR, second_order=True) * 4 * np.pi,
                       "creux (Hicks)": theory.ring_speed(1.0, 1.0, aR, "hollow") * 4 * np.pi})
    c.graphique(c.courbe("Vitesse réduite U·4πR/Γ en fonction de a/R (survolez les courbes pour lire les valeurs)", df,
                         list(df.columns[1:]), "U · 4πR/Γ", x="a/R", x_label="a/R", couleurs=viz.SERIES[:4]))
    # Section 2 : vortex de Hill (invariants analytiques).
    st.markdown("#### Vortex sphérique de Hill")
    inv = theory.hill_invariants(p["hill_a"], p["hill_U"])
    c.afficher_metriques([
        c.metrique("Circulation Γ = 5Ua", f"{inv['circulation']:.3f}"),
        c.metrique("Impulsion I = 2πρa³U", f"{inv['impulse']:.3f}"),
        c.metrique("Énergie E = (10/7)πρa³U²", f"{inv['energy']:.3f}"),
        c.metrique("Énergie adimensionnelle α", f"{inv['dimensionless_energy']:.3f}", f"C_A = {inv['added_mass_coefficient']:.2f}",
                   aide("Énergie adimensionnelle α", "Masse ajoutée")),
    ])
    # Figure adimensionnelle, tracée une seule fois.
    st.image(figure_hill(), width="stretch", alt="Vortex de Hill : lignes de courant dans les repères du vortex et du "
                                                 "laboratoire, et champ de pression.")
    st.caption("Figure adimensionnelle (a = U = 1) : la solution de Hill est autosemblable — les longueurs se mettent "
               "à l'échelle avec a, les vitesses avec U et la pression avec ρU². Seules les grandeurs ci-dessus "
               "dépendent des curseurs.")
    # Section 3 : nombre de formation.
    st.markdown("#### Nombre de formation")
    F = float(theory.formation_number(p["alpha"]))
    st.latex(r"\alpha_{\text{slug}} = \frac{\sqrt{\pi/2}}{L/D}\quad\Rightarrow\quad F^* = \frac{\sqrt{\pi/2}}{\alpha^*}")
    c.afficher_metriques([c.metrique("Nombre de formation F*", f"{F:.2f}", f"α* = {p['alpha']:.2f}",
                                     aide("Nombre de formation F*", "Modèle « slug »"))])
    # Énergie fournie par le modèle slug en fonction de L/D, comparée au seuil α*.
    LD = np.linspace(1.0, 10.0, 200)
    df = pd.DataFrame({"L/D": LD, "α du modèle slug": theory.slug_dimensionless_energy(LD),
                       "seuil α*": np.full_like(LD, p["alpha"])})
    c.graphique(c.courbe("Énergie adimensionnelle fournie par le piston : l'anneau se détache quand elle passe sous α*",
                         df, ["α du modèle slug", "seuil α*"], "énergie adimensionnelle α", x="L/D", x_label="L/D",
                         couleurs=viz.SERIES[:2]))
