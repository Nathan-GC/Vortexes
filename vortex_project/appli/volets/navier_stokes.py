"""
appli/volets/navier_stokes.py — Outils communs aux volets résolus par le solveur axisymétrique
(anneau visqueux, vortex de Hill, deux anneaux coaxiaux).

Estimation de la durée (grille, pas de temps CFL/visqueux, coefficients calibrés), garde-fous de résolution
(rapport a/Δr et nombre de Péclet de maille), enregistrement d'images légères pendant le calcul (animation),
indicateur de qualité a posteriori (vorticité négative parasite), rendu volumique et GIF téléchargeable.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# matplotlib (rendu « Agg ») : images du GIF ; numpy : tableaux ; pandas : courbes ; TwoSlopeNorm : échelle
# centrée sur 0 ; cumulative_trapezoid : intégrale cumulée (bilan d'énergie).
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.integrate import cumulative_trapezoid  # noqa: E402

from appli import calibration  # noqa: E402
from appli.commun import VERROU, courbe, gif_depuis_png, png  # noqa: E402
from src import solver_2d_axisymmetric as ax  # noqa: E402
from src import theory  # noqa: E402
from src import visualization as viz  # noqa: E402
from src import visualization_3d as v3  # noqa: E402

# Longueur du domaine périodique en z des fonctions run_* du solveur.
LZ = 10.0
# Rapport u_θ,max · 2πa/Γ du tourbillon de Lamb–Oseen (vitesse azimutale maximale autour d'un cœur gaussien).
LAMB_OSEEN_VMAX = 0.6382
# Pas de maillage proposés (du plus grossier au plus fin) et nombres de Reynolds (échelle quasi logarithmique).
OPTIONS_DR = [0.06, 0.05, 0.04, 0.03, 0.025, 0.02]
OPTIONS_RE = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 10000]


def vitesse_max(anneaux):
    """Vitesse maximale initiale estimée ≈ 2 u_θ,max + U (cœur gaussien et propagation), maximum sur les anneaux."""
    # Expression génératrice : u_θ,max = 0,638 Γ/(2πa) ; saffman_speed : vitesse de propagation de l'anneau.
    return max(2.0 * LAMB_OSEEN_VMAX * abs(an["Gamma"]) / (2.0 * np.pi * an["a"])
               + abs(float(theory.saffman_speed(an["Gamma"], an["R"], an["a"]))) for an in anneaux)


def peclet(vitesse, dr, nu):
    """Nombre de Péclet de maille Pe_Δ = u_max Δr / ν (transport / diffusion à l'échelle d'une maille)."""
    return vitesse * dr / nu


def alertes_resolution(a_min, dr, pe):
    """Garde-fous de résolution, établis par une étude du solveur (tutoriel, § garde-fous numériques).

    Sur un anneau de Lamb–Oseen (t = 3, Re jusqu'à 40 000), aucune oscillation notable n'apparaît tant que
    a/Δr ≥ 6, quel que soit Pe_Δ ; pour 4 ≤ a/Δr < 6, la vorticité négative parasite dépasse 0,2 % au-delà de
    Pe_Δ ≈ 200, et dès Pe_Δ ≈ 70 pour a/Δr < 4. La vitesse de propagation reste peu sensible dans tous ces cas.
    """
    alertes = []
    rapport = a_min / dr
    # Résolution du cœur : moins de 5 mailles par rayon de cœur (règle du tutoriel).
    if rapport < 5:
        alertes.append(f"Résolution du cœur limitée : a/Δr = {rapport:.1f} < 5. Les tendances restent justes, mais "
                       "affinez Δr pour des résultats précis (a/Δr ≥ 8 pour une précision de rapport).")
    # Seuil de Péclet selon la résolution (None : pas de limite constatée).
    seuil = None if rapport >= 6 else (200.0 if rapport >= 4 else 70.0)
    if seuil is not None and pe > seuil:
        alertes.append(f"Nombre de Péclet de maille Pe_Δ ≈ {pe:.0f} > {seuil:.0f} pour a/Δr = {rapport:.1f} : de petites "
                       "oscillations (vorticité négative parasite) peuvent apparaître. Affinez Δr (a/Δr ≥ 6) ou "
                       "baissez Re.")
    return alertes


def estimation(type_cout, anneaux, Re, t_end, dr, Lr, r_uniforme, figures_brut, vitesse=None):
    """Durée estimée d'un calcul axisymétrique : même grille que run_*, pas de temps limité par la condition CFL
    (0,8, majorée de 10 % car la vitesse décroît) et par la diffusion (Δr²/4ν), coefficients calibrés."""
    # Grille étirée identique à celle des fonctions run_* (croissance 5 % au-delà de la zone uniforme).
    grille = ax.AxisymmetricGrid.stretched(dr, r_uniforme, Lr, 1.05, LZ, int(round(LZ / dr)))
    # Vitesse maximale (fournie, ex. vortex de Hill, ou estimée) et viscosité ν = max|Γ| / Re.
    v = vitesse if vitesse is not None else vitesse_max(anneaux)
    nu = max(abs(an["Gamma"]) for an in anneaux) / Re
    # Pas de temps limitant et nombre de pas (avant la correction calibrée).
    dt = min(1.1 * 0.8 * dr / v, dr**2 / (4.0 * nu))
    est = calibration.estimation(type_cout, grille.Nr * grille.Nz, t_end / dt, figures_brut,
                                 f"grille {grille.Nr} × {grille.Nz}")
    # Nombre de Péclet de maille (affiché avec l'estimation).
    est["peclet"] = peclet(v, dr, nu)
    return est


def enregistreur_images(t_end, prog, libelle, n_images=24):
    """Fonction de rappel du solveur : progression et images légères (t, ω, ψ) pour l'animation.

    Renvoie (rappel, images) ; le rappel est appelé après chaque pas de temps avec le solveur en argument.
    """
    # Instants visés (t = 0 est ajouté après le calcul, depuis l'instantané initial).
    cibles = np.linspace(0.0, t_end, n_images + 1)[1:]
    images, suivant = [], [0]

    def rappel(solveur):
        # Calcul : 72 % de la barre, proportionnellement au temps simulé.
        prog(0.72 * solveur.time / t_end, f"{libelle} : t = {solveur.time:.2f} / {t_end:g} — {solveur.n_steps} pas")
        # Premier pas qui atteint chaque instant visé : copie de la vorticité et de la fonction de courant.
        while suivant[0] < len(cibles) and solveur.time >= cibles[suivant[0]] - 1e-9:
            images.append({"t": solveur.time, "omega": solveur.vorticity(), "psi": solveur.stream_function()})
            suivant[0] += 1

    return rappel, images


def donnees_animation(images, snap0, centre, demi, r_max):
    """Données de l'animation : image initiale ajoutée, grille, échelle de couleur commune, fenêtre fixe."""
    # Image t = 0 tirée de l'instantané complet initial (insert(0, …) : en tête de liste).
    images.insert(0, {"t": 0.0, "omega": snap0["omega"], "psi": snap0["psi"]})
    # vmax : maximum de |ω| sur toutes les images (la décroissance de la vorticité reste visible).
    return {"images": images, "r_f": snap0["r_f"], "z_f": snap0["z_f"],
            "vmax": float(max(np.abs(im["omega"]).max() for im in images)), "centre": centre, "demi": demi,
            "r_max": r_max}


def vorticite_parasite(snap):
    """Indicateur de qualité : −min ω / max ω (vorticité négative créée par des oscillations numériques, pour un
    écoulement dont la vorticité physique est positive)."""
    om = snap["omega"]
    return float(max(-om.min(), 0.0) / om.max()) if om.max() > 0 else float("nan")


def figure_vorticite(image, exp):
    """Image matplotlib de la vorticité (coupe méridienne en miroir) et des lignes de courant (pour le GIF)."""
    r_f, z_f = exp["r_f"], exp["z_f"]
    # viz._window : colonnes z de la fenêtre fixe (domaine périodique) ; rayons affichés.
    kf, zw = viz._window(z_f, exp["centre"], exp["demi"])
    irf = r_f <= exp["r_max"]
    # viz._mirror : coupe x–z complète (vorticité impaire, fonction de courant paire).
    rr, oo = viz._mirror(r_f[irf], image["omega"][irf][:, kf], odd=True)
    _, pp = viz._mirror(r_f[irf], image["psi"][irf][:, kf])
    # Figure de taille fixe (toutes les images du GIF ont les mêmes dimensions).
    fig, axe = plt.subplots(figsize=(7.6, 3.5), constrained_layout=True)
    carte = axe.pcolormesh(zw, rr, oo, cmap=viz.CMAP_DIV, norm=TwoSlopeNorm(0.0, -exp["vmax"], exp["vmax"]),
                           shading="gouraud", rasterized=True)
    if pp.max() > pp.min():
        # 12 niveaux intermédiaires de ψ (lignes de courant du repère du laboratoire).
        axe.contour(zw, rr, pp, levels=np.linspace(pp.min(), pp.max(), 14)[1:-1], colors=viz.INK["secondary"],
                    linewidths=0.55, alpha=0.75)
    axe.set_aspect("equal")
    axe.grid(False)
    axe.axhline(0.0, color=viz.INK["axis"], lw=0.8, ls=":")
    axe.set_xlabel("z")
    axe.set_ylabel("x  (r en miroir)")
    axe.set_title(f"Vorticité et lignes de courant — t = {image['t']:.2f}")
    fig.colorbar(carte, ax=axe, shrink=0.9, label=r"$\omega_y$")
    return fig


def gif_vorticite(exp, prog, debut):
    """GIF animé (téléchargeable) de la vorticité ; la barre de progression avance de debut à 100 %."""
    pngs = []
    with VERROU:
        for k, image in enumerate(exp["images"]):
            prog(debut + (1.0 - debut) * k / len(exp["images"]), f"GIF : image {k + 1}/{len(exp['images'])}")
            # serre=False : dimensions identiques ; dpi 90 : fichier léger.
            pngs.append(png(figure_vorticite(image, exp), dpi=90, serre=False))
    return gif_depuis_png(pngs)


def rendu_volumique(snaps, Z_list, demi, r_max, titre):
    """Rendu volumique de l'enstrophie ½ω² reconstruite par révolution, une vue par instantané."""
    images, titres = [], []
    for snap, Zc in zip(snaps, Z_list):
        # Volume 3D (révolution autour de l'axe) puis lancer de rayons à émission–absorption.
        vol, origine, pas = v3.axisym_enstrophy_volume(snap, Zc, half_width=demi, r_max=r_max, resolution=90)
        images.append(v3.volume_render(vol, origine, pas, elev=18, azim=-60, image_size=(300, 240)))
        titres.append(f"t = {snap['t']:.1f}")
    return v3.plot_volume_renderings(images, titres, title=titre)


def bilan_energie(d):
    """Énergie prévue par le bilan exact dE/dt = −ν∫ω² dV et résidu relatif max |E − E_bilan| / E₀."""
    # cumulative_trapezoid(..., initial=0) : intégrale cumulée de la dissipation depuis t = 0.
    E_th = d["energy"][0] - cumulative_trapezoid(d["dissipation"], d["t"], initial=0.0)
    return E_th, float(np.abs(d["energy"] - E_th).max() / d["energy"][0])


def courbe_bilan(d):
    """Courbe interactive : énergie simulée et énergie E₀ − ∫νΩ dt (bilan exact)."""
    E_th, _ = bilan_energie(d)
    df = pd.DataFrame({"t": d["t"], "E simulée": d["energy"], "E₀ − ∫νΩ dt": E_th})
    return courbe("Énergie cinétique E(t) et bilan visqueux", df, ["E simulée", "E₀ − ∫νΩ dt"], "E",
                  couleurs=viz.SERIES[:2])
