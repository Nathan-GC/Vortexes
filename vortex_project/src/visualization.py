"""
visualization.py — Phase 4 : rendu graphique et exports.

* Champs axisymétriques : vorticité ω_θ (coupe méridienne complète), lignes de courant ψ (repère
  du laboratoire et de l'anneau, bulle tourbillonnaire), vitesse et pression.
* Validation : vitesse de propagation simulée vs formules de Saffman / Fukumoto–Moffatt.
* Méthode lagrangienne : instantanés 3D des filaments, diagnostics saute-mouton / collision,
  animations 3D MP4 (ffmpeg) ou GIF (Pillow, repli automatique).
* Moteurs : vortex bidirectionnel (profils, lignes de courant, trajectoires hélicoïdales,
  temps de séjour), paroi froide, propulseur pulsé (formation, surpression, cartes de rendement).

Conventions graphiques : palette catégorielle à ordre fixe (validée daltonisme), une seule
teinte pour les grandeurs (bleu séquentiel), bleu ↔ gris ↔ rouge pour les grandeurs signées,
traits fins, grilles discrètes, jamais de double axe des ordonnées (petits multiples à la place).
"""

# Annotations non évaluées à l'exécution.
from __future__ import annotations

# os : chemins de fichiers et création de dossiers (os.makedirs, os.path...).
import os
# shutil.which : recherche d'un exécutable (ffmpeg) dans le PATH.
import shutil

# matplotlib : bibliothèque de tracé.
import matplotlib

# matplotlib.use("Agg") : moteur de rendu sans fenêtre (écrit directement des fichiers PNG/MP4) ;
# doit être appelé avant d'importer pyplot, d'où les « noqa: E402 » des imports suivants.
matplotlib.use("Agg")

# pyplot : interface de haut niveau (figures, axes, tracés).
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
# animation : création d'animations (FuncAnimation) et écriture MP4/GIF (FFMpegWriter, PillowWriter).
from matplotlib import animation  # noqa: E402
# LinearSegmentedColormap : carte de couleurs interpolée entre des couleurs données ;
# TwoSlopeNorm : normalisation qui place une valeur donnée (ici 0) exactement au centre de la carte ;
# LogNorm : échelle de couleur logarithmique (champs couvrant plusieurs décades, ex. dissipation).
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
# cumulative_trapezoid : intégrale cumulée (bilans d'énergie).
from scipy.integrate import cumulative_trapezoid  # noqa: E402

# Module théorique du même paquet (vortex de Hill, énergie adimensionnelle du modèle slug).
from . import theory  # noqa: E402
# Diagnostics du solveur axisymétrique (dissipation Φ, enstrophie, intégrales volumiques).
from . import solver_2d_axisymmetric as ax2d  # noqa: E402

# Noms publics du module.
__all__ = [
    "SERIES", "apply_style", "save_figure",
    "plot_axisymmetric_fields", "plot_vorticity_snapshots", "plot_ring_speed_validation", "plot_hill_vortex",
    "plot_filament_snapshots", "animate_filaments_3d", "plot_leapfrog_diagnostics", "plot_collision_diagnostics",
    "plot_kernel_convergence",
    "plot_bidirectional_vortex", "plot_cold_wall", "plot_reactor_conversion", "plot_thruster_formation",
    "plot_efficiency_map",
    "plot_dissipation_map", "plot_energy_enstrophy_budget", "plot_hovmoller", "plot_hovmoller_density",
    "plot_phase_diagram",
]

# ---------------------------------------------------------------------------------------------
# Palette (ordre fixe, jamais cyclé) et encre
# ---------------------------------------------------------------------------------------------
# Couleurs des séries (bleu, orange, turquoise, jaune, magenta, vert, violet, rouge), toujours
# attribuées dans cet ordre pour rester distinguables par les daltoniens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Couleurs d'« encre » : textes, axes, grilles, fond, gris neutre.
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "axis": "#c3c2b7", "surface": "#fcfcfb", "neutral": "#f0efec"}
# Rampe bleue (clair → foncé) pour les grandeurs positives (cartes séquentielles).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Rampe orange pour une deuxième grandeur séquentielle affichée en même temps.
ORANGE_RAMP = ["#fde3d6", "#f9c2a6", "#f39c76", "#eb6834", "#c9531f", "#9f4016", "#722d0e"]

# LinearSegmentedColormap.from_list(nom, couleurs) : carte continue interpolant la liste.
CMAP_SEQ = LinearSegmentedColormap.from_list("vortex_blue", BLUE_RAMP)
CMAP_SEQ2 = LinearSegmentedColormap.from_list("vortex_orange", ORANGE_RAMP)
# divergente : deux bras de 5 pas autour d'un gris neutre placé exactement au centre (valeur 0)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "vortex_div", ["#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#b7d3f6", INK["neutral"],
                   "#f5c4c3", "#ee8e8d", "#e34948", "#b0302f", "#7a1c1b"])
# with_extremes(bad=...) : copie de la palette dont la couleur des valeurs invalides (NaN ou masquées)
# est le gris neutre (remplace set_bad, appelé à disparaître des versions récentes de matplotlib).
CMAP_SEQ = CMAP_SEQ.with_extremes(bad=INK["neutral"])
CMAP_SEQ2 = CMAP_SEQ2.with_extremes(bad=INK["neutral"])


def ordinal_blues(n):
    """n pas ordonnés de la rampe bleue (du clair lisible au foncé) pour des séries ordonnées."""
    # n indices répartis de 3 (premier bleu assez contrasté) à la fin de la rampe ;
    # .round().astype(int) arrondit et convertit en entiers.
    idx = np.linspace(3, len(BLUE_RAMP) - 1, n).round().astype(int)
    # Compréhension de liste : couleurs correspondantes.
    return [BLUE_RAMP[i] for i in idx]


def apply_style():
    # plt.rcParams.update(dict) : modifie les réglages par défaut de toutes les figures suivantes.
    plt.rcParams.update({
        # Couleurs de fond de la figure, des axes et du fichier enregistré.
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
        # Police (Segoe UI sous Windows, DejaVu Sans en repli), taille, police des formules LaTeX.
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "mathtext.fontset": "dejavusans",
        # Couleurs des axes, des étiquettes et des titres.
        "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
        # Titres de panneaux : taille, gras, alignés à gauche.
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        # Suppression des cadres haut/droit et grille activée.
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        # Grille discrète, dessinée sous les courbes.
        "grid.color": INK["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        # Couleurs des graduations et de leurs libellés.
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        # Épaisseur des courbes, taille des marqueurs, légende sans cadre.
        "lines.linewidth": 1.8, "lines.markersize": 6, "legend.frameon": False,
        "legend.labelcolor": INK["secondary"], "legend.fontsize": 8.5,
        # Couleur du texte, résolution à l'écran et à l'enregistrement.
        "text.color": INK["primary"], "figure.dpi": 110, "savefig.dpi": 150,
        # matplotlib.cycler : ordre des couleurs par défaut = palette SERIES ; les niveaux négatifs
        # des contours sont tracés en trait plein (et non en tirets par défaut).
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "contour.negative_linestyle": "solid",
    })


# Le style est appliqué dès l'import du module.
apply_style()


def save_figure(fig, path, dpi=150):
    """Enregistre la figure (crée le dossier) et la ferme ; renvoie le chemin."""
    # os.makedirs(..., exist_ok=True) : crée le dossier de destination s'il n'existe pas
    # (os.path.abspath : chemin absolu ; os.path.dirname : dossier parent).
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # fig.savefig : écrit le fichier ; bbox_inches="tight" rogne les marges inutiles ; facecolor reprend
    # la couleur de fond de la figure (indispensable pour les rendus sur fond sombre).
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    # plt.close libère la mémoire de la figure.
    plt.close(fig)
    return path


def _suptitle(fig, text):
    # fig.suptitle : titre général de la figure, placé à gauche (x = 0.01, ha = alignement horizontal).
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK["primary"])


def _no_grid(ax):
    # ax.grid(False) : supprime la grille (inutile sur les cartes de couleur).
    ax.grid(False)


# =============================================================================================
# Champs axisymétriques
# =============================================================================================
def _mirror(r, field, odd=False):
    """Coupe x–z complète à partir d'un champ défini pour r ≥ 0 (odd : composante antisymétrique)."""
    # r[::-1] inverse l'ordre ; on construit l'axe x = [-r (décroissant), +r].
    rr = np.concatenate([-r[::-1], r])
    # Composante impaire (vorticité hors plan, vitesse radiale) : changement de signe de l'autre côté.
    sign = -1.0 if odd else 1.0
    # Champ symétrisé : lignes inversées (miroir) puis champ original, empilés selon l'axe 0 (r).
    ff = np.concatenate([sign * field[::-1], field], axis=0)
    return rr, ff


def _window(z, zc, half):
    """Indices périodiques d'une fenêtre [zc - half, zc + half] et coordonnées déroulées."""
    # Période du domaine = pas × nombre de points.
    period = (z[1] - z[0]) * len(z)
    # Écart périodique au centre de la fenêtre, ramené dans [-période/2, période/2).
    dz = (z - zc + 0.5 * period) % period - 0.5 * period
    # np.argsort : indices qui trient les écarts par ordre croissant.
    order = np.argsort(dz)
    # On ne garde que les points dans la fenêtre, triés.
    keep = order[np.abs(dz[order]) <= half]
    # Indices à extraire et coordonnées z continues (« déroulées ») correspondantes.
    return keep, zc + dz[keep]


def plot_axisymmetric_fields(snap, Z_ring=None, U_ring=None, half_width=2.6, r_max=2.4, title=None):
    """Planche 2×2 d'un instantané du solveur axisymétrique (coupe méridienne x–z complète).

    (a) vorticité hors-plan ω_y = ±ω_θ ; (b) lignes de courant dans le repère de l'anneau
    ψ' = ψ - U r²/2 (la séparatrice ψ' = 0 délimite la bulle) ; (c) norme de la vitesse et
    vecteurs ; (d) pression p/ρ.
    """
    # Coordonnées (faces et centres) et champs de l'instantané.
    rf, zf, rc, zc = snap["r_f"], snap["z_f"], snap["r_c"], snap["z_c"]
    om, psi = snap["omega"], snap["psi"]
    # Position de l'anneau estimée si non fournie : colonne z de moment ∬ω r² maximal.
    if Z_ring is None:
        # np.clip(om, 0, None) : ne garde que la vorticité positive.
        w = np.clip(om, 0, None) * rf[:, None] ** 2
        Z_ring = float(zf[np.argmax(w.sum(axis=0))])
    # Fenêtres en z centrées sur l'anneau (coins et centres).
    kf, zf_w = _window(zf, Z_ring, half_width)
    kc, zc_w = _window(zc, Z_ring, half_width)
    # Masques des rayons affichés.
    irf = rf <= r_max
    irc = rc <= r_max
    # plt.subplots(2, 2) : figure et grille 2×2 d'axes ; constrained_layout ajuste les espacements.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), constrained_layout=True)
    # (a) vorticité
    ax = axes[0, 0]
    # Coupe complète : vorticité antisymétrique de part et d'autre de l'axe.
    rr, oo = _mirror(rf[irf], om[irf][:, kf], odd=True)
    # Échelle de couleur symétrique ±|ω|max.
    vmax = np.abs(om).max()
    # ax.pcolormesh(X, Y, C) : carte de couleurs ; TwoSlopeNorm centre 0 sur le gris neutre ;
    # shading="gouraud" lisse entre les nœuds ; rasterized allège les fichiers vectoriels.
    im = ax.pcolormesh(zf_w, rr, oo, cmap=CMAP_DIV, norm=TwoSlopeNorm(0.0, -vmax, vmax), shading="gouraud",
                       rasterized=True)
    # fig.colorbar : barre de couleur associée (shrink = réduction de sa hauteur).
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\omega_y$  (= $\pm\,\omega_\theta$)")
    ax.set_title("(a) Vorticité (coupe méridienne)")
    # (b) lignes de courant repère anneau (l'axe r = 0, où ψ' = 0, est exclu pour ne pas tracer
    # la portion axiale de la séparatrice)
    ax = axes[0, 1]
    # Vitesse de l'anneau (0 si inconnue).
    U = 0.0 if U_ring is None else U_ring
    # ψ' = ψ - U r²/2 : fonction de courant dans le repère qui suit l'anneau.
    psi_ring = psi - 0.5 * U * rf[:, None] ** 2
    # Masque des rayons affichés, axe exclu.
    irf_b = irf & (rf > 0)
    rr, pp = _mirror(rf[irf_b], psi_ring[irf_b][:, kf])
    # Niveaux de contour du 1er centile (np.percentile) au maximum.
    lev = np.linspace(np.percentile(pp, 1), pp.max(), 26)
    # ax.contourf : remplissage de la région ψ' > 0 (la bulle).
    ax.contourf(zf_w, rr, pp, levels=[0.0, pp.max() * 1.01], colors=[BLUE_RAMP[1]], alpha=0.9)
    # ax.contour : lignes de courant (niveaux non nuls).
    ax.contour(zf_w, rr, pp, levels=lev[np.abs(lev) > 1e-9 * pp.max()], colors=INK["secondary"], linewidths=0.7)
    # Séparatrice ψ' = 0 (frontière de la bulle) en trait épais.
    ax.contour(zf_w, rr, pp, levels=[0.0], colors=[SERIES[0]], linewidths=1.8)
    ax.set_title(f"(b) Lignes de courant, repère de l'anneau (U = {U:.3f}) ;\n     en bleu : bulle tourbillonnaire ψ' > 0")
    # (c) vitesse
    ax = axes[1, 0]
    # Composantes de vitesse aux centres dans la fenêtre.
    ur = snap["u_r"][irc][:, kc]
    uz = snap["u_z"][irc][:, kc]
    # np.hypot(a, b) = √(a² + b²) : norme de la vitesse.
    speed = np.hypot(ur, uz)
    rr, ss = _mirror(rc[irc], speed)
    im = ax.pcolormesh(zc_w, rr, ss, cmap=CMAP_SEQ, shading="gouraud", rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|\mathbf{u}|$")
    # Composantes miroir : u_x antisymétrique, u_z symétrique (« _ » ignore l'axe déjà connu).
    _, ux_m = _mirror(rc[irc], ur, odd=True)
    _, uz_m = _mirror(rc[irc], uz)
    # Pas de sous-échantillonnage pour ne dessiner qu'environ 26 × 30 flèches.
    sr = max(1, len(rr) // 26)
    sz = max(1, len(zc_w) // 30)
    # ax.quiver(X, Y, U, V) : champ de flèches ; « [::s] » prend un point sur s ;
    # scale fixe la longueur des flèches (np.nanmax ignore d'éventuels NaN).
    ax.quiver(zc_w[::sz], rr[::sr], uz_m[::sr, ::sz], ux_m[::sr, ::sz], color=INK["primary"], width=0.0022,
              scale=18 * np.nanmax(ss), alpha=0.75)
    ax.set_title("(c) Champ de vitesse (repère du laboratoire)")
    # (d) pression
    ax = axes[1, 1]
    rr, pr = _mirror(rc[irc], snap["p"][irc][:, kc])
    # Pression relative à sa médiane (≈ valeur lointaine) ; np.median = médiane.
    pr = pr - np.median(pr)
    pm = np.abs(pr).max()
    im = ax.pcolormesh(zc_w, rr, pr, cmap=CMAP_DIV, norm=TwoSlopeNorm(0.0, -pm, pm), shading="gouraud",
                       rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$p/\rho$ (référence : médiane)")
    ax.set_title("(d) Pression — dépression dans les cœurs")
    # Réglages communs aux 4 panneaux (axes.flat parcourt la grille 2×2 à plat).
    for ax in axes.flat:
        # Même échelle en x et en z (géométrie non déformée).
        ax.set_aspect("equal")
        ax.set_xlabel("z")
        ax.set_ylabel("x  (r en miroir)")
        _no_grid(ax)
        # ax.axhline : droite horizontale (ici l'axe de symétrie x = 0, en pointillés).
        ax.axhline(0.0, color=INK["axis"], lw=0.8, ls=":")
    _suptitle(fig, title or f"Anneau tourbillonnaire axisymétrique — t = {snap['t']:.2f}")
    return fig


def plot_vorticity_snapshots(snapshots, Z_list, half_width=2.2, r_max=2.2, title=None):
    """Vorticité ω_y à plusieurs instants (même échelle de couleur)."""
    n = len(snapshots)
    # squeeze=False : « axes » reste un tableau 2D même avec une seule ligne.
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.9), constrained_layout=True, squeeze=False)
    # Échelle commune : maximum de |ω| sur tous les instantanés (expression génératrice dans max).
    vmax = max(np.abs(s["omega"]).max() for s in snapshots)
    im = None
    # zip parcourt en parallèle les axes, les instantanés et les positions d'anneau.
    for ax, s, Zc in zip(axes[0], snapshots, Z_list):
        rf, zf = s["r_f"], s["z_f"]
        kf, zw = _window(zf, Zc, half_width)
        irf = rf <= r_max
        rr, oo = _mirror(rf[irf], s["omega"][irf][:, kf], odd=True)
        im = ax.pcolormesh(zw, rr, oo, cmap=CMAP_DIV, norm=TwoSlopeNorm(0.0, -vmax, vmax), shading="gouraud",
                           rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"t = {s['t']:.1f}")
        ax.set_xlabel("z")
        _no_grid(ax)
    axes[0, 0].set_ylabel("x  (r en miroir)")
    # Une seule barre de couleur partagée par tous les panneaux (ax=liste d'axes).
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, label=r"$\omega_y$")
    _suptitle(fig, title or "Diffusion visqueuse du cœur et sillage")
    return fig


def plot_ring_speed_validation(res, title=None):
    """Petits multiples : U(t) simulée vs théories, écart relatif, rayon de cœur σ(t), bilan d'énergie."""
    d = res["diagnostics"]
    t = d["t"]
    p = res["params"]
    # Masque booléen des instants postérieurs au transitoire initial.
    skip = t > 0.5  # écarte le transitoire d'ajustement initial du cœur
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    # (a) vitesse simulée et théoriques (ls = style de trait : "--" tirets, "-." tiret-point).
    ax = axes[0, 0]
    ax.plot(t[skip], res["U_sim"][skip], color=SERIES[0], label="simulation (dZ/dt)")
    ax.plot(t, res["U_saffman"], color=SERIES[1], ls="--", label="Saffman (1970)")
    ax.plot(t, res["U_saffman_fm"], color=SERIES[2], ls="-.", label="Saffman + Fukumoto–Moffatt (2000)")
    ax.set_xlabel("t")
    ax.set_ylabel("vitesse de propagation U")
    ax.set_title("(a) Vitesse de l'anneau")
    # ax.legend : légende des courbes munies d'un « label ».
    ax.legend(loc="upper right")
    # (b) écarts relatifs en pourcentage.
    ax = axes[0, 1]
    e1 = (res["U_sim"] - res["U_saffman"]) / res["U_saffman"] * 100
    e2 = (res["U_sim"] - res["U_saffman_fm"]) / res["U_saffman_fm"] * 100
    ax.plot(t[skip], e1[skip], color=SERIES[1], ls="--", label="vs Saffman")
    ax.plot(t[skip], e2[skip], color=SERIES[2], ls="-.", label="vs Saffman + F–M")
    ax.axhline(0.0, color=INK["axis"], lw=1.0)
    ax.set_xlabel("t")
    ax.set_ylabel("écart relatif  (%)")
    ax.set_title("(b) Écart simulation – théorie")
    ax.legend(loc="lower left")
    # (c) rayon de cœur mesuré et théorique.
    ax = axes[1, 0]
    ax.plot(t, d["sigma"], color=SERIES[0], label="σ mesuré (cœur : ω > 0,1 ω_max, corrigé de la troncature)")
    ax.plot(t, res["sigma_theory"], color=SERIES[1], ls="--", label=r"$\sqrt{a^2 + 4\nu t}$")
    ax.set_xlabel("t")
    ax.set_ylabel("rayon de cœur σ")
    ax.set_title("(c) Épaississement du cœur")
    ax.legend(loc="upper left")
    # (d) bilan d'énergie.
    ax = axes[1, 1]
    # Import local de l'intégrale cumulée par trapèzes.
    from scipy.integrate import cumulative_trapezoid
    E = d["energy"]
    # E0 - ∫ dissipation dt : énergie attendue si la seule perte est visqueuse.
    budget = E[0] - cumulative_trapezoid(d["dissipation"], t, initial=0.0)
    ax.plot(t, E, color=SERIES[0], label="énergie cinétique E(t)")
    ax.plot(t, budget, color=SERIES[1], ls="--", label=r"$E_0 - \int_0^t 2\pi\nu\!\iint \omega^2 r\,dr\,dz\;dt$")
    ax.set_xlabel("t")
    ax.set_ylabel("E / ρ")
    ax.set_title("(d) Bilan d'énergie : aucune dissipation numérique")
    ax.legend(loc="upper right")
    # Titre général ; « title or (...) » utilise le titre fourni sinon le texte par défaut.
    _suptitle(fig, title or (f"Validation du solveur axisymétrique — Γ = {p['Gamma']:g}, R = {p['R']:g}, "
                             f"a = {p['a']:g}, Re_Γ = {p['Re']:g}"))
    return fig


def plot_hill_vortex(a=1.0, U=1.0, n=301, title=None):
    """Vortex de Hill : lignes de courant (repères du vortex et du laboratoire), vorticité, pression."""
    # Grille carrée de la coupe méridienne (x de -2.2a à 2.2a, z idem).
    x = np.linspace(-2.2 * a, 2.2 * a, n)
    zz = np.linspace(-2.2 * a, 2.2 * a, n)
    # np.meshgrid(zz, x) : Z varie selon les colonnes, X selon les lignes.
    Z, X = np.meshgrid(zz, x)
    # Rayon cylindrique r = |x| (la coupe x–z contient l'axe).
    R = np.abs(X)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
    # Angles pour tracer le cercle frontière de la sphère.
    circ = np.linspace(0, 2 * np.pi, 400)
    # Deux premiers panneaux : lignes de courant dans les deux repères (enumerate → (k, repère)).
    for k, frame in enumerate(("body", "lab")):
        ax = axes[k]
        psi = theory.hill_stream_function(R, Z, a, U, frame=frame)
        lev = np.linspace(psi.min(), psi.max(), 40)
        ax.contour(Z, X, psi, levels=lev, colors=INK["secondary"], linewidths=0.7)
        # Frontière de la sphère de rayon a.
        ax.plot(a * np.cos(circ), a * np.sin(circ), color=SERIES[0], lw=1.8)
        # Titre selon le repère (expression conditionnelle « A if condition else B »).
        ax.set_title("(a) Repère du vortex (écoulement −U à l'infini)" if frame == "body"
                     else "(b) Repère du laboratoire (vortex se déplaçant à +U)")
    # Troisième panneau : coefficient de pression C_p = (p - p∞)/(½ρU²).
    ax = axes[2]
    pr = theory.hill_pressure(R, Z, a, U) / (0.5 * U**2)
    im = ax.pcolormesh(Z, X, pr, cmap=CMAP_DIV, norm=TwoSlopeNorm(0.0, pr.min(), pr.max()), shading="gouraud",
                       rasterized=True)
    ax.contour(Z, X, pr, levels=np.linspace(pr.min(), pr.max(), 15), colors=INK["secondary"], linewidths=0.5)
    ax.plot(a * np.cos(circ), a * np.sin(circ), color=SERIES[0], lw=1.2, ls="--")
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$C_p = (p - p_\infty)/(\frac{1}{2}\rho U^2)$")
    ax.set_title("(c) Pression : minimum −29/16 au centre du cœur")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("z / a")
        ax.set_ylabel("x / a")
        _no_grid(ax)
    _suptitle(fig, title or "Vortex sphérique de Hill — solution analytique")
    return fig


# =============================================================================================
# Méthode lagrangienne
# =============================================================================================
def _ring_colors(ring_ids):
    # Identifiants d'anneaux uniques et triés.
    uniq = sorted(set(ring_ids))
    # Dictionnaire identifiant → couleur de la palette (modulo si plus de 8 anneaux).
    return {rid: SERIES[i % len(SERIES)] for i, rid in enumerate(uniq)}


def _frame_bounds(frames, pad=0.15):
    # Tous les nœuds de toutes les images empilés dans un seul tableau (N, 3).
    allp = np.concatenate([np.concatenate(fr, axis=0) for fr in frames], axis=0)
    # Boîte englobante (minimum et maximum par coordonnée).
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    # Plus grande dimension : on fabrique une boîte cubique (aspect 1:1:1).
    span = (hi - lo).max()
    ctr = 0.5 * (lo + hi)
    # Demi-côté avec une marge de 15 %.
    half = 0.5 * span * (1 + pad)
    return ctr - half, ctr + half


def _draw_filaments(ax, frame, ring_ids, colors, lw=1.6):
    lines = []
    for nodes, rid in zip(frame, ring_ids):
        # On répète le premier nœud en fin de liste pour fermer visuellement le polygone.
        closed = np.vstack([nodes, nodes[:1]])
        # ax.plot(x, y, z) en 3D renvoie une liste d'une ligne ; « (ln,) = ... » la dépaquette.
        (ln,) = ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=colors[rid], lw=lw)
        lines.append(ln)
    # Lignes renvoyées pour pouvoir les mettre à jour dans une animation.
    return lines


def _style_3d(ax, lo, hi):
    # Limites des trois axes.
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # set_box_aspect : boîte d'affichage cubique (échelles identiques).
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Style des « panneaux » de fond 3D et de leur grille (attribut interne _axinfo de mplot3d).
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(INK["surface"])
        axis.pane.set_edgecolor(INK["grid"])
        axis._axinfo["grid"]["color"] = INK["grid"]
        axis._axinfo["grid"]["linewidth"] = 0.5


def plot_filament_snapshots(hist, n=4, elev=18, azim=-62, title=None, labels=None, follow=False, window=3.2):
    """Instantanés 3D des filaments à n instants régulièrement espacés.

    ``follow`` : chaque vue est centrée sur le barycentre des filaments (fenêtre de côté ``window``),
    sinon une boîte commune englobe toute la trajectoire.
    """
    # Indices de n images régulièrement espacées dans l'historique.
    idx = np.linspace(0, len(hist.frames) - 1, n).round().astype(int)
    colors = _ring_colors(hist.ring_ids)
    # Boîte commune à toutes les images sélectionnées.
    lo, hi = _frame_bounds([hist.frames[i] for i in idx])
    # plt.figure : nouvelle figure vide (les axes 3D sont ajoutés un par un).
    fig = plt.figure(figsize=(3.6 * n, 4.3))
    for k, i in enumerate(idx):
        # fig.add_subplot(1, n, k+1, projection="3d") : k-ième axe 3D d'une rangée de n.
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        _draw_filaments(ax, hist.frames[i], hist.ring_ids, colors)
        if follow:
            # Fenêtre centrée sur le barycentre des nœuds (moyenne selon l'axe 0).
            c = np.concatenate(hist.frames[i], axis=0).mean(axis=0)
            _style_3d(ax, c - 0.5 * window, c + 0.5 * window)
        else:
            _style_3d(ax, lo, hi)
        # ax.view_init : angle de vue (élévation et azimut en degrés).
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"t = {hist.times[i]:.1f}", loc="center", pad=0)
    # Marges de la figure réglées manuellement (espace pour le titre et la légende).
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.84, wspace=0.02)
    if labels:
        # plt.Line2D([], [], ...) : ligne « fantôme » servant seulement d'entrée de légende.
        handles = [plt.Line2D([], [], color=colors[rid], lw=2) for rid in sorted(colors)]
        # fig.legend : légende au niveau de la figure (bbox_to_anchor = position de l'ancre).
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.99, 0.99))
    _suptitle(fig, title or "Filaments tourbillonnaires")
    return fig


def animate_filaments_3d(hist, path, fps=24, dpi=100, elev=18, azim=-62, rotate_deg=0.0, follow=False,
                         window=None, title=None, stride=1):
    """Anime les filaments en 3D et exporte en MP4 (ffmpeg) ou GIF (Pillow).

    ``follow`` : la fenêtre suit le barycentre des anneaux (utile pour le saute-mouton).
    Si ffmpeg est indisponible, un chemin .mp4 est remplacé par .gif. Renvoie le chemin écrit.
    """
    # Sous-échantillonnage des images (une sur « stride »).
    frames = hist.frames[::stride]
    times = hist.times[::stride]
    colors = _ring_colors(hist.ring_ids)
    lo, hi = _frame_bounds(frames)
    if follow:
        # Côté de la fenêtre mobile : valeur fournie, sinon étendue horizontale maximale.
        span = window if window is not None else 0.5 * (hi - lo)[:2].max() * 2.0
        half = 0.5 * span
    fig = plt.figure(figsize=(6.4, 6.0))
    # add_subplot(111, ...) : un seul axe (1 ligne, 1 colonne, position 1), en 3D.
    ax = fig.add_subplot(111, projection="3d")
    # Lignes de la première image (mises à jour ensuite).
    lines = _draw_filaments(ax, frames[0], hist.ring_ids, colors)
    _style_3d(ax, lo, hi)
    ax.view_init(elev=elev, azim=azim)
    # Objet texte du titre, modifié à chaque image.
    ttl = ax.set_title("", loc="left")
    head = title or "Anneaux tourbillonnaires"

    # Fonction appelée par FuncAnimation pour dessiner l'image numéro k.
    def update(k):
        fr = frames[k]
        for ln, nodes in zip(lines, fr):
            closed = np.vstack([nodes, nodes[:1]])
            # set_data_3d : remplace les coordonnées d'une ligne 3D existante.
            ln.set_data_3d(closed[:, 0], closed[:, 1], closed[:, 2])
        if follow:
            # Recentrage de la fenêtre sur le barycentre courant.
            c = np.concatenate(fr, axis=0).mean(axis=0)
            ax.set_xlim(c[0] - half, c[0] + half)
            ax.set_ylim(c[1] - half, c[1] + half)
            ax.set_zlim(c[2] - half, c[2] + half)
        if rotate_deg:
            # Rotation progressive de la caméra (rotate_deg degrés au total).
            ax.view_init(elev=elev, azim=azim + rotate_deg * k / max(1, len(frames) - 1))
        ttl.set_text(f"{head}   t = {times[k]:.2f}")
        return lines

    # animation.FuncAnimation : appelle update(k) pour k = 0..len(frames)-1 ; interval en ms.
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    # Dossier de destination.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # MP4 si ffmpeg est disponible (writers.is_available ou exécutable trouvé par shutil.which).
    if path.lower().endswith(".mp4") and (animation.writers.is_available("ffmpeg") or shutil.which("ffmpeg")):
        # Encodeur H.264 ; « -pix_fmt yuv420p » assure la lecture par la plupart des lecteurs vidéo.
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    else:
        # Repli : GIF animé (os.path.splitext sépare le nom de son extension).
        path = os.path.splitext(path)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
    # anim.save : calcule toutes les images et écrit le fichier vidéo.
    anim.save(path, writer=writer, dpi=dpi)
    plt.close(fig)
    return path


def plot_leapfrog_diagnostics(hist, model=None, title=None):
    """Saute-mouton : positions axiales et rayons des deux anneaux (filaments vs modèle coaxial),
    et conservation des invariants."""
    # Diagnostics convertis en tableaux numpy.
    d = hist.as_arrays()
    t = d["t"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    # Nombre d'anneaux (nombre de colonnes du tableau des rayons).
    n_rings = d["radius"].shape[1]
    for i in range(n_rings):
        # Simulation 3D en trait plein.
        axes[0].plot(t, d["axial"][:, i], color=SERIES[i])
        axes[1].plot(t, d["radius"][:, i], color=SERIES[i])
        # Modèle coaxial en tirets (même couleur que l'anneau).
        if model is not None:
            axes[0].plot(model["t"], model["Z"][i], color=SERIES[i], ls="--", lw=1.2)
            axes[1].plot(model["t"], model["R"][i], color=SERIES[i], ls="--", lw=1.2)
    axes[0].set_ylabel("position axiale Z")
    axes[0].set_title("(a) Positions : les anneaux se dépassent")
    axes[1].set_ylabel("rayon R")
    axes[1].set_title("(b) Rayons : contraction / dilatation alternées")
    # Composante z de l'impulsion (axe de propagation).
    P = d["impulse"][:, 2]
    # Dérive relative de l'impulsion (+1e-16 pour pouvoir tracer en échelle log).
    axes[2].plot(t[1:], np.abs(P[1:] - P[0]) / abs(P[0]) + 1e-16, color=SERIES[0], label="impulsion  |ΔP|/P₀")
    E = d["energy"]
    # Masque des instants où l'énergie a été calculée (np.isfinite exclut les NaN).
    ok = np.isfinite(E)
    if ok.sum() > 2:
        te, Ee = t[ok][1:], E[ok]
        axes[2].plot(te, np.abs(Ee[1:] - Ee[0]) / abs(Ee[0]) + 1e-16, color=SERIES[1], ls="--",
                     marker="o", ms=3.5, label="énergie  |ΔE|/E₀")
    # Échelle logarithmique en ordonnée et bornes fixes.
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-7, 1e-1)
    axes[2].set_ylabel("dérive relative")
    axes[2].set_title("(c) Invariants (RK4, noyau de Rosenhead–Moore)")
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.set_xlabel("t")
    # Légende commune : couleurs = anneaux, styles de trait = méthode.
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2) for i in range(n_rings)]
    labels = [f"anneau {'AB'[i]}" for i in range(n_rings)]
    handles += [plt.Line2D([], [], color=INK["secondary"], lw=1.8),
                plt.Line2D([], [], color=INK["secondary"], lw=1.2, ls="--")]
    labels += ["filaments 3D (trait plein)", "modèle coaxial régularisé (tirets)"]
    # loc="outside lower center" : légende sous les panneaux (possible avec constrained_layout).
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    _suptitle(fig, title or "Saute-mouton (leapfrogging) de deux anneaux coaxiaux")
    return fig


def plot_collision_diagnostics(hist, model=None, delta=None, title=None):
    """Collision frontale : rayons et positions des anneaux vs modèle coaxial.

    Si ``delta`` est donné, la zone où les deux filaments sont à moins de 2δ l'un de l'autre est
    grisée : la régularisation y domine (en fluide réel, les cœurs s'annihilent par diffusion).
    """
    d = hist.as_arrays()
    t = d["t"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    n_rings = d["radius"].shape[1]
    for i in range(n_rings):
        axes[0].plot(t, d["radius"][:, i], color=SERIES[i])
        axes[1].plot(t, d["axial"][:, i], color=SERIES[i])
        if model is not None:
            axes[0].plot(model["t"], model["R"][i], color=SERIES[i], ls="--", lw=1.2)
            axes[1].plot(model["t"], model["Z"][i], color=SERIES[i], ls="--", lw=1.2)
    if delta is not None and n_rings == 2:
        # Écart axial entre les deux anneaux.
        gap = np.abs(d["axial"][:, 1] - d["axial"][:, 0])
        close = gap < 2.0 * delta
        if close.any():
            # Premier instant où l'écart passe sous 2δ (np.argmax renvoie le premier True).
            t_close = t[np.argmax(close)]
            for ax in axes:
                # ax.axvspan : bande verticale grisée de t_close à la fin ; zorder=0 la met au fond.
                ax.axvspan(t_close, t[-1], color=INK["neutral"], zorder=0)
                # Texte positionné en coordonnées mixtes (x en données, y en fraction de l'axe).
                ax.text(t_close + 0.02 * (t[-1] - t[0]), 0.97, "écart < 2δ :\nrégularisation\ndominante",
                        transform=ax.get_xaxis_transform(), va="top", fontsize=7.5, color=INK["secondary"])
    axes[0].set_ylabel("rayon R")
    axes[0].set_title("(a) Étirement radial à l'approche")
    axes[1].set_ylabel("position axiale Z")
    axes[1].set_title("(b) Ralentissement près du plan de symétrie")
    for ax in axes:
        ax.set_xlabel("t")
    # Légende commune (couleurs = anneaux, styles = méthode).
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2) for i in range(n_rings)]
    labels = ["anneau A (Γ > 0)", "anneau B (Γ < 0)"][:n_rings]
    handles += [plt.Line2D([], [], color=INK["secondary"], lw=1.8),
                plt.Line2D([], [], color=INK["secondary"], lw=1.2, ls="--")]
    labels += ["filaments 3D", "modèle coaxial régularisé"]
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    _suptitle(fig, title or "Collision frontale de deux anneaux (méthode des images)")
    return fig


def plot_kernel_convergence(table, title=None):
    """Erreur relative de la vitesse propre discrète vs Γ/(4πR)[ln(8R/δ) - 1] en fonction de Δs/δ."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    # table : dictionnaire {nom du noyau: liste de (Δs/δ, erreur relative)}.
    for k, (name, rows) in enumerate(table.items()):
        rows = np.asarray(rows)
        # ax.loglog : axes logarithmiques en x et en y.
        ax.loglog(rows[:, 0], np.abs(rows[:, 1]), color=SERIES[k], marker="o", ms=5,
                  ls="-" if k == 0 else "--", label=name)
    # Droite de référence de pente 2 (convergence d'ordre 2).
    x = np.array([0.12, 1.0])
    ax.loglog(x, 0.05 * x**2, color=INK["muted"], lw=1.0, ls=":", label="pente 2")
    ax.set_xlabel("pas de discrétisation Δs / δ")
    ax.set_ylabel("|U_discret / U_RM − 1|")
    ax.set_title("Convergence de la vitesse propre d'un anneau")
    ax.legend(loc="upper left")
    if title:
        _suptitle(fig, title)
    return fig


# =============================================================================================
# Moteurs à vortex
# =============================================================================================
def plot_bidirectional_vortex(chamber, trajectories, rtd=None, title=None):
    """Chambre à vortex bidirectionnel : profils de vitesse, lignes de courant méridiennes,
    trajectoires hélicoïdales 3D et distribution des temps de séjour."""
    a, L, U = chamber.a, chamber.L, chamber.U
    fig = plt.figure(figsize=(12.5, 9.0), constrained_layout=True)
    # fig.add_gridspec(2, 2) : grille de placement permettant de mélanger axes 2D et 3D.
    gs = fig.add_gridspec(2, 2)
    # (a) profils radiaux en sortie
    ax = fig.add_subplot(gs[0, 0])
    r = np.linspace(1e-3, 1.0, 400) * a
    ur, ut, uz = chamber.velocity(r, L)
    ax.plot(r / a, ut / U, color=SERIES[0], label=r"$u_\theta/U$ (swirl)")
    ax.plot(r / a, uz / U, color=SERIES[1], ls="--", label=r"$u_z/U$ en $z = L$")
    # u_r est petit : multiplié par 10 pour être visible.
    ax.plot(r / a, ur / U * 10, color=SERIES[2], ls="-.", label=r"$10\,u_r/U$")
    # ax.axvline : droite verticale marquant le manteau.
    ax.axvline(1 / np.sqrt(2), color=INK["muted"], lw=1.0, ls=":")
    ax.text(1 / np.sqrt(2) + 0.01, 0.95, "manteau r = a/√2", transform=ax.get_xaxis_transform(),
            color=INK["secondary"], fontsize=8.5, va="top")
    ax.axhline(0, color=INK["axis"], lw=0.8)
    ax.set_xlabel("r / a")
    ax.set_ylabel("vitesse / U")
    ax.set_title(f"(a) Profils de vitesse (κ = {chamber.kappa:.4f})")
    ax.legend(loc="lower left")
    # (b) lignes de courant méridiennes
    ax = fig.add_subplot(gs[0, 1])
    rr = np.linspace(0, a, 200)
    zz = np.linspace(0, L, 300)
    Rg, Zg = np.meshgrid(rr, zz, indexing="ij")
    psi = chamber.stream_function(Rg, Zg)
    # Iso-ψ (niveau 0 exclu par [1:]) dans le plan (z, r).
    ax.contour(Zg / a, Rg / a, psi, levels=np.linspace(0, psi.max(), 16)[1:], colors=INK["secondary"],
               linewidths=0.7)
    ax.axhline(1 / np.sqrt(2), color=SERIES[0], lw=1.4, ls="--")
    ax.set_xlabel("z / a   (fond fermé → tuyère)")
    ax.set_ylabel("r / a")
    ax.set_title("(b) Lignes de courant méridiennes ψ = cste")
    # Fond opaque derrière les annotations (lisibilité par-dessus les lignes).
    box = dict(facecolor=INK["surface"], edgecolor="none", pad=1.5)
    # transform=ax.transAxes : coordonnées en fraction de l'axe (0 à 1).
    ax.text(0.03, 0.9, "vortex externe → fond", transform=ax.transAxes, color=INK["secondary"], fontsize=8.5, bbox=box)
    ax.text(0.03, 0.12, "vortex interne → tuyère", transform=ax.transAxes, color=INK["secondary"], fontsize=8.5, bbox=box)
    _no_grid(ax)
    # (c) trajectoires 3D (cm)
    ax = fig.add_subplot(gs[1, 0], projection="3d")
    th = np.linspace(0, 2 * np.pi, 60)
    # Facteur de conversion m → cm.
    s = 100.0
    # Cercles de la paroi à 4 cotes (np.full_like : tableau de même forme rempli d'une valeur).
    for zc in np.linspace(0, L, 4):
        ax.plot(s * a * np.cos(th), s * a * np.sin(th), np.full_like(th, s * zc), color=INK["axis"], lw=0.6)
    # Génératrices verticales de la paroi (une sur dix).
    for k in range(0, 60, 10):
        ax.plot([s * a * np.cos(th[k])] * 2, [s * a * np.sin(th[k])] * 2, [0, s * L], color=INK["axis"], lw=0.6)
    labels = []
    for i, tr in enumerate(trajectories):
        ax.plot(s * tr["x"], s * tr["y"], s * tr["z"], color=SERIES[i], lw=1.1)
        labels.append(f"r₀ = {tr['r'][0] / a:.2f} a : {tr['revolutions']:.1f} tours, parcours {tr['path_ratio']:.0f} L")
    # Boîte d'affichage aux proportions réelles de la chambre.
    ax.set_box_aspect((1, 1, L / a))
    ax.set_xticks([-s * a, 0, s * a])
    ax.set_yticks([-s * a, 0, s * a])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.view_init(elev=14, azim=-50)
    ax.set_title("(c) Trajectoires hélicoïdales (injection en z = L)")
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2) for i in range(len(trajectories))]
    # Légende placée à gauche de l'axe 3D (bbox_to_anchor négatif).
    ax.legend(handles, labels, loc="center left", fontsize=7.5, bbox_to_anchor=(-0.42, 0.5))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(INK["surface"])
    # (d) DTS
    ax = fig.add_subplot(gs[1, 1])
    # DTS fournie ou calculée.
    rtd = chamber.residence_time_distribution() if rtd is None else rtd
    tmean = rtd["mean"]
    # 40 classes de 0 à 4 fois le temps moyen.
    bins = np.linspace(0, 4 * tmean, 41)
    # ax.hist : histogramme pondéré par les fractions de débit (weights).
    ax.hist(rtd["t"] / tmean, bins=bins / tmean, weights=rtd["weight"], color=SERIES[0], edgecolor=INK["surface"],
            linewidth=1.0)
    ax.axvline(1.0, color=INK["primary"], lw=1.2, ls="--")
    ax.text(1.05, 0.95, f"<t> = V/Q = {tmean * 1e3:.1f} ms\nσ_θ² = {rtd['sigma_theta2']:.2f} (≈ {rtd['n_tanks']:.1f} RPA en série)",
            transform=ax.get_xaxis_transform(), va="top", color=INK["secondary"], fontsize=8.5, bbox=box)
    ax.set_xlabel("t / <t>")
    ax.set_ylabel("fraction du débit")
    ax.set_title("(d) Distribution des temps de séjour (forme close)")
    _suptitle(fig, title or f"Chambre à vortex bidirectionnel — a = {a * 100:.0f} cm, L = {L * 100:.0f} cm, U = {U:g} m/s")
    return fig


def plot_cold_wall(kappa, cold_wall_fn, pe_r_values=None, profile_pe_r=(0.0, 2.0, 5.0, 10.0, 20.0), title=None):
    """Paroi froide : rapport de flux pariétal Φ(Pe κ) et profils de température."""
    # np.geomspace : valeurs régulièrement espacées en échelle logarithmique.
    pe_r_values = np.geomspace(0.05, 60, 120) if pe_r_values is None else pe_r_values
    # Rapport de flux pour chaque Péclet radial (Pe = Pe_r/κ transmis au modèle).
    phi = np.array([cold_wall_fn(p / kappa, kappa)["ratio"] for p in pe_r_values])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    ax = axes[0]
    # Asymptote de Laplace correspondante.
    asym = np.array([cold_wall_fn(p / kappa, kappa)["ratio_asymptotic"] for p in pe_r_values])
    # ax.semilogx : abscisse logarithmique.
    ax.semilogx(pe_r_values, phi, color=SERIES[0], label="modèle 1D (convection radiale + diffusion)")
    # L'asymptote n'est tracée que là où elle est pertinente (Pe·κ > 2).
    big = pe_r_values > 2.0
    ax.semilogx(pe_r_values[big], asym[big], color=SERIES[1], ls="--",
                label=r"asymptote $\ln\sqrt{2}\,Pe\,\kappa\;e^{-0.2406\,Pe\,\kappa}$")
    ax.set_xlabel("nombre de Péclet radial  Pe·κ = |u_r| a / α_t")
    ax.set_ylabel("flux pariétal  Φ = q_w / q_w,conduction")
    ax.set_title("(a) Réduction du flux thermique pariétal")
    ax.legend(loc="lower left")
    ax = axes[1]
    # Couleurs ordonnées (bleus de plus en plus foncés) pour des valeurs croissantes de Pe·κ.
    cols = ordinal_blues(len(profile_pe_r))
    for c, p in zip(cols, profile_pe_r):
        res = cold_wall_fn(p / kappa if p > 0 else 0.0, kappa)
        ax.plot(res["r"], res["theta"], color=c, label=f"Pe·κ = {p:g}  (Φ = {res['ratio']:.2f})")
    ax.set_xlabel("r / a   (manteau → paroi)")
    ax.set_ylabel("θ = (T − T_w) / (T_f − T_w)")
    ax.set_title("(b) Profils de température dans le vortex externe")
    ax.legend(loc="upper right")
    _suptitle(fig, title or "Chambre à vortex : effet « paroi froide » (analogie de transpiration)")
    return fig


def plot_reactor_conversion(rc, title=None):
    """Conversion d'une réaction d'ordre 1 : réacteur piston, parfaitement agité, vortex."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    # Trois courbes de conversion en fonction du nombre de Damköhler (abscisse log).
    ax.semilogx(rc["Da"], rc["plug"], color=SERIES[0], label="piston (même <t>)")
    ax.semilogx(rc["Da"], rc["vortex"], color=SERIES[1], ls="--", label="vortex bidirectionnel (DTS exacte)")
    ax.semilogx(rc["Da"], rc["cstr"], color=SERIES[2], ls="-.", label="parfaitement agité")
    ax.set_xlabel("nombre de Damköhler  Da = <t> / τ_chimie")
    ax.set_ylabel("conversion  X")
    ax.set_title("Macro-mélange : le vortex se situe entre piston et RPA")
    ax.legend(loc="upper left")
    if title:
        _suptitle(fig, title)
    return fig


def plot_thruster_formation(thruster, U_j=1.0, title=None):
    """Formation de l'anneau : énergie adimensionnelle, surpression, bulle, masse ajoutée, vitesse et
    épaisseur de cœur en fonction de L/D (une grandeur par panneau)."""
    # Courses adimensionnelles étudiées.
    ld = np.linspace(0.5, 8.0, 76)
    fs = thruster.formation_number()
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), constrained_layout=True)
    # (a) énergie adimensionnelle du modèle slug et seuil α*.
    ax = axes[0, 0]
    alpha = theory.slug_dimensionless_energy(ld)
    ax.plot(ld, alpha, color=SERIES[0], label=r"modèle slug : $\alpha = \sqrt{\pi/2}\,/\,(L/D)$")
    ax.axhline(thruster.alpha_crit, color=SERIES[1], ls="--", lw=1.4, label=f"α* = {thruster.alpha_crit:g} (Gharib et al. 1998)")
    ax.set_ylim(0, 1.6)
    ax.set_ylabel(r"$E/(\rho^{1/2} I^{1/2} \Gamma^{3/2})$")
    ax.set_title("(a) Nombre de formation (Kelvin–Benjamin)")
    ax.legend(loc="center right")
    # (b) fraction de surpression pour plusieurs co-écoulements.
    ax = axes[0, 1]
    lams = [0.0, 0.2, 0.4, 0.6]
    # [::-1] inverse l'ordre des bleus : λ = 0 (effet maximal) le plus foncé.
    for c, lam in zip(ordinal_blues(len(lams))[::-1], lams):
        ax.plot(ld, thruster.overpressure_fraction(ld, lam), color=c, label=f"λ = U_co/U_j = {lam:g}")
    ax.set_ylabel(r"$\beta = I_{op} / I_{slug}$")
    ax.set_title("(b) Impulsion de surpression (semi-empirique)")
    ax.legend(loc="upper right")
    # Propriétés de l'anneau de tête (calcul de bulle) pour L/D de 1 à 8.
    ldr = np.linspace(1.0, 8.0, 29)  # anneaux assez épais pour une bulle attachée à l'axe
    rings = [thruster.leading_ring(x, U_j) for x in ldr]
    # (c) volume de bulle et bouclage de l'impulsion.
    ax = axes[0, 2]
    ax.plot(ldr, [r["bubble_to_slug_volume"] for r in rings], color=SERIES[0], marker="o", ms=3.5,
            label="volume de bulle / volume éjecté")
    ax.plot(ldr, [r["impulse_closure"] for r in rings], color=SERIES[1], ls="--", marker="s", ms=3.5,
            label=r"$(1 + C_A)\,\rho V_b U / I_{slug}$")
    ax.axhline(1.0, color=INK["axis"], lw=1.0)
    ax.set_ylim(0.8, 1.25)
    ax.set_ylabel("rapport")
    ax.set_title("(c) Bulle et bouclage de l'impulsion")
    ax.legend(loc="upper right")
    # (d) coefficient de masse ajoutée.
    ax = axes[1, 0]
    ax.plot(ldr, [r["added_mass"] for r in rings], color=SERIES[0], marker="o", ms=3.5)
    ax.axhline(0.5, color=INK["muted"], lw=0.8, ls=":")
    ax.text(7.9, 0.55, "sphère (Hill) : 1/2", ha="right", color=INK["secondary"], fontsize=8)
    ax.set_ylabel(r"$C_A$")
    ax.set_title("(d) Masse ajoutée de la bulle (sphéroïde équivalent)")
    # (e) vitesse de l'anneau rapportée à la vitesse du jet.
    ax = axes[1, 1]
    ax.plot(ldr, [r["U_self"] / U_j for r in rings], color=SERIES[0], marker="o", ms=3.5)
    ax.set_ylabel(r"$U_{anneau} / U_j$")
    ax.set_title("(e) Vitesse de l'anneau (Kelvin)")
    # (f) épaisseur relative du cœur.
    ax = axes[1, 2]
    ax.plot(ldr, [r["core_ratio"] for r in rings], color=SERIES[0], marker="o", ms=3.5)
    ax.axhline(0.5, color=INK["muted"], lw=0.8, ls=":")
    ax.text(1.1, 0.52, "limite de validité « anneau mince »", color=INK["secondary"], fontsize=8)
    ax.set_ylabel("a / R")
    ax.set_title("(f) Épaisseur relative du cœur")
    # Réglages communs : abscisse et droite verticale au nombre de formation.
    for ax in axes.flat:
        ax.set_xlabel("L / D")
        ax.axvline(fs, color=INK["muted"], ls=":", lw=1.0)
    axes[0, 0].text(fs + 0.1, 0.92, f"F* = {fs:.2f}", transform=axes[0, 0].get_xaxis_transform(), color=INK["secondary"])
    _suptitle(fig, title or "Propulseur pulsé : formation de l'anneau de tête (lignes pointillées : F*)")
    return fig


def plot_efficiency_map(m, title=None):
    """Cartes du rendement propulsif η(f, L/D) et du gain vs jet continu équivalent."""
    # Axes de la carte et grilles 2D correspondantes.
    f, ld = m["frequency"], m["L_over_D"]
    F, LD = np.meshgrid(f, ld)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    for ax in axes:
        ax.set_facecolor(INK["neutral"])  # cellules infaisables (NaN) : gris neutre
    # (a) rendement propulsif.
    ax = axes[0]
    # np.ma.masked_invalid : masque les NaN (points infaisables, laissés transparents).
    eta = np.ma.masked_invalid(m["efficiency"])
    # shading="nearest" : une couleur uniforme par cellule centrée sur chaque point.
    im = ax.pcolormesh(F, LD, eta, cmap=CMAP_SEQ, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.9, label="rendement propulsif η")
    # Iso-lignes de la vitesse de jet nécessaire.
    cs = ax.contour(F, LD, np.ma.masked_invalid(m["U_j"]), levels=[1.2, 2, 3, 5, 8], colors=INK["primary"],
                    linewidths=0.6)
    # ax.clabel : étiquettes sur les iso-lignes (fmt = fonction de formatage du texte).
    ax.clabel(cs, fmt=lambda v: f"U_j = {v:g} m/s", fontsize=7)
    # Position du maximum (np.nanargmax ignore les NaN ; unravel_index → indices 2D).
    i, j = np.unravel_index(np.nanargmax(m["efficiency"]), m["efficiency"].shape)
    ax.plot(f[j], ld[i], marker="*", ms=12, color=SERIES[1], markeredgecolor=INK["surface"])
    ax.set_title(f"(a) η(f, L/D) à poussée imposée ({m['thrust']:.2f} N, U∞ = {m['U_inf']:g} m/s)")
    # (b) gain de rendement en pourcentage.
    ax = axes[1]
    gain = np.ma.masked_invalid((m["gain"] - 1.0) * 100)
    im = ax.pcolormesh(F, LD, gain, cmap=CMAP_SEQ2, shading="nearest", rasterized=True)
    fig.colorbar(im, ax=ax, shrink=0.9, label="gain de rendement vs jet continu (%)")
    # dict.get : renvoie None si la clé est absente (pas de droite F* dans ce cas).
    fs_line = m.get("formation_number")
    if fs_line is not None:
        ax.axhline(fs_line, color=INK["primary"], ls=":", lw=1.0)
        ax.text(f[0] * 1.08, fs_line + 0.15, "F*", color=INK["primary"],
                bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=1.0))
    ax.set_title("(b) Gain dû à la formation des anneaux (même poussée, même débit)")
    for ax in axes:
        # Axe des fréquences en échelle logarithmique.
        ax.set_xscale("log")
        ax.set_xlabel("fréquence d'éjection f  (Hz)")
        ax.set_ylabel("rapport d'aspect L / D")
        _no_grid(ax)
        ax.text(0.98, 0.97, "zone grise : infaisable\n(rapport cyclique f L/U_j > 1)", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=INK["secondary"])
    _suptitle(fig, title or "Propulseur à anneaux tourbillonnaires : rendement propulsif")
    return fig


# =============================================================================================
# Diagnostics énergétiques et cartes de pertes
# =============================================================================================
def plot_dissipation_map(snaps, Z_list, half_width=2.2, r_max=2.2, title=None):
    """Cartes du taux de dissipation visqueuse Φ(r, z) = 2μ S:S à plusieurs instants (échelle log).

    Les contours noirs indiquent la vorticité (±20 % et ±60 % du maximum) pour situer les pertes par
    rapport aux cœurs ; chaque panneau rappelle le bilan ∫Φ dV = μ ∫ω² dV.
    """
    # Un panneau par instantané, sur une ligne (squeeze=False : axes toujours dans un tableau 2D).
    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.5), constrained_layout=True, squeeze=False)
    # Dissipation de chaque instantané et échelle log commune sur 4 décades.
    phis = [ax2d.dissipation_rate(s) for s in snaps]
    # Maximum global (expression génératrice parcourue par max) : borne haute commune à tous les panneaux.
    vmax = max(float(p.max()) for p in phis)
    # LogNorm(vmin, vmax) : couleur proportionnelle à log(Φ), de vmax·10⁻⁴ à vmax.
    norm = LogNorm(vmax * 1e-4, vmax)
    # Dernier objet tracé (support de la barre de couleur commune).
    im = None
    # zip parcourt en parallèle les axes de la ligne 0, les instantanés, leurs dissipations et positions.
    for ax, s, phi, Zc in zip(axes[0], snaps, phis, Z_list):
        # Fenêtre centrée sur l'anneau (centres des cellules) et coupe méridienne en miroir.
        kc, zw = _window(s["z_c"], Zc, half_width)
        # Masque des rayons affichés (r ≤ r_max).
        irc = s["r_c"] <= r_max
        # np.maximum(..., vmin) : plancher pour l'échelle log (pas de valeur nulle).
        rr, ff = _mirror(s["r_c"][irc], np.maximum(phi[irc][:, kc], vmax * 1e-4))
        # Carte de Φ (ombrage « gouraud » : interpolation continue des couleurs entre les points).
        im = ax.pcolormesh(zw, rr, ff, cmap=CMAP_SEQ, norm=norm, shading="gouraud", rasterized=True)
        # Contours de vorticité (aux coins) superposés.
        kf, zfw = _window(s["z_f"], Zc, half_width)
        irf = s["r_f"] <= r_max
        # odd=True : ω_θ change de signe dans le miroir (champ impair en r).
        rr2, oo = _mirror(s["r_f"][irf], s["omega"][irf][:, kf], odd=True)
        # Vorticité maximale en valeur absolue : niveaux ±20 % et ±60 % du maximum.
        om = np.abs(oo).max()
        ax.contour(zfw, rr2, oo, levels=[-0.6 * om, -0.2 * om, 0.2 * om, 0.6 * om], colors=INK["primary"],
                   linewidths=0.6)
        # Bilan intégral : ∫Φ dV et ν ∫ω² dV (ω² moyenné aux centres = 2 × densité d'enstrophie).
        I_phi = ax2d.volume_integral(s, phi)
        I_om = s["nu"] * ax2d.volume_integral(s, 2.0 * ax2d.enstrophy_density(s))
        # Encadré en bas à gauche (coordonnées relatives aux axes : transform=ax.transAxes ; « :.3e » →
        # notation scientifique à 3 décimales ; bbox = fond semi-transparent pour la lisibilité).
        ax.text(0.02, 0.03, f"∫Φ dV = {I_phi:.3e}\nμ∫ω² dV = {I_om:.3e}", transform=ax.transAxes, fontsize=8,
                color=INK["secondary"], bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=1.5, alpha=0.85))
        # Échelles égales en z et r (formes des cœurs non déformées), titre, étiquette, sans grille.
        ax.set_aspect("equal")
        ax.set_title(f"t = {s['t']:.1f}")
        ax.set_xlabel("z")
        _no_grid(ax)
    # Étiquette de l'axe vertical sur le premier panneau seulement.
    axes[0, 0].set_ylabel("x  (r en miroir)")
    # Barre de couleur commune (ax=liste des axes : l'espace est pris sur toute la ligne).
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, label="Φ = 2μ S:S  (échelle logarithmique)")
    _suptitle(fig, title or "Où l'énergie mécanique devient chaleur : taux de dissipation visqueuse Φ(r, z)")
    return fig


def plot_energy_enstrophy_budget(res, title=None):
    """Bilan temporel : E_k(t), P_z(t) et Ω(t) = ∫ω² dV (indexés sur t = 0), E_k face à la décroissance
    visqueuse théorique E_0 - ν∫Ω dt, et résidu (dissipation numérique)."""
    # Séries temporelles des diagnostics du solveur et viscosité cinématique ν.
    d = res["diagnostics"]
    t = d["t"]
    nu = res["params"]["nu"]
    # Énergie cinétique E_k/ρ et impulsion axiale P_z/ρ à chaque instant de diagnostic.
    E = d["energy"]
    P = d["impulse"]
    # Enstrophie globale Ω = ∫ω² dV = (taux de dissipation) / ν.
    Om = d["dissipation"] / nu
    # Décroissance visqueuse exacte : dE/dt = -νΩ  ⇒  E_th(t) = E_0 - ν ∫_0^t Ω dt
    # (cumulative_trapezoid(y, t, initial=0) : intégrale cumulée par les trapèzes, commençant à 0).
    E_th = E[0] - nu * cumulative_trapezoid(Om, t, initial=0.0)
    # Trois panneaux côte à côte.
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    # (a) grandeurs indexées sur leur valeur initiale (un seul axe, pas de double échelle).
    ax = axes[0]
    # Une couleur catégorielle et un style de trait par grandeur (lisible aussi en niveaux de gris).
    ax.plot(t, E / E[0], color=SERIES[0], label="énergie cinétique  E_k(t)/E_k(0)")
    ax.plot(t, P / P[0], color=SERIES[1], ls="--", label="impulsion  P_z(t)/P_z(0)")
    ax.plot(t, Om / Om[0], color=SERIES[2], ls="-.", label="enstrophie  Ω(t)/Ω(0)")
    # Ligne de référence « valeur initiale » en trait fin.
    ax.axhline(1.0, color=INK["axis"], lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("valeur / valeur initiale")
    ax.set_title("(a) Évolution indexée des invariants")
    ax.legend(loc="lower left")
    # (b) énergie simulée et décroissance visqueuse théorique.
    ax = axes[1]
    ax.plot(t, E, color=SERIES[0], label="E_k(t) simulée")
    # Libellé en notation mathématique (chaîne brute r"..." interprétée par mathtext entre les $).
    ax.plot(t, E_th, color=SERIES[1], ls="--", label=r"$E_0 - \nu\int_0^t \Omega\,dt$ (théorie visqueuse)")
    ax.set_xlabel("t")
    ax.set_ylabel("E_k / ρ")
    ax.set_title("(b) Décroissance de l'énergie")
    ax.legend(loc="upper right")
    # (c) résidu relatif du bilan (échelle log) : écart dû au schéma numérique.
    ax = axes[2]
    resid = np.abs(E - E_th) / E[0]
    # semilogy : ordonnée logarithmique ; t = 0 exclu (résidu nul par construction), +1e-16 évite log(0).
    ax.semilogy(t[1:], resid[1:] + 1e-16, color=SERIES[0])
    ax.set_xlabel("t")
    ax.set_ylabel("|E_k − E_th| / E_0")
    ax.set_title("(c) Résidu : dissipation numérique")
    # Valeur maximale du résidu affichée en bas à droite (« :.1e » : notation scientifique).
    ax.text(0.98, 0.05, f"max = {resid.max():.1e}", transform=ax.transAxes, ha="right", color=INK["secondary"])
    _suptitle(fig, title or "Bilan temporel d'énergie, d'impulsion et d'enstrophie")
    return fig


# =============================================================================================
# Diagrammes spatio-temporels et diagrammes de phase
# =============================================================================================
def _hovmoller_panel(fig, ax, z, t, data, cmap, cbar_label, tracks=None, theory_tracks=None, zlim=None):
    """Carte z–t (temps en ordonnée) avec trajectoires superposées."""
    # pcolormesh(x, y, C) : C de forme (nt, nz) ; shading="gouraud" : interpolation linéaire des couleurs
    # entre les points (pas d'effet d'escalier quand les instants enregistrés sont espacés) ;
    # rasterized=True : carte stockée en image (fichiers vectoriels légers).
    im = ax.pcolormesh(z, t, data, cmap=cmap, shading="gouraud", rasterized=True)
    # Barre de couleur propre au panneau (shrink : 90 % de la hauteur des axes).
    fig.colorbar(im, ax=ax, shrink=0.9, label=cbar_label)
    # Trajectoires mesurées : traits fins d'encre, un style de trait par anneau (plein, tirets, pointillés,
    # mixte) pour que la légende les distingue sans ajouter de couleur sur la carte ; théoriques : tirets orange.
    for k, (tt, zz, lab) in enumerate(tracks or []):
        ax.plot(zz, tt, color=INK["primary"], lw=0.9, ls=("-", "--", ":", "-.")[k % 4], alpha=0.85, label=lab)
    for tt, zz, lab in theory_tracks or []:
        # Abscisse z, ordonnée t : la pente locale dz/dt de la courbe est la vitesse de l'anneau.
        ax.plot(zz, tt, color=SERIES[1], lw=1.4, ls="--", label=lab)
    if zlim is not None:
        # set_xlim(*zlim) : le tuple (zmin, zmax) est dépaqueté en deux arguments.
        ax.set_xlim(*zlim)
    ax.set_xlabel("position axiale z")
    ax.set_ylabel("temps t")
    _no_grid(ax)
    # Légende seulement si des trajectoires sont tracées (« or » : l'une ou l'autre liste non vide).
    if tracks or theory_tracks:
        ax.legend(loc="upper left", fontsize=7.5)


def plot_hovmoller(prof, tracks=None, theory_tracks=None, title=None, margin=1.5):
    """Diagrammes de Hovmöller (z–t) : max_r ω(z, t) et dépression -min_r p(z, t).

    La pente des bandes donne la vitesse instantanée U(t) = dz/dt de chaque anneau. ``tracks`` et
    ``theory_tracks`` : listes de (t, z, libellé) superposées aux cartes.
    """
    # Instants enregistrés (ordonnée commune des deux cartes).
    t = prof["t"]
    # Fenêtre en z autour des trajectoires (± margin), sinon tout le domaine.
    zlim = None
    if tracks:
        # np.nanmin / np.nanmax ignorent les NaN (instants où un anneau n'est pas suivi).
        zmin = min(np.nanmin(zz) for _, zz, _ in tracks) - margin
        zmax = max(np.nanmax(zz) for _, zz, _ in tracks) + margin
        zlim = (zmin, zmax)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    # (a) max_r ω(z, t), aux positions axiales des coins z_f (où la vorticité est définie), palette bleue.
    _hovmoller_panel(fig, axes[0], prof["z_f"], t, prof["omega_max"], CMAP_SEQ, r"$\max_r\,\omega_\theta(z, t)$",
                     tracks, theory_tracks, zlim)
    axes[0].set_title("(a) Vorticité maximale")
    # (b) dépression -min_r (p - p∞) aux centres z_c (où la pression est définie), palette orange.
    _hovmoller_panel(fig, axes[1], prof["z_c"], t, -prof["p_min"], CMAP_SEQ2, r"$-\min_r\,(p - p_\infty)/\rho$",
                     tracks, theory_tracks, zlim)
    axes[1].set_title("(b) Dépression maximale (cœurs)")
    _suptitle(fig, title or "Diagrammes de Hovmöller z–t : la pente donne la vitesse de l'anneau")
    return fig


def plot_hovmoller_density(z, t, density, tracks=None, title=None, label=r"$\lambda(z, t)$"):
    """Diagramme de Hovmöller d'une densité axiale (ex. vorticité linéique des filaments λ(z, t))."""
    # Figure à un seul panneau.
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    # Carte z–t de la densité (palette bleue) et trajectoires éventuelles des anneaux.
    _hovmoller_panel(fig, ax, z, t, density, CMAP_SEQ, label, tracks)
    ax.set_title("Densité axiale de vorticité des filaments")
    # Titre général facultatif.
    if title:
        _suptitle(fig, title)
    return fig


def plot_phase_diagram(curves, title=None):
    """Diagramme de phase du saute-mouton dans le plan (R₁ − R₂, z₁ − z₂) et série R₁ − R₂ (t).

    ``curves`` : liste de dictionnaires {label, t, R (nt, 2), Z (nt, 2), ls}. Un cycle fermé traduit une
    dynamique périodique (fluide parfait) ; une spirale traduit l'effet de la viscosité.
    """
    # plt.subplots(1, 2) : deux panneaux côte à côte ; constrained_layout=True répartit automatiquement
    # l'espace (titres, étiquettes et légende placée « outside » sous les panneaux).
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    for k, c in enumerate(curves):
        # Écarts de rayon et de position entre l'anneau 1 (initialement arrière) et l'anneau 2
        # (c["R"] et c["Z"] sont de forme (nt, 2) : colonne 0 = anneau 1, colonne 1 = anneau 2).
        dR = c["R"][:, 0] - c["R"][:, 1]
        dZ = c["Z"][:, 0] - c["Z"][:, 1]
        # dict.get(clé, défaut) : style de trait propre à la courbe (trait plein par défaut).
        ls = c.get("ls", "-")
        # Trajectoire dans le plan des phases ; SERIES[k] : couleur catégorielle fixe de la k-ième courbe.
        axes[0].plot(dR, dZ, color=SERIES[k], ls=ls, lw=1.4 if ls == "-" else 1.2, label=c["label"])
        # Point de départ (cercle bordé de la couleur du fond pour rester lisible sur les traits).
        axes[0].plot(dR[0], dZ[0], marker="o", ms=6, color=SERIES[k], markeredgecolor=INK["surface"])
        # Sens de parcours : flèche entre deux échantillons successifs pris au 1/12 de la trajectoire
        # (max(1, ...) garantit un indice valide pour les séries très courtes).
        q = max(1, len(dR) // 12)
        # ax.annotate("", xy=arrivée, xytext=départ, arrowprops=...) : flèche sans texte.
        axes[0].annotate("", xy=(dR[q + 1], dZ[q + 1]), xytext=(dR[q], dZ[q]),
                         arrowprops=dict(arrowstyle="->", color=SERIES[k], lw=1.4))
        # Série temporelle R₁ − R₂ (t) dans le panneau (b), même couleur et même style.
        axes[1].plot(c["t"], dR, color=SERIES[k], ls=ls, lw=1.4 if ls == "-" else 1.2, label=c["label"])
    # Axes de référence (anneaux de même rayon / au même plan) en trait fin.
    axes[0].axhline(0.0, color=INK["axis"], lw=0.8)
    axes[0].axvline(0.0, color=INK["axis"], lw=0.8)
    axes[0].set_xlabel("R₁ − R₂")
    axes[0].set_ylabel("z₁ − z₂")
    # set_aspect("equal", adjustable="datalim") : même échelle sur les deux axes (orbite non déformée),
    # obtenue en élargissant les limites de données plutôt qu'en rétrécissant le panneau.
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("(a) Portrait de phase des cœurs")
    axes[1].axhline(0.0, color=INK["axis"], lw=0.8)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("R₁ − R₂")
    axes[1].set_title("(b) Oscillation des rayons (quasi-périodique)")
    # Légende commune placée sous les deux panneaux (elle ne masque ainsi aucune orbite) :
    # get_legend_handles_labels() récupère les traits étiquetés du panneau (a) ;
    # loc="outside lower center" (constrained_layout) réserve une bande sous les axes ;
    # ncols = nombre de courbes → une seule ligne d'entrées.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=max(1, len(labels)), fontsize=8.5)
    _suptitle(fig, title or "Saute-mouton : dynamique des cœurs dans l'espace des phases")
    return fig
