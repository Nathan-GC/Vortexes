"""
visualization_3d.py — Visualisations physiques et topologiques 3D avancées.

* Reconstruction volumique 3D par révolution : un champ axisymétrique f(r, z) est « balayé » en
  azimut pour obtenir le champ 3D complet f(x, y, z) = f(√(x² + y²), z) sur une grille cartésienne.
* Isosurfaces des critères Q et λ2 : pour un champ axisymétrique, l'isosurface 3D est exactement la
  surface de révolution de l'iso-contour méridien ; on l'extrait avec contourpy, on la fait tourner
  autour de l'axe (avec une coupe qui révèle l'intérieur) et on la colore par |u| ou par la pression.
* Tubes de vorticité 3D : chaque filament lagrangien devient un tube lissé (spline périodique,
  repère de transport parallèle sans torsion parasite) de rayon ∝ δ et de couleur Γ.
* Rendu volumique direct : lancer de rayons orthographique à émission-absorption (« alpha blending »
  avant → arrière) du champ d'enstrophie ½ω², avec éclairage par le gradient (effet de fumée).

Toutes les surfaces d'une vue sont réunies dans une seule Poly3DCollection pour que matplotlib les
trie ensemble par profondeur (occultations correctes entre anneaux qui s'interpénètrent).
"""

# Annotations non évaluées à l'exécution.
from __future__ import annotations

# os / shutil : dossiers de sortie et détection de ffmpeg pour les animations.
import os
import shutil

# contourpy : extraction d'iso-contours 2D (bibliothèque utilisée en interne par matplotlib).
import contourpy
# numpy : calcul vectoriel.
import numpy as np
# scipy.ndimage.map_coordinates : interpolation (bi/tri)linéaire en coordonnées fractionnaires.
from scipy import ndimage
# CubicSpline : lissage périodique des filaments ; RegularGridInterpolator : interpolation sur grille (r, z).
from scipy.interpolate import CubicSpline, RegularGridInterpolator

# Le module 2D fixe le moteur de rendu « Agg », le style et les palettes : on l'importe en premier.
from . import visualization as viz
# Diagnostics du solveur axisymétrique (critères Q, λ2, enstrophie) et reconstruction des filaments.
from . import solver_2d_axisymmetric as ax2d
from . import vortex_particles as vp

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
# ScalarMappable : objet « couleur ↔ valeur » servant à tracer une barre de couleur sans image.
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb  # noqa: E402
# Poly3DCollection : ensemble de polygones 3D (facettes) triés par profondeur au rendu.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# Noms publics du module.
__all__ = [
    "revolve_field", "isosurface_of_revolution", "tube_mesh", "volume_render", "axisym_enstrophy_volume",
    "filament_enstrophy_volume", "plot_isosurfaces", "plot_isosurface_sequence", "plot_vortex_tubes",
    "plot_volume_renderings", "animate_smoke_rings",
]

# Palette « fumée » : du gris-bleu sombre (faible densité) au blanc (forte densité).
SMOKE_CMAP = LinearSegmentedColormap.from_list("smoke", ["#1b2330", "#56657a", "#aeb9c7", "#e9eef4", "#ffffff"])
# Fond sombre des rendus volumiques (dégradé vertical : haut, bas).
DARK_BG = ("#131a26", "#030405")
# Direction de la lumière (vers la source) pour les surfaces et la fumée.
LIGHT_DIR = np.array([-0.45, -0.35, 0.82])


# =============================================================================================
# Reconstruction 3D par révolution
# =============================================================================================
def revolve_field(r, z, field, x, y, zq, periodic_z=True, fill=0.0):
    """Champ 3D f(x, y, z) = f(√(x² + y²), z) obtenu par balayage azimutal d'un champ axisymétrique.

    ``r`` (Nr,) croissant (éventuellement non uniforme), ``z`` (Nz,) uniforme, ``field`` (Nr, Nz) ;
    ``x``, ``y``, ``zq`` : axes 1D de la grille cartésienne cible. Interpolation bilinéaire dans le plan
    méridien ; points au-delà de r_max mis à ``fill``. Renvoie un tableau (nx, ny, nz) en float32.
    """
    # np.asarray(..., dtype=float) : conversion en tableaux de réels (listes acceptées, pas de copie inutile).
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    field = np.asarray(field, dtype=float)
    # np.meshgrid(..., indexing="ij") : grilles 3D des coordonnées cartésiennes (axe 0 = x, 1 = y, 2 = z).
    X, Y, Z = np.meshgrid(x, y, zq, indexing="ij")
    # Rayon cylindrique de chaque point (np.hypot = √(x² + y²)).
    R = np.hypot(X, Y)
    # Indice radial fractionnaire : np.interp inverse la relation indice → rayon (grille non uniforme).
    fi = np.interp(R, r, np.arange(r.size))
    # Indice axial fractionnaire (grille uniforme de pas Δz).
    fj = (Z - z[0]) / (z[1] - z[0])
    if periodic_z:
        # Domaine périodique : indice ramené dans [0, Nz) (np.mod) et colonne 0 recopiée en fin de
        # tableau pour interpoler entre la dernière colonne et la première.
        fj = np.mod(fj, z.size)
        field = np.concatenate([field, field[:, :1]], axis=1)
    # ndimage.map_coordinates(tableau, [indices_axe0, indices_axe1], order=1) : interpolation bilinéaire
    # (prefilter=False, inutile à l'ordre 1) ; mode="nearest" prolonge par la valeur de bord.
    vals = ndimage.map_coordinates(field, [fi.ravel(), fj.ravel()], order=1, mode="nearest", prefilter=False)
    # Remise en forme 3D, puis valeur de remplissage hors du domaine radial.
    vals = vals.reshape(X.shape)
    vals[R > r[-1]] = fill
    return vals.astype(np.float32)


def _meridian_window(r, z, field, Z_ring, half_width, r_max):
    """Sous-domaine méridien r ≤ r_max, z ∈ [Z_ring ± half_width] (z déroulé, périodicité gérée)."""
    # viz._window renvoie les indices périodiques de la fenêtre et les z « continus » correspondants.
    k, zw = viz._window(z, Z_ring, half_width)
    # Masque booléen des rayons conservés (r ≤ r_max).
    ir = r <= r_max
    # field[ir][:, k] : lignes radiales retenues, puis colonnes axiales de la fenêtre (dans l'ordre déroulé).
    return r[ir], zw, field[ir][:, k]


# =============================================================================================
# Isosurfaces de révolution
# =============================================================================================
def isosurface_of_revolution(r, z, field, level, color_field=None, n_theta=96, theta0=0.0, theta_span=2.0 * np.pi):
    """Isosurface ``field = level`` d'un champ axisymétrique, comme surfaces de révolution.

    Les iso-contours méridiens sont extraits par contourpy puis tournés de θ0 à θ0 + θ_span autour de
    l'axe z. Renvoie une liste de dictionnaires {X, Y, Z (n, n_theta), C (valeurs de couleur), rs, zs}.
    """
    # contour_generator(x, y, z) attend z de forme (ny, nx) : lignes = r (axe 0), colonnes = z (axe 1).
    gen = contourpy.contour_generator(x=z, y=r, z=field)
    # Interpolateur de la grandeur de coloration aux points des contours : RegularGridInterpolator((axes), valeurs)
    # interpole linéairement sur la grille (r, z) ; bounds_error=False et fill_value=None → extrapolation au bord.
    interp = None
    if color_field is not None:
        interp = RegularGridInterpolator((r, z), color_field, bounds_error=False, fill_value=None)
    # Angles de révolution : n_theta valeurs régulières de θ0 à θ0 + θ_span (bornes comprises).
    theta = np.linspace(theta0, theta0 + theta_span, n_theta)
    # Liste des surfaces (une par ligne de niveau méridienne).
    surfaces = []
    # gen.lines(level) : liste de tableaux (n, 2) de colonnes (z, r) décrivant chaque ligne de niveau.
    for line in gen.lines(level):
        # Lignes trop courtes (quelques points isolés) : surface dégénérée, ignorée.
        if len(line) < 4:
            continue
        # Colonnes du contour : z (abscisse du générateur) et r (ordonnée).
        zs, rs = line[:, 0], line[:, 1]
        # Valeur de coloration le long du contour (constante en θ par axisymétrie).
        cvals = interp(np.column_stack([rs, zs])) if interp is not None else np.zeros(len(rs))
        # Coordonnées 3D : (r cos θ, r sin θ, z) par broadcasting (n points × n_theta angles).
        X = rs[:, None] * np.cos(theta)[None, :]
        Y = rs[:, None] * np.sin(theta)[None, :]
        # np.repeat : recopie la colonne des z / des couleurs pour chaque angle.
        Z = np.repeat(zs[:, None], n_theta, axis=1)
        C = np.repeat(cvals[:, None], n_theta, axis=1)
        # Surface structurée (n points du contour × n_theta angles) et contour méridien d'origine.
        surfaces.append({"X": X, "Y": Y, "Z": Z, "C": C, "rs": rs, "zs": zs})
    return surfaces


def _grid_quads(X, Y, Z, C=None):
    """Quadrilatères (n_quads, 4, 3) d'une surface structurée et valeur moyenne de C par facette."""
    # np.stack(..., axis=-1) : points 3D (n, m, 3).
    P = np.stack([X, Y, Z], axis=-1)
    # Les 4 sommets de chaque maille (ordre cyclique) puis mise à plat en liste de facettes.
    verts = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2).reshape(-1, 4, 3)
    # Pas de grandeur de coloration : seules les facettes sont renvoyées.
    if C is None:
        return verts, None
    # Couleur de la facette = moyenne des 4 sommets.
    c = 0.25 * (C[:-1, :-1] + C[1:, :-1] + C[1:, 1:] + C[:-1, 1:])
    # c.ravel() : mise à plat dans le même ordre que les facettes (ligne par ligne).
    return verts, c.ravel()


def _view_vector(elev, azim):
    """Vecteur unitaire de la scène vers la caméra pour ax.view_init(elev, azim) (degrés)."""
    # np.radians convertit des degrés en radians.
    e, a = np.radians(elev), np.radians(azim)
    # Coordonnées sphériques → cartésiennes : élévation e au-dessus du plan (x, y), azimut a autour de z.
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _shade(rgb, verts, view, light=LIGHT_DIR, ambient=0.30, diffuse=0.70, specular=0.22, shininess=28.0):
    """Éclairage de Blinn–Phong (bilatéral) de facettes : couleur × (ambiante + diffuse) + spéculaire."""
    # Normale de chaque facette : produit vectoriel des deux diagonales, normalisé.
    n = np.cross(verts[:, 2] - verts[:, 0], verts[:, 3] - verts[:, 1])
    # Normalisation (keepdims=True : norme de forme (n, 1) pour diviser chaque ligne ; +1e-30 évite 0/0).
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    # Direction unitaire de la lumière.
    L = light / np.linalg.norm(light)
    # Terme de Lambert |n·L| (valeur absolue : les deux faces sont éclairées ; « @ » = produit matriciel).
    ndl = np.abs(n @ L)
    # Vecteur « mi-chemin » entre lumière et caméra pour la tache spéculaire.
    H = (L + view) / np.linalg.norm(L + view)
    # Alignement normale / vecteur mi-chemin : maximal là où la lumière se réfléchit vers la caméra.
    ndh = np.abs(n @ H)
    # Couleur × (ambiante + diffuse) + reflet blanc (ndh^shininess : tache d'autant plus petite que
    # shininess est grand) ; [:, None] transforme les coefficients (n,) en colonnes (n, 1).
    out = rgb[:, :3] * (ambient + diffuse * ndl)[:, None] + specular * (ndh**shininess)[:, None]
    # np.clip borne les couleurs dans [0, 1].
    return np.clip(out, 0.0, 1.0)


def _add_polys(ax, verts, rgb):
    """Ajoute des facettes colorées à un axe 3D (bords de même couleur pour masquer les jointures)."""
    # Poly3DCollection(facettes, facecolors, edgecolors, linewidths) : bords fins de la couleur de la facette.
    coll = Poly3DCollection(verts, facecolors=rgb, edgecolors=rgb, linewidths=0.25)
    # add_collection3d : insertion dans la scène 3D (triée par profondeur au dessin).
    ax.add_collection3d(coll)
    return coll


def _style_axes3d(ax, lo, hi, elev, azim, zoom=1.0):
    """Limites, proportions réelles, angle de vue et habillage discret d'un axe 3D."""
    # Limites de la boîte sur chacun des trois axes (coins opposés lo et hi).
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # set_box_aspect : proportions de la boîte = étendues réelles (pas de déformation) ;
    # zoom > 1 agrandit la scène dans son cadre (moins de marge blanche autour de la boîte).
    ax.set_box_aspect(tuple(np.asarray(hi) - np.asarray(lo)), zoom=zoom)
    # view_init(elev, azim) : position de la caméra (élévation et azimut en degrés).
    ax.view_init(elev=elev, azim=azim)
    # Noms des trois axes.
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        # Panneaux de fond de la boîte : couleur de surface claire et bord gris discret.
        axis.pane.set_facecolor(viz.INK["surface"])
        axis.pane.set_edgecolor(viz.INK["grid"])
        # _axinfo (attribut interne de mplot3d) : couleur et épaisseur des lignes de grille 3D.
        axis._axinfo["grid"]["color"] = viz.INK["grid"]
        axis._axinfo["grid"]["linewidth"] = 0.5


def _criterion_and_color(snap, criterion, color_by):
    """Champ de critère (positif dans les cœurs) et champ de coloration aux centres des cellules."""
    if criterion == "Q":
        # Critère Q = ½(|Ω|² − |S|²) aux centres des cellules : positif là où la rotation domine.
        crit, crit_label = ax2d.q_criterion(snap), "Q"
    else:
        # -λ2 > 0 dans les cœurs tourbillonnaires (λ2 < 0) : même convention « positif = tourbillon ».
        crit, crit_label = -ax2d.lambda2_criterion(snap), "−λ₂"
    if color_by == "speed":
        # Norme de la vitesse aux centres.
        col, col_label = np.hypot(snap["u_r"], snap["u_z"]), "|u|"
    else:
        # Dépression par rapport à la pression moyenne sur la paroi extérieure (champ lointain) :
        # grandeur positive dans les cœurs, représentée en orange séquentiel comme sur le Hovmöller.
        col, col_label = -(snap["p"] - snap["p"][-1].mean()), "dépression −(p − p∞)/ρ"
    return crit, crit_label, col, col_label


def _isosurface_data(snap, Z_ring, criterion, color_by, level_fraction, half_width, r_max, cutaway, azim, n_theta):
    """Prépare une isosurface de révolution avec coupe : surfaces, faces de coupe, niveau, libellés."""
    # Champ du critère (positif dans les cœurs) et grandeur de coloration, avec leurs libellés.
    crit, crit_label, col, col_label = _criterion_and_color(snap, criterion, color_by)
    # Fenêtre méridienne autour de l'anneau (centres des cellules).
    rw, zw, cw = _meridian_window(snap["r_c"], snap["z_c"], crit, Z_ring, half_width, r_max)
    # Même fenêtre pour la grandeur de coloration (« _ » : valeurs renvoyées mais inutilisées).
    _, _, colw = _meridian_window(snap["r_c"], snap["z_c"], col, Z_ring, half_width, r_max)
    # Niveau de l'isosurface : fraction du maximum du critère dans la fenêtre.
    level = level_fraction * cw.max()
    # Coupe : secteur de « cutaway » degrés retiré face à la caméra (centré sur l'azimut de vue).
    cut = np.radians(cutaway)
    # La surface commence juste après le secteur retiré (azimut de vue + demi-coupe) et en couvre le reste.
    theta0 = np.radians(azim) + 0.5 * cut
    span = 2.0 * np.pi - cut
    # Isosurface de révolution colorée par la grandeur choisie.
    surfaces = isosurface_of_revolution(rw, zw, cw, level, colw, n_theta=n_theta, theta0=theta0, theta_span=span)
    # Faces de coupe : plans méridiens θ = θ0 et θ = θ0 + span, limités aux mailles où le critère dépasse le niveau.
    Rg, Zg = np.meshgrid(rw, zw, indexing="ij")
    # Critère moyen de chaque maille (les arguments X, Y de _grid_quads sont sans importance ici).
    _, cm = _grid_quads(Rg, Rg, Zg, cw)
    # Mailles « intérieures » à l'isosurface (critère moyen au-dessus du niveau).
    keep = cm > level
    cuts = []
    for th in (theta0, theta0 + span):
        # Plan méridien d'angle th : (r cos th, r sin th, z) ; couleur moyenne de chaque maille.
        verts, c = _grid_quads(Rg * np.cos(th), Rg * np.sin(th), Zg, colw)
        # keep.any() : au moins une maille à dessiner ; indexation booléenne verts[keep] = mailles retenues.
        if keep.any():
            cuts.append((verts[keep], c[keep]))
    # Valeurs de coloration présentes sur la surface (pour une normalisation à fort contraste).
    cvals = np.concatenate([s["C"].ravel() for s in surfaces] + [c for _, c in cuts]) if surfaces else colw.ravel()
    # Dictionnaire regroupant tout ce qu'il faut pour dessiner la vue.
    return {"surfaces": surfaces, "cuts": cuts, "level": level, "crit_label": crit_label, "col_label": col_label,
            "cvals": cvals}


def _norm_for(cvals):
    """Normalisation couleur sur la plage utile des valeurs (2ᵉ–98ᵉ centiles, np.percentile)."""
    lo, hi = np.percentile(cvals, [2.0, 98.0])
    # Normalize(vmin, vmax) : envoie [vmin, vmax] sur [0, 1] (max(...) évite vmin = vmax si le champ est uniforme).
    return Normalize(float(lo), float(max(hi, lo + 1e-9)))


def _render_isosurface(ax, data, norm, cmap, Z_ring, half_width, r_max, elev, azim):
    """Dessine les surfaces et les faces de coupe préparées par :func:`_isosurface_data`."""
    # Direction de la caméra (pour le reflet spéculaire).
    view = _view_vector(elev, azim)
    # Facettes et couleurs accumulées de toutes les surfaces de la vue.
    all_verts, all_rgb = [], []
    for s in data["surfaces"]:
        # Facettes quadrilatères de la surface structurée et couleur moyenne de chacune.
        verts, c = _grid_quads(s["X"], s["Y"], s["Z"], s["C"])
        # cmap(norm(c)) : couleurs RGBA des facettes ; puis éclairage de Blinn–Phong.
        all_verts.append(verts)
        all_rgb.append(_shade(cmap(norm(c)), verts, view))
    for verts, c in data["cuts"]:
        # Faces de coupe planes : éclairage plus uniforme, sans reflet.
        all_verts.append(verts)
        all_rgb.append(_shade(cmap(norm(c)), verts, view, ambient=0.55, diffuse=0.45, specular=0.0))
    if all_verts:
        # Une seule collection : tri de profondeur commun à toutes les facettes (np.concatenate).
        _add_polys(ax, np.concatenate(all_verts), np.concatenate(all_rgb))
    # Boîte de la vue : [-r_max, r_max]² en (x, y) et fenêtre axiale centrée sur l'anneau.
    lo = np.array([-r_max, -r_max, Z_ring - half_width])
    hi = np.array([r_max, r_max, Z_ring + half_width])
    _style_axes3d(ax, lo, hi, elev, azim)


def plot_isosurfaces(snap, Z_ring, level_fraction=0.04, half_width=1.3, r_max=1.6, cutaway=90.0, elev=24.0,
                     azim=-60.0, n_theta=110, title=None):
    """Deux vues 3D d'un instantané axisymétrique reconstruit par révolution :
    (a) isosurface Q > 0 colorée par |u| ; (b) isosurface λ2 < 0 colorée par la dépression."""
    # Figure de 13 × 5,8 pouces accueillant deux vues 3D côte à côte.
    fig = plt.figure(figsize=(13.0, 5.8))
    # (critère, grandeur de coloration, titre) de chacune des deux vues.
    specs = (("Q", "speed", "(a) Critère Q, couleur |u|"), ("lambda2", "pressure", "(b) Critère λ₂, couleur = dépression"))
    for k, (crit, color_by, label) in enumerate(specs):
        # Axes 3D n° k + 1 d'une grille 1 × 2.
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        # Surfaces, faces de coupe et valeurs de coloration de la vue.
        data = _isosurface_data(snap, Z_ring, crit, color_by, level_fraction, half_width, r_max, cutaway, azim, n_theta)
        # Normalisation à fort contraste et palette : bleue pour |u|, orange pour la dépression.
        norm = _norm_for(data["cvals"])
        cmap = viz.CMAP_SEQ if color_by == "speed" else viz.CMAP_SEQ2
        _render_isosurface(ax, data, norm, cmap, Z_ring, half_width, r_max, elev, azim)
        # Titre sur deux lignes (« \n ») : grandeur représentée, niveau (« :.0% » → pourcentage) et coupe.
        ax.set_title(f"{label}\nisosurface {data['crit_label']} = {level_fraction:.0%} du maximum, coupe {cutaway:.0f}°",
                     loc="left")
        # Barre de couleur à partir d'un ScalarMappable (norm + cmap).
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label=data["col_label"])
    viz._suptitle(fig, title or f"Reconstruction 3D par révolution — peau toroïdale du vortex (t = {snap['t']:.1f})")
    # subplots_adjust : marges de la figure (fractions de sa taille) et espacement horizontal entre vues.
    fig.subplots_adjust(left=0.0, right=0.97, top=0.86, bottom=0.03, wspace=0.08)
    return fig


def plot_isosurface_sequence(snaps, Z_list, criterion="Q", color_by="speed", level_fraction=0.04, half_width=1.7,
                             r_max=1.7, cutaway=90.0, elev=20.0, azim=-60.0, n_theta=100, ncols=2, title=None):
    """Séquence d'isosurfaces de révolution (même échelle de couleur) — ex. saute-mouton visqueux."""
    # Nombre de vues et de lignes de la grille (np.ceil : arrondi supérieur).
    n = len(snaps)
    nrows = int(np.ceil(n / ncols))
    # Taille proportionnelle au nombre de colonnes et de lignes.
    fig = plt.figure(figsize=(6.2 * ncols, 5.0 * nrows))
    # Préparation de toutes les surfaces, puis échelle de couleur commune à tous les instantanés.
    datas = [_isosurface_data(s, Zr, criterion, color_by, level_fraction, half_width, r_max, cutaway, azim, n_theta)
             for s, Zr in zip(snaps, Z_list)]
    norm = _norm_for(np.concatenate([dd["cvals"] for dd in datas]))
    cmap = viz.CMAP_SEQ if color_by == "speed" else viz.CMAP_SEQ2
    for k, (s, Zr, dd) in enumerate(zip(snaps, Z_list, datas)):
        # Une vue 3D par instantané, centrée sur la position axiale Zr correspondante.
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        _render_isosurface(ax, dd, norm, cmap, Zr, half_width, r_max, elev, azim)
        ax.set_title(f"t = {s['t']:.1f}", loc="center")
    # Barre de couleur unique pour toutes les vues (ax=fig.axes : place prise sur l'ensemble des axes).
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02, label=datas[0]["col_label"])
    # Nom typographique du critère pour le titre.
    crit_name = "Q" if criterion == "Q" else "λ₂"
    viz._suptitle(fig, title or f"Isosurfaces du critère {crit_name} ({level_fraction:.0%} du maximum), reconstruction par révolution")
    return fig


# =============================================================================================
# Tubes de vorticité
# =============================================================================================
def _rotate_about(v, k, ang):
    """Rotation de Rodrigues de vecteurs v (M, 3) autour d'axes unitaires k (M, 3) d'angles ang (M,)."""
    # Cosinus et sinus des angles, en colonnes (M, 1) pour multiplier chaque vecteur par son propre angle.
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    # Formule de Rodrigues : v cos θ + (k × v) sin θ + k (k·v)(1 - cos θ).
    # Produit scalaire k·v ligne par ligne (np.sum(..., axis=1) ; keepdims=True → forme (M, 1)).
    kv = np.sum(k * v, axis=1, keepdims=True)
    # np.cross(k, v) : produit vectoriel ligne par ligne.
    return v * c + np.cross(k, v) * s + k * kv * (1.0 - c)


def _transport(n, t0, t1):
    """Transport parallèle d'une normale n de la tangente t0 à la tangente t1 (rotation minimale)."""
    # Axe et angle de la rotation qui amène t0 sur t1 (np.arctan2(sin, cos) : angle signé robuste).
    v = np.cross(t0, t1)
    # |t0 × t1| = sin de l'angle entre les deux tangentes.
    s = np.linalg.norm(v)
    if s < 1e-12:
        # Tangentes (quasi) parallèles : aucune rotation nécessaire.
        out = n
    else:
        # Rotation de n autour de l'axe v/|v| de l'angle (t0, t1) ; n[None, :] : vecteur mis en tableau (1, 3)
        # pour _rotate_about, [0] : extraction du résultat unique.
        out = _rotate_about(n[None, :], (v / s)[None, :], np.array([np.arctan2(s, np.dot(t0, t1))]))[0]
    # Reprojection orthogonale à t1 et normalisation (suppression des erreurs d'arrondi).
    out = out - np.dot(out, t1) * t1
    return out / np.linalg.norm(out)


def _parallel_transport_frames(T):
    """Repères (N, B) sans torsion le long d'une courbe fermée de tangentes unitaires T (M, 3).

    La normale est transportée de proche en proche ; l'écart angulaire résiduel à la fermeture
    (holonomie) est réparti linéairement le long de la courbe pour que le tube se referme sans raccord.
    """
    # Nombre de points de la courbe.
    M = len(T)
    # Normale initiale : un vecteur de référence non colinéaire à T[0], orthogonalisé.
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    # np.empty_like : tableau non initialisé de même forme que T (rempli ci-dessous).
    N = np.empty_like(T)
    # Procédé de Gram–Schmidt : on retire à ref sa composante selon T[0], puis on normalise.
    n0 = ref - np.dot(ref, T[0]) * T[0]
    N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, M):
        # Normale suivante = normale précédente transportée de la tangente i-1 à la tangente i.
        N[i] = _transport(N[i - 1], T[i - 1], T[i])
    # Normale obtenue après un tour complet, comparée à la normale de départ (angle signé autour de T[0]).
    n_end = _transport(N[-1], T[-1], T[0])
    # Angle signé de N[0] à n_end autour de T[0] : atan2((N0 × n_end)·T0, N0·n_end).
    ang = np.arctan2(np.dot(np.cross(N[0], n_end), T[0]), np.dot(N[0], n_end))
    # Correction progressive -ang·i/M : le repère n° M coïncide alors avec le repère n° 0.
    N = _rotate_about(N, T, -ang * np.arange(M) / M)
    # Binormale B = T × N (repère orthonormé direct).
    return N, np.cross(T, N)


def tube_mesh(points, radius, n_circle=16, n_samples=None):
    """Maillage (M+1, n_circle+1, 3) d'un tube lisse de rayon ``radius`` autour d'une courbe fermée."""
    # Courbe fermée : premier point répété à la fin ; abscisse curviligne cumulée.
    P = np.vstack([points, points[:1]])
    # Longueurs des segments (np.diff : différences entre points successifs) et abscisse cumulée depuis 0.
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # Spline cubique périodique x(s) : ligne centrale lissée et sa dérivée (tangente) ;
    # bc_type="periodic" impose la continuité des dérivées au raccord (P[-1] = P[0]).
    spline = CubicSpline(s, P, bc_type="periodic", axis=0)
    # Nombre de sections du tube : imposé, sinon deux par nœud (au moins 48).
    M = n_samples or max(2 * len(points), 48)
    # Abscisses régulières sur un tour (endpoint=False : le point s = longueur totale est le point 0).
    ss = np.linspace(0.0, s[-1], M, endpoint=False)
    # Points de la ligne centrale lissée et tangentes (spline(ss, 1) = dérivée première), normalisées.
    C = spline(ss)
    T = spline(ss, 1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # Repères orthonormés (T, N, B) sans torsion parasite le long de la courbe.
    N, B = _parallel_transport_frames(T)
    # Fermeture du maillage le long de la courbe (le dernier anneau de points = le premier).
    C = np.vstack([C, C[:1]])
    N = np.vstack([N, N[:1]])
    B = np.vstack([B, B[:1]])
    # Cercle de rayon « radius » dans le plan (N, B) de chaque point de la ligne centrale.
    # Angles autour de la ligne centrale (n_circle + 1 valeurs : le dernier point referme le cercle).
    phi = np.linspace(0.0, 2.0 * np.pi, n_circle + 1)
    # Vecteurs unitaires cos φ N + sin φ B par broadcasting → tableau (M+1, n_circle+1, 3).
    ring = np.cos(phi)[None, :, None] * N[:, None, :] + np.sin(phi)[None, :, None] * B[:, None, :]
    # Points du tube = centre de la section + rayon × direction radiale.
    return C[:, None, :] + radius * ring


def _cut_tube(P, center, theta_c, cut):
    """Écorché d'un tube fermé : retire la portion située dans le secteur azimutal θ_c ± cut/2 (autour de
    l'axe vertical passant par ``center``) et ferme les deux extrémités par des disques.

    ``P`` : maillage fermé (M+1, n_circle+1, 3) de :func:`tube_mesh`. Renvoie (maillage ouvert, facettes
    des disques (2·n_circle, 4, 3)) ; (P, None) si le tube évite le secteur, (None, None) s'il y est entier.
    """
    # Centre de chaque section du tube : moyenne des points du cercle (le dernier, répété, est exclu).
    C = P[:, :-1].mean(axis=1)
    # Azimut de chaque section autour de l'axe (np.arctan2(y, x) ∈ ]-π, π]).
    th = np.arctan2(C[:, 1] - center[1], C[:, 0] - center[0])
    # Écart angulaire signé au centre du secteur, ramené dans [-π, π) (np.mod gère le passage à ±π).
    dth = np.mod(th - theta_c + np.pi, 2.0 * np.pi) - np.pi
    # Sections conservées (hors du secteur) ; la dernière rangée répète la première, on l'écarte.
    k = (np.abs(dth) > 0.5 * cut)[:-1]
    # Tube entièrement hors du secteur : rendu intact ; entièrement dedans : rien à dessiner.
    if k.all():
        return P, None
    if not k.any():
        return None, None
    # Nombre de sections distinctes du tube fermé.
    M = len(k)
    # Débuts des plages conservées : section gardée précédée d'une section retirée (np.roll décale d'un cran,
    # de façon circulaire) ; on retient la plus longue plage (un anneau déformé peut traverser le secteur deux fois).
    best_s, best_n = 0, 0
    for s in np.nonzero(k & ~np.roll(k, 1))[0]:
        # Longueur de la plage commençant en s (indices pris modulo M : la plage peut franchir la fin).
        n = 0
        while n < M and k[(s + n) % M]:
            n += 1
        # Mémorisation de la plage la plus longue rencontrée.
        if n > best_n:
            best_s, best_n = s, n
    # Indices (circulaires) de la plage retenue, puis tube ouvert correspondant.
    Q = P[:-1][(best_s + np.arange(best_n)) % M]
    # Facettes des deux disques de fermeture (première et dernière section du tube ouvert).
    caps = []
    for end in (Q[0], Q[-1]):
        # Disque de fermeture : éventail de triangles (centre, point j, point j + 1), écrits comme des
        # quadrilatères dégénérés (centre répété) pour garder des facettes à 4 sommets.
        # Centre du disque (moyenne des points du cercle) recopié pour chaque triangle (np.broadcast_to : vue sans copie).
        c = np.broadcast_to(end[:-1].mean(axis=0), (len(end) - 1, 3))
        # np.stack(..., axis=1) : facettes (n_circle, 4, 3) de sommets (centre, point j, point j + 1, centre).
        caps.append(np.stack([c, end[:-1], end[1:], c], axis=1))
    # Tube ouvert et facettes des deux disques réunies (np.concatenate).
    return Q, np.concatenate(caps)


def plot_vortex_tubes(frames, gammas, deltas_list, times, radius_scale=1.0, n_circle=14, elev=18.0, azim=-60.0,
                      follow=False, window=3.2, title=None, ncols=None, cutaway=None):
    """Tubes de vorticité 3D : un tube lissé par filament, rayon = ``radius_scale`` × δ, couleur = Γ.

    ``frames`` : liste d'instantanés (listes de tableaux de nœuds) ; ``deltas_list`` : δ de chaque
    filament pour chaque instantané ; ``times`` : instants affichés. ``cutaway`` (degrés) : secteur
    retiré face à la caméra, dont les sections révèlent l'organisation interne d'un cœur multi-filaments.
    """
    # Nombre de vues, de colonnes (toutes sur une ligne par défaut) et de lignes (arrondi supérieur).
    n = len(frames)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    # plt.figure(figsize=(largeur, hauteur)) en pouces : 4,6 pouces par vue.
    fig = plt.figure(figsize=(4.6 * ncols, 4.6 * nrows))
    cmap, norm = _gamma_colormap(gammas)
    # Vecteur unitaire dirigé vers l'observateur (sert à l'ombrage spéculaire).
    view = _view_vector(elev, azim)
    # Boîte commune à tous les instantanés (si la vue ne suit pas les anneaux) ; pad = marge relative.
    lo_all, hi_all = viz._frame_bounds(frames, pad=0.25)
    # enumerate(zip(...)) : indice de la vue et, en parallèle, nœuds, δ des filaments et instant.
    for k, (frame, deltas, t) in enumerate(zip(frames, deltas_list, times)):
        # fig.add_subplot(nrows, ncols, index, projection="3d") : axes 3D n° k + 1 (numérotation dès 1).
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        # Facettes et couleurs de tous les tubes de la vue (réunies ensuite en une seule collection).
        all_verts, all_rgb = [], []
        # Barycentre de tous les nœuds : l'axe vertical qui y passe sert de référence à l'écorché.
        center = np.concatenate(frame, axis=0).mean(axis=0)
        for nodes, g, d in zip(frame, gammas, deltas):
            # Maillage du tube lissé autour du filament (rayon proportionnel à son δ courant).
            P = tube_mesh(nodes, radius_scale * d, n_circle=n_circle)
            # Couleur RGBA du filament (Γ normalisé puis converti par la palette).
            rgba1 = np.array([cmap(norm(g))])
            if cutaway:
                # Écorché : secteur centré sur l'azimut de la caméra retiré, extrémités fermées par des disques.
                P, caps = _cut_tube(P, center, np.radians(azim), np.radians(cutaway))
                if caps is not None:
                    # Disques de coupe : éclairage plus uniforme et sans reflet (comme les faces de coupe
                    # des isosurfaces) pour faire ressortir la section du cœur.
                    all_verts.append(caps)
                    all_rgb.append(_shade(np.repeat(rgba1, len(caps), axis=0), caps, view, ambient=0.55,
                                          diffuse=0.45, specular=0.0))
                if P is None:
                    continue
            # Découpage de la grille (rangées × points du cercle) en quadrilatères (4 sommets 3D chacun).
            verts, _ = _grid_quads(P[..., 0], P[..., 1], P[..., 2])
            # Couleur uniforme du tube recopiée pour chacune de ses facettes
            # (np.repeat(tableau (1, 4), n, axis=0) → tableau (n, 4) de couleurs RGBA).
            rgba = np.repeat(rgba1, len(verts), axis=0)
            all_verts.append(verts)
            # Ombrage de Blinn–Phong de chaque facette selon son orientation.
            all_rgb.append(_shade(rgba, verts, view))
        # np.concatenate : une seule Poly3DCollection → tri en profondeur commun à tous les tubes.
        _add_polys(ax, np.concatenate(all_verts), np.concatenate(all_rgb))
        if follow:
            # Vue qui suit les anneaux : cube de côté « window » centré sur le barycentre des nœuds.
            c = np.concatenate(frame, axis=0).mean(axis=0)
            lo, hi = c - 0.5 * window, c + 0.5 * window
        else:
            lo, hi = lo_all, hi_all
        # Limites, proportions égales, angle de vue et habillage sobre des axes 3D.
        _style_axes3d(ax, lo, hi, elev, azim)
        # Titre : instant et rayon de régularisation (valeur unique ou intervalle si les δ diffèrent).
        d_lo, d_hi = min(deltas), max(deltas)
        d_txt = f"δ = {d_lo:.3f}" if d_hi - d_lo < 5e-4 else f"δ ∈ [{d_lo:.3f}, {d_hi:.3f}]"
        ax.set_title(f"t = {t:.1f}   {d_txt}", loc="center")
    # ScalarMappable(norm, cmap) : objet « valeur → couleur » sans image, support de la barre de couleur
    # commune (ax=fig.axes : l'espace est pris sur l'ensemble des vues).
    sm = ScalarMappable(norm=norm, cmap=cmap)
    # pad = 0,06 : espace entre la dernière vue et la barre (le libellé « z » de l'axe 3D reste visible).
    fig.colorbar(sm, ax=fig.axes, shrink=0.55, pad=0.06, label="circulation Γ du filament")
    viz._suptitle(fig, title or "Tubes de vorticité (rayon ∝ δ, couleur = Γ)")
    return fig


def _gamma_colormap(gammas):
    """Palette et normalisation des circulations Γ des filaments.

    * Circulations des deux signes : palette divergente centrée sur Γ = 0 (bleu : Γ < 0, rouge : Γ > 0).
    * Un seul signe : demi-palette de même teinte (rouge si Γ > 0, bleu si Γ < 0), étirée entre |Γ|min et
      |Γ|max, pour distinguer les filaments centraux (intenses, foncés) des filaments périphériques.
    """
    g = np.asarray(gammas, dtype=float)
    # |Γ| maximal (borne commune des deux bras de la palette divergente).
    gmax = float(np.abs(g).max())
    if g.min() < 0.0 < g.max():
        # TwoSlopeNorm(centre, vmin, vmax) : 0 → gris neutre au milieu de la palette.
        return viz.CMAP_DIV, TwoSlopeNorm(0.0, -gmax, gmax)
    # |Γ| minimal ; toutes les circulations égales → plage [0, |Γ|] (couleur franche, pas de division par 0).
    gmin = float(np.abs(g).min())
    if gmax - gmin < 1e-9 * max(gmax, 1e-300):
        gmin = 0.0
    if g.max() > 0.0:
        # Bras rouge de la palette divergente : de 0,6 (rose clair) à 1,0 (rouge foncé) ;
        # LinearSegmentedColormap.from_list(nom, couleurs) interpole entre les 256 couleurs prélevées.
        cmap = LinearSegmentedColormap.from_list("gamma_pos", viz.CMAP_DIV(np.linspace(0.6, 1.0, 256)))
        return cmap, Normalize(gmin, gmax)
    # Γ < 0 : Normalize(-|Γ|max, -|Γ|min) envoie le filament le plus intense sur 0 → bleu foncé (0,0)
    # et le plus faible sur 1 → bleu clair (0,4 dans la palette divergente).
    cmap = LinearSegmentedColormap.from_list("gamma_neg", viz.CMAP_DIV(np.linspace(0.0, 0.4, 256)))
    return cmap, Normalize(-gmax, -gmin)


# =============================================================================================
# Rendu volumique direct (émission-absorption)
# =============================================================================================
def volume_render(volume, origin, spacing, elev=20.0, azim=-60.0, image_size=(420, 330), n_steps=240, density=8.0,
                  gamma=0.7, cmap=SMOKE_CMAP, light=LIGHT_DIR, ambient=0.35, diffuse=0.75, background=DARK_BG, zoom=1.2):
    """Rendu volumique d'un champ scalaire 3D positif par lancer de rayons orthographique.

    Valeur normalisée e = (V/V_max)^γ ; le long de chaque rayon (avant → arrière) :
        α_k = 1 - exp(-density · e_k · Δs)           (opacité d'un pas, loi de Beer–Lambert)
        C  += T · α_k · couleur(e_k) · éclairage_k     (alpha blending)
        T  *= (1 - α_k)                                (transmittance restante)
    L'éclairage utilise la normale -∇e/|∇e| (Lambert). Image finale = C + T × fond.
    Renvoie un tableau RGB (ny, nx, 3) dans [0, 1].
    """
    # Volume en float32 (deux fois moins de mémoire que float64, précision suffisante pour l'image).
    vol = np.asarray(volume, dtype=np.float32)
    # Valeur maximale (référence de normalisation).
    vmax = float(vol.max())
    # Dimensions de l'image en pixels (largeur, hauteur).
    nx, ny = image_size
    # Fond : dégradé vertical entre les deux couleurs (to_rgb convertit « #rrggbb » en triplet).
    # Poids de mélange de 0 (haut) à 1 (bas), de forme (ny, 1, 1) pour se diffuser sur les colonnes et le RGB.
    w = np.linspace(0.0, 1.0, ny)[:, None, None]
    bg = (1.0 - w) * np.array(to_rgb(background[0])) + w * np.array(to_rgb(background[1]))
    # Volume vide : l'image est le fond seul (np.broadcast_to étend la colonne à toute la largeur ; .copy()
    # rend un tableau modifiable indépendant).
    if vmax <= 0.0:
        return np.broadcast_to(bg, (ny, nx, 3)).copy()
    # Normalisation et courbe de contraste γ.
    vol = (vol / vmax) ** gamma
    # Origine (coin du voxel 0) et pas de la grille, en tableaux de 3 réels.
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    # Boîte englobant les voxels non négligeables : restreint la longueur des rayons.
    # np.argwhere : indices (i, j, k) des voxels significatifs ; min/max par axe → coins de la boîte.
    idx = np.argwhere(vol > 1e-3)
    box_lo = origin + idx.min(axis=0) * spacing
    box_hi = origin + idx.max(axis=0) * spacing
    # Centre de la scène et demi-diagonale (+ 2 voxels de marge) : rayon de la sphère englobante.
    center = 0.5 * (box_lo + box_hi)
    half_diag = 0.5 * np.linalg.norm(box_hi - box_lo) + 2.0 * spacing.max()
    # Repère caméra : d = direction de visée, (e_u, e_v) = axes horizontal et vertical de l'image.
    d = -_view_vector(elev, azim)
    a = np.radians(azim)
    # Axe horizontal de l'image : horizontal dans la scène et perpendiculaire à l'azimut de visée.
    e_u = np.array([-np.sin(a), np.cos(a), 0.0])
    # Axe vertical de l'image : e_v = e_u × d (orthogonal aux deux autres, dirigé vers le haut).
    e_v = np.cross(e_u, d)
    # Fenêtre image (unités physiques), rapport d'aspect de l'image conservé ; zoom > 1 resserre le cadrage.
    half_w = half_diag / zoom
    half_h = half_w * ny / nx
    # Coordonnées des pixels dans le plan image (u vers la droite, v vers le haut).
    u = np.linspace(-half_w, half_w, nx)
    v = np.linspace(half_h, -half_h, ny)  # ligne 0 = haut de l'image
    U, V = np.meshgrid(u, v)
    # Origine de chaque rayon dans le plan image passant par le centre (projection orthographique :
    # tous les rayons sont parallèles à d) ; reshape(-1, 3) → une ligne par pixel.
    base = (center + U[..., None] * e_u + V[..., None] * e_v).reshape(-1, 3)
    # Abscisses des échantillons le long des rayons (de l'avant vers l'arrière de la sphère englobante).
    s_vals = np.linspace(-half_diag, half_diag, n_steps)
    # Longueur d'un pas Δs.
    ds = s_vals[1] - s_vals[0]
    # Table de couleurs (256 entrées RGB) : plus rapide que d'appeler la colormap à chaque pas.
    lut = cmap(np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    # Gradient du volume (np.gradient avec les pas physiques) pour l'éclairage, en float32 (mémoire).
    gx, gy, gz = (g.astype(np.float32) for g in np.gradient(vol, *spacing))
    # Direction unitaire de la lumière.
    L = np.asarray(light, dtype=float) / np.linalg.norm(light)
    # Couleur accumulée (noire au départ) et transmittance (1 = rien d'absorbé) de chaque pixel.
    C = np.zeros((nx * ny, 3), dtype=np.float32)
    T = np.ones(nx * ny, dtype=np.float32)
    for s in s_vals:
        # Points d'échantillonnage du pas courant et indices fractionnaires dans la grille (forme (3, npix)).
        coords = ((base + s * d - origin) / spacing).T
        # Interpolation trilinéaire (order=1) ; hors de la grille : 0 (mode="constant", cval=0).
        val = ndimage.map_coordinates(vol, coords, order=1, mode="constant", cval=0.0, prefilter=False)
        # Tranche vide : rien à accumuler.
        if val.max() < 1e-4:
            continue
        # Opacité de Beer–Lambert et couleur d'émission (indice de LUT entre 0 et 255).
        alpha = 1.0 - np.exp(-density * val * ds)
        col = lut[np.clip((val * 255.0).astype(int), 0, 255)]
        # Éclairage : normale -∇e/|∇e| et terme de Lambert borné dans [0, 1].
        g = np.stack([ndimage.map_coordinates(gq, coords, order=1, mode="constant", cval=0.0, prefilter=False)
                      for gq in (gx, gy, gz)], axis=1)
        lam = np.clip(-(g @ L) / (np.linalg.norm(g, axis=1) + 1e-12), 0.0, 1.0)
        # Couleur éclairée : terme ambiant (fumée jamais totalement noire) + terme diffus de Lambert.
        col = col * (ambient + diffuse * lam)[:, None]
        # Accumulation avant → arrière (alpha blending) puis atténuation de la transmittance.
        C += (T * alpha)[:, None] * col
        T *= 1.0 - alpha
    # Image finale : lumière émise + fond vu à travers la transmittance restante, remise en forme (ny, nx, 3).
    img = C.reshape(ny, nx, 3) + T.reshape(ny, nx, 1) * bg
    return np.clip(img, 0.0, 1.0)


def axisym_enstrophy_volume(snap, Z_ring, half_width=2.0, r_max=2.0, resolution=150):
    """Volume 3D de l'enstrophie ½ω² reconstruit par révolution autour de l'anneau (grille cubique).

    Renvoie (volume (n, n, nz), origine (3,), pas (3,)).
    """
    # Densité d'enstrophie aux centres des cellules.
    ens = ax2d.enstrophy_density(snap)
    # Grille cartésienne : même pas h dans les trois directions.
    h = 2.0 * r_max / (resolution - 1)
    # Axe x (et y, identique) de -r_max à r_max.
    x = np.linspace(-r_max, r_max, resolution)
    # Nombre de plans en z pour garder un pas voisin de h sur la fenêtre axiale.
    nz = int(round(2.0 * half_width / h)) + 1
    zq = np.linspace(Z_ring - half_width, Z_ring + half_width, nz)
    # Balayage azimutal du champ méridien (z périodique).
    vol = revolve_field(snap["r_c"], snap["z_c"], ens, x, x, zq, periodic_z=True)
    # Volume, origine (premier nœud de la grille) et pas selon x, y, z.
    return vol, np.array([x[0], x[0], zq[0]]), np.array([h, h, zq[1] - zq[0]])


def _upsample_closed(nodes, factor):
    """Rééchantillonne finement une courbe fermée (spline cubique périodique, ``factor`` × plus de points)."""
    # Polygone fermé (premier nœud répété) et abscisse curviligne cumulée.
    P = np.vstack([nodes, nodes[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    # Spline périodique passant par les nœuds, évaluée en factor × N abscisses régulières.
    spline = CubicSpline(s, P, bc_type="periodic", axis=0)
    return spline(np.linspace(0.0, s[-1], factor * len(nodes), endpoint=False))


def filament_enstrophy_volume(frame, gammas, deltas, resolution=112, margin=0.35, upsample=3, min_delta_cells=2.0):
    """Volume 3D de l'enstrophie ½|ω|² des filaments (vorticité reconstruite par « splatting »).

    Les filaments sont d'abord rééchantillonnés (``upsample`` × plus de nœuds) : l'espacement des
    particules devient petit devant δ, ce qui supprime l'aspect « perlé » du rendu.
    Anti-crénelage : quand les anneaux s'étalent, la boîte grandit et le pas h de la grille aussi ; un noyau
    plus étroit que ``min_delta_cells`` mailles n'est alors plus résolu (tubes « en pointillés »). Le δ de
    rendu est porté à min_delta_cells × h ; la circulation (intégrale du noyau) est inchangée.
    """
    if upsample > 1:
        # Liste en compréhension : chaque filament est remplacé par sa version rééchantillonnée.
        frame = [_upsample_closed(nodes, upsample) for nodes in frame]
    # Boîte englobant tous les nœuds, élargie de la marge et de 4 δ.
    pts = np.concatenate(frame, axis=0)
    pad = margin + 4.0 * max(deltas)
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    # Pas commun (voxels cubiques) fixé par la plus grande dimension.
    h = (hi - lo).max() / (resolution - 1)
    # Nombre de nœuds par axe (np.ceil : arrondi supérieur, la boîte est entièrement couverte).
    shape = np.ceil((hi - lo) / h).astype(int) + 1
    # δ de rendu : au moins min_delta_cells mailles (anti-crénelage), sinon le δ physique du filament.
    deltas_r = [max(d, min_delta_cells * h) for d in deltas]
    # Vorticité 3D reconstruite (3, nx, ny, nz) à partir des segments des filaments.
    omega = vp.splat_vorticity(frame, gammas, deltas_r, lo, np.array([h, h, h]), shape)
    # ½ |ω|² (somme des carrés des trois composantes).
    return 0.5 * (omega**2).sum(axis=0), lo, np.array([h, h, h])


def plot_volume_renderings(images, titles, title=None, ncols=None):
    """Planche d'images de rendu volumique sur fond sombre."""
    # Nombre d'images, de colonnes (une seule ligne par défaut) et de lignes.
    n = len(images)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    # Rapport d'aspect des images (hauteur / largeur) pour dimensionner la figure.
    h, w = images[0].shape[:2]
    # Largeur 4.8 pouces par image, hauteur proportionnelle (+ place pour le titre).
    # squeeze=False : axes toujours rangés dans un tableau 2D ; facecolor : fond de figure sombre.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.8 * (h / w) * nrows + 0.6),
                             squeeze=False, facecolor=DARK_BG[1])
    # axes.flat : parcours des axes ligne par ligne, en parallèle des images et de leurs titres.
    for ax, img, ttl in zip(axes.flat, images, titles):
        # imshow : affichage de l'image RGB (interpolation bilinéaire à l'agrandissement) ; axis("off") masque les axes.
        ax.imshow(img, interpolation="bilinear")
        # Titre clair, lisible sur le fond sombre.
        ax.set_title(ttl, color="#e9eef4", loc="center", fontsize=10)
        ax.axis("off")
    # Panneaux inutilisés masqués.
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    # Titre général en blanc (viz._suptitle écrirait en encre noire, invisible sur ce fond).
    fig.suptitle(title or "Rendu volumique de l'enstrophie ½ω²", x=0.01, ha="left", color="#ffffff",
                 fontsize=12, fontweight="bold")
    # Marges réduites : les images occupent presque toute la figure.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03, hspace=0.12)
    return fig


def animate_smoke_rings(hist, path, fps=20, stride=2, resolution=88, image_size=(360, 280), elev=16.0, azim=-62.0,
                        density=7.0, title="Anneaux de fumée (enstrophie des filaments)"):
    """Animation MP4/GIF du rendu volumique de l'enstrophie des filaments (une image par sauvegarde)."""
    # Indices des sauvegardes animées (une sur « stride »).
    idx = range(0, len(hist.frames), stride)

    def render(k):
        # Image n° k : enstrophie des filaments reconstruite sur une grille 3D, puis lancer de rayons.
        vol, org, sp = filament_enstrophy_volume(hist.frames[k], hist.gammas, hist.deltas[k], resolution=resolution)
        return volume_render(vol, org, sp, elev=elev, azim=azim, image_size=image_size, density=density)
    # Liste explicite des indices (accès par numéro d'image dans update).
    ks = list(idx)
    # Figure aux dimensions de l'image (80 pixels par pouce), fond sombre.
    fig, ax = plt.subplots(figsize=(image_size[0] / 80, image_size[1] / 80), facecolor=DARK_BG[1])
    # Premier rendu : crée l'objet image, dont seules les données seront remplacées ensuite.
    im = ax.imshow(render(ks[0]), interpolation="bilinear")
    ax.axis("off")
    # Titre vide, complété à chaque image (instant courant).
    ttl = ax.set_title("", color="#e9eef4", fontsize=9)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)

    def update(n):
        # Mise à jour de l'image n de l'animation : nouveau rendu (im.set_data) et nouveau titre.
        k = ks[n]
        im.set_data(render(k))
        ttl.set_text(f"{title}   t = {hist.times[k]:.1f}")
        # FuncAnimation attend la liste (ici un tuple) des objets modifiés.
        return (im,)

    # FuncAnimation(fig, fonction, frames, interval) : appelle update(0), update(1)... ; interval en ms.
    anim = animation.FuncAnimation(fig, update, frames=len(ks), interval=1000 / fps, blit=False)
    # Création du dossier de destination si besoin (exist_ok=True : pas d'erreur s'il existe déjà).
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".mp4") and (animation.writers.is_available("ffmpeg") or shutil.which("ffmpeg")):
        # MP4 (codec H.264, format de pixels yuv420p lisible par tous les lecteurs) si ffmpeg est installé.
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    else:
        # Sinon GIF animé via Pillow (extension du fichier remplacée : os.path.splitext sépare l'extension).
        path = os.path.splitext(path)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
    # anim.save : calcule toutes les images et les écrit dans le fichier ; dpi = pixels par pouce ;
    # savefig_kwargs transmet la couleur de fond sombre de la figure (sinon le bandeau du titre serait clair).
    anim.save(path, writer=writer, dpi=100, savefig_kwargs={"facecolor": fig.get_facecolor()})
    # Libération de la figure (sinon elle resterait en mémoire).
    plt.close(fig)
    return path
