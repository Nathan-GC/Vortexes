"""
appli/vues3d.py — Vues 3D et animations interactives (Plotly) : rotation à la souris, zoom, lecture/pause.

* animation_filaments : filaments 3D animés (bouton Lecture/Pause et curseur temporel), vue orientable ;
* tubes : tubes de vorticité d'un instant (rayon ∝ δ, couleur = Γ) ;
* isosurface : isosurface du critère Q ou λ₂ d'un instantané axisymétrique, reconstruite par révolution ;
* animation_vorticite : vorticité du plan méridien animée, avec les lignes de courant.

Les figures sont construites au moment de l'affichage à partir des données du calcul (aucun calcul lourd).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# numpy : tableaux ; plotly.graph_objects : figures interactives (traces Scatter3d, Surface, Heatmap...).
import numpy as np
import plotly.graph_objects as go
# to_hex : conversion d'une couleur matplotlib en code hexadécimal « #rrggbb ».
from matplotlib.colors import to_hex

# Modules du projet : critères Q / λ₂ (solveur), palette et fenêtres (figures 2D), maillages 3D.
from src import solver_2d_axisymmetric as ax
from src import visualization as viz
from src import visualization_3d as v3


def echelle_couleurs(cmap, n=11):
    """Échelle de couleurs Plotly [[position, couleur], …] échantillonnée sur une palette matplotlib du projet."""
    # n positions régulières de 0 à 1 ; cmap(x) renvoie la couleur RGBA à la position x.
    return [[k / (n - 1), to_hex(cmap(k / (n - 1)))] for k in range(n)]


# Palettes du projet : divergente (bleu ↔ gris ↔ rouge, grandeurs signées) et séquentielle bleue.
DIVERGENTE = echelle_couleurs(viz.CMAP_DIV)
SEQUENTIELLE = echelle_couleurs(viz.CMAP_SEQ)
# Éclairage des surfaces 3D (ambiant, diffus, spéculaire, rugosité) et point de vue initial de la caméra.
ECLAIRAGE = dict(ambient=0.45, diffuse=0.75, specular=0.25, roughness=0.6)
CAMERA = dict(eye=dict(x=1.45, y=-1.45, z=0.75))


def _mise_en_page(fig, hauteur, titre=None):
    """Réglages communs : hauteur, marges, police, fond transparent (s'accorde au thème clair ou sombre)."""
    fig.update_layout(height=hauteur, margin=dict(l=10, r=10, t=40 if titre else 10, b=10),
                      font=dict(family="Segoe UI, DejaVu Sans, sans-serif", size=12),
                      paper_bgcolor="rgba(0,0,0,0)", title=dict(text=titre, x=0.01) if titre else None,
                      legend=dict(orientation="h", y=1.02, x=0.0))


def _controles_animation(fig, noms, duree_ms=80, prefixe="t = "):
    """Boutons Lecture/Pause et curseur temporel d'une figure animée (une étape par image nommée)."""
    # Réglages d'une étape : durée d'affichage, redessin complet (nécessaire en 3D), pas de transition.
    def etape(duree):
        return dict(frame=dict(duration=duree, redraw=True), mode="immediate", transition=dict(duration=0))

    fig.update_layout(
        # updatemenus : boutons ; « animate » avec None joue toutes les images à partir de l'image courante.
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                          pad=dict(t=45, r=10), showactive=False,
                          buttons=[dict(label="▶ Lecture", method="animate",
                                        args=[None, dict(etape(duree_ms), fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate", args=[[None], etape(0)])])],
        # sliders : curseur dont chaque étape affiche l'image correspondante.
        sliders=[dict(active=0, x=0.17, len=0.83, y=0.0, yanchor="top", pad=dict(t=35),
                      currentvalue=dict(prefix=prefixe, font=dict(size=13)),
                      steps=[dict(label=nom, method="animate", args=[[nom], etape(0)]) for nom in noms])])


def animation_filaments(hist, etiquettes=None, suivre=False, fenetre=3.2, n_images=60):
    """Filaments 3D animés : une courbe fermée par filament, couleur par anneau, vue orientable à la souris.

    ``suivre`` : la boîte de vue suit le barycentre des filaments (côté ``fenetre``), sinon elle englobe tout le
    mouvement. ``n_images`` : nombre maximal d'images de l'animation (sous-échantillonnage de l'historique).
    """
    n = len(hist.frames)
    # Indices des images retenues (np.unique retire les doublons dus à l'arrondi).
    idx = np.unique(np.linspace(0, n - 1, min(n_images, n)).round().astype(int))
    # Une couleur et un nom par anneau (identifiants triés), dans l'ordre de la palette du projet.
    anneaux = sorted(set(hist.ring_ids))
    couleur = {rid: viz.SERIES[k % len(viz.SERIES)] for k, rid in enumerate(anneaux)}
    nom = {rid: (etiquettes[k] if etiquettes and k < len(etiquettes) else f"anneau {k + 1}")
           for k, rid in enumerate(anneaux)}

    def traces(k):
        # Courbes de l'image k : premier nœud répété en fin de liste pour fermer chaque filament ; un seul
        # élément de légende par anneau (showlegend) même s'il compte plusieurs filaments.
        vus, liste = set(), []
        for nodes, rid in zip(hist.frames[k], hist.ring_ids):
            P = np.vstack([nodes, nodes[:1]])
            liste.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", name=nom[rid],
                                      line=dict(color=couleur[rid], width=5), legendgroup=str(rid),
                                      showlegend=rid not in vus, hoverinfo="name"))
            vus.add(rid)
        return liste

    # Boîte englobant tout le mouvement (cube, viz._frame_bounds) ou boîte mobile centrée sur les filaments.
    bas, haut = viz._frame_bounds(hist.frames)

    def axes(k):
        if suivre:
            centre = np.concatenate(hist.frames[k]).mean(axis=0)
            lo, hi = centre - 0.5 * fenetre, centre + 0.5 * fenetre
        else:
            lo, hi = bas, haut
        return dict(xaxis=dict(range=[lo[0], hi[0]], title="x"), yaxis=dict(range=[lo[1], hi[1]], title="y"),
                    zaxis=dict(range=[lo[2], hi[2]], title="z"))

    # Noms des images = instants ; en mode suivi, chaque image déplace aussi la boîte de vue (layout de l'image).
    noms = [f"{hist.times[k]:.2f}" for k in idx]
    images = []
    for k, nm in zip(idx, noms):
        options = {"layout": go.Layout(scene=axes(k))} if suivre else {}
        images.append(go.Frame(data=traces(k), name=nm, **options))
    fig = go.Figure(data=traces(idx[0]), frames=images)
    # aspectmode="cube" : mêmes échelles sur les trois axes (géométrie non déformée).
    fig.update_layout(scene=dict(**axes(idx[0]), aspectmode="cube", camera=CAMERA))
    _mise_en_page(fig, 640)
    _controles_animation(fig, noms)
    return fig


def tubes(frame, gammas, deltas, temps, rayon_echelle=1.0, fenetre=3.2, n_cercle=12):
    """Tubes de vorticité 3D d'un instant : surface lisse autour de chaque filament, couleur = circulation Γ."""
    # Circulation maximale (échelle de couleur symétrique) ; « or 1.0 » évite une échelle nulle.
    gmax = max(abs(g) for g in gammas) or 1.0
    traces = []
    for k, (nodes, gamma, delta) in enumerate(zip(frame, gammas, deltas)):
        # v3.tube_mesh : maillage (M+1, n_cercle+1, 3) d'un tube de rayon rayon_echelle × δ autour du filament.
        P = v3.tube_mesh(nodes, rayon_echelle * delta, n_circle=n_cercle)
        # Surface Plotly ; surfacecolor constant = Γ du filament ; une seule barre de couleur (premier tube).
        traces.append(go.Surface(x=P[..., 0], y=P[..., 1], z=P[..., 2], surfacecolor=np.full(P.shape[:2], gamma),
                                 cmin=-gmax, cmax=gmax, colorscale=DIVERGENTE, showscale=k == 0,
                                 colorbar=dict(title="Γ", len=0.6), lighting=ECLAIRAGE,
                                 hovertemplate=f"Γ = {gamma:.3f}<extra></extra>"))
    # Boîte de vue centrée sur le barycentre des filaments.
    centre = np.concatenate(frame).mean(axis=0)
    lo, hi = centre - 0.5 * fenetre, centre + 0.5 * fenetre
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(xaxis=dict(range=[lo[0], hi[0]], title="x"), yaxis=dict(range=[lo[1], hi[1]], title="y"),
                                 zaxis=dict(range=[lo[2], hi[2]], title="z"), aspectmode="cube", camera=CAMERA))
    _mise_en_page(fig, 580, titre=f"Tubes de vorticité — t = {temps:.2f}")
    return fig


def isosurface(snap, Z_ring, demi, r_max, critere="Q", fraction=0.04, coupe=90.0, n_theta=72):
    """Isosurface du critère Q (ou −λ₂) d'un instantané axisymétrique, reconstruite par révolution autour de
    l'axe z et colorée par la norme de la vitesse ; un secteur de ``coupe`` degrés est retiré face à la caméra."""
    # Critère aux centres des cellules (positif dans les cœurs) et norme de la vitesse (np.hypot).
    crit = ax.q_criterion(snap) if critere == "Q" else -ax.lambda2_criterion(snap)
    vitesse = np.hypot(snap["u_r"], snap["u_z"])
    # Fenêtre méridienne autour de l'anneau (v3._meridian_window gère la périodicité en z).
    rw, zw, cw = v3._meridian_window(snap["r_c"], snap["z_c"], crit, Z_ring, demi, r_max)
    _, _, colw = v3._meridian_window(snap["r_c"], snap["z_c"], vitesse, Z_ring, demi, r_max)
    # Niveau de l'isosurface : fraction du maximum du critère dans la fenêtre.
    niveau = fraction * cw.max()
    # Secteur retiré centré sur l'azimut de la caméra (−45°) ; balayage sur le reste du tour.
    ouverture = np.radians(coupe)
    theta_c = -0.25 * np.pi
    surfaces = v3.isosurface_of_revolution(rw, zw, cw, niveau, colw, n_theta=n_theta, theta0=theta_c + 0.5 * ouverture,
                                           theta_span=2.0 * np.pi - ouverture)
    # Échelle de couleur sur la plage utile (2ᵉ–98ᵉ centiles, np.percentile).
    valeurs = np.concatenate([s["C"].ravel() for s in surfaces]) if surfaces else colw.ravel()
    cmin, cmax = np.percentile(valeurs, [2.0, 98.0])
    traces = [go.Surface(x=s["X"], y=s["Y"], z=s["Z"], surfacecolor=s["C"], colorscale=SEQUENTIELLE, cmin=cmin,
                         cmax=max(cmax, cmin + 1e-9), showscale=k == 0, colorbar=dict(title="|u|", len=0.6),
                         lighting=ECLAIRAGE, hovertemplate="|u| = %{surfacecolor:.3f}<extra></extra>")
              for k, s in enumerate(surfaces)]
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(xaxis=dict(range=[-r_max, r_max], title="x"), yaxis=dict(range=[-r_max, r_max], title="y"),
                                 zaxis=dict(range=[Z_ring - demi, Z_ring + demi], title="z"), aspectmode="data",
                                 camera=CAMERA))
    nom = "Q" if critere == "Q" else "−λ₂"
    _mise_en_page(fig, 580, titre=f"Isosurface {nom} = {fraction:.0%} du maximum (couleur |u|) — t = {snap['t']:.2f}")
    return fig


def animation_vorticite(exp, taille_max=120):
    """Vorticité ω_y du plan méridien (r en miroir) animée, avec les lignes de courant ψ, dans une fenêtre fixe.

    ``exp`` : données enregistrées pendant le calcul (images {t, omega, psi}, grille r_f/z_f, échelle vmax,
    fenêtre centre/demi, r_max). Les champs sont sous-échantillonnés à ``taille_max`` points en z pour garder
    une figure légère.
    """
    images, r_f, z_f, vmax = exp["images"], exp["r_f"], exp["z_f"], exp["vmax"]
    # Colonnes z de la fenêtre (viz._window, domaine périodique) et rayons affichés.
    kf, zw = viz._window(z_f, exp["centre"], exp["demi"])
    ir = np.nonzero(r_f <= exp["r_max"])[0]
    # Pas de sous-échantillonnage (np.ceil : arrondi supérieur) ; indices et coordonnées retenus.
    pas_z = max(1, int(np.ceil(len(kf) / taille_max)))
    pas_r = max(1, int(np.ceil(len(ir) / (taille_max // 2))))
    kz, zs, ir, rs = kf[::pas_z], zw[::pas_z], ir[::pas_r], r_f[ir[::pas_r]]

    def champs(image):
        # np.ix_ : sous-tableau (rayons ir × colonnes kz) ; viz._mirror : coupe complète x = ±r (ω impaire, ψ paire).
        rr, oo = viz._mirror(rs, image["omega"][np.ix_(ir, kz)], odd=True)
        _, pp = viz._mirror(rs, image["psi"][np.ix_(ir, kz)])
        # float32 : données deux fois plus légères à transmettre au navigateur.
        return rr, oo.astype(np.float32), pp.astype(np.float32)

    rr, _, p0 = champs(images[0])
    # Niveaux fixes des lignes de courant, répartis sur la plage de ψ de l'image initiale.
    pmin, pmax = float(p0.min()), float(p0.max())
    pas_psi = (pmax - pmin) / 14.0 if pmax > pmin else 1.0

    def traces(image):
        _, oo, pp = champs(image)
        # Carte de vorticité (échelle ±vmax commune à toutes les images) et contours de ψ en gris.
        return [go.Heatmap(x=zs, y=rr, z=oo, colorscale=DIVERGENTE, zmin=-vmax, zmax=vmax, zsmooth="best",
                           colorbar=dict(title="ω_y"),
                           hovertemplate="z = %{x:.2f}<br>x = %{y:.2f}<br>ω = %{z:.3f}<extra></extra>"),
                go.Contour(x=zs, y=rr, z=pp, showscale=False, hoverinfo="skip", line=dict(width=0.8),
                           contours=dict(coloring="lines", start=pmin + pas_psi, end=pmax - pas_psi, size=pas_psi),
                           colorscale=[[0.0, viz.INK["secondary"]], [1.0, viz.INK["secondary"]]])]

    # Images nommées par leur instant ; un instant déjà présent (deux images au même pas) est ignoré.
    noms, cadres = [], []
    for image in images:
        nm = f"{image['t']:.2f}"
        if nm not in noms:
            noms.append(nm)
            cadres.append(go.Frame(data=traces(image), name=nm))
    fig = go.Figure(data=cadres[0].data, frames=cadres)
    # Bornes des axes = fenêtre affichée ; même échelle sur les deux axes (scaleanchor), la zone de tracé se réduisant
    # au besoin (constrain="domain") au lieu d'élargir les bornes : géométrie non déformée, sans marge vide.
    fig.update_layout(xaxis=dict(title="z", range=[float(zs.min()), float(zs.max())], constrain="domain"),
                      yaxis=dict(title="x  (r en miroir)", range=[float(rr.min()), float(rr.max())], scaleanchor="x",
                                 scaleratio=1, constrain="domain"))
    _mise_en_page(fig, 470)
    _controles_animation(fig, noms, duree_ms=150)
    return fig
