"""
Tests des diagnostics et visualisations avancés :

* critères Q et λ2, tenseur des gradients, dissipation Φ = 2μ S:S et densité d'enstrophie
  (``src/solver_2d_axisymmetric.py``) ;
* suivi individuel de deux anneaux et enregistrement des profils de Hovmöller ;
* reconstruction 3D par révolution, isosurfaces de révolution, tubes de vorticité, rendu volumique
  (``src/visualization_3d.py``) ;
* densité axiale de vorticité et « splatting » 3D des filaments (``src/vortex_particles.py``).
"""

# numpy : tableaux et calcul vectoriel ; pytest : fixtures et comparaisons approchées (pytest.approx).
import numpy as np
import pytest
# ListedColormap : palette réduite à une seule couleur (sert à annuler l'émission dans le test de la dalle).
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm

# Modules testés (visualization fixe le moteur de rendu non interactif « Agg » dès son import).
from src import solver_2d_axisymmetric as ax
from src import visualization as viz
from src import visualization_3d as v3
from src import vortex_particles as vp

# pyplot après l'import de visualization (moteur « Agg » déjà choisi : aucune fenêtre ne s'ouvre).
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================================
# Fixtures : instantanés partagés (construits une seule fois pour tout le fichier)
# =============================================================================================
@pytest.fixture(scope="module")
def hill_snap():
    # Vortex sphérique de Hill (a = 1, U = 1) centré en z = 5, sur une grille fine près du vortex.
    g = ax.AxisymmetricGrid.stretched(0.025, 1.6, 10.0, 1.05, 10.0, 400)
    s = ax.AxisymmetricNavierStokesSolver(g, nu=0.0)
    s.init_hill_vortex(a=1.0, U=1.0, z0=5.0)
    # fields() : vitesses aux centres, pression cohérente, vitesses sur la grille décalée, grille, ν.
    return s.fields()


@pytest.fixture(scope="module")
def ring_case():
    # Anneau de Lamb–Oseen (R = 1, a = 0.2, Γ = 1) en z = 2.5, ν = 1e-3 : solveur et instantané.
    g = ax.AxisymmetricGrid.stretched(0.025, 2.0, 6.0, 1.06, 6.0, 240)
    s = ax.AxisymmetricNavierStokesSolver(g, nu=1e-3)
    s.init_lamb_oseen_ring(R=1.0, a=0.2, Gamma=1.0, z0=2.5)
    return s, s.fields()


@pytest.fixture(scope="module")
def two_rings():
    # Deux anneaux coaxiaux identiques (σ = 0.155) séparés de 0.6 ; la liste est volontairement donnée
    # dans le désordre (anneau avant en premier) pour vérifier le classement « 0 = le plus en arrière ».
    g = ax.AxisymmetricGrid.stretched(0.025, 2.0, 6.0, 1.06, 8.0, 320)
    s = ax.AxisymmetricNavierStokesSolver(g, nu=1.0 / 3000.0)
    s.init_rings([{"R": 1.0, "a": 0.155, "Gamma": 1.0, "z0": 2.6}, {"R": 1.0, "a": 0.155, "Gamma": 1.0, "z0": 2.0}])
    # Suivi à t = 0, puis court calcul avec profils de Hovmöller et suivi des deux anneaux.
    first = s.track_rings(2)
    n_peaks0 = s._last_n_peaks
    res = s.run(0.3, diag_interval=0.1, record_profiles=True, n_rings=2)
    return {"solver": s, "first": first, "n_peaks0": n_peaks0, "res": res}


# =============================================================================================
# Critères d'identification des tourbillons, dissipation, enstrophie
# =============================================================================================
def test_q_lambda2_signs_hill_vortex(hill_snap):
    """Q > 0 et λ2 < 0 au cœur du vortex de Hill ; Q < 0 et λ2 > 0 dans l'écoulement potentiel extérieur."""
    Q = ax.q_criterion(hill_snap)
    L2 = ax.lambda2_criterion(hill_snap)
    g = hill_snap["grid"]
    # Centre du cœur (r = a/√2, z = z0) : point où la vitesse relative au vortex s'annule (rotation pure).
    i0 = np.argmin(np.abs(g.rc - 1.0 / np.sqrt(2.0)))
    j0 = np.argmin(np.abs(g.zc - 5.0))
    assert Q[i0, j0] > 0.3 * Q.max()
    assert L2[i0, j0] < 0.3 * L2.min()
    # Extérieur de la sphère (ρ > 1.3 a) : écoulement irrotationnel → Q = -½ S:S ≤ 0 et S² ⪰ 0 → λ2 ≥ 0
    # (tolérance relative 1e-8 pour les arrondis du champ discret).
    Rc, Zc = g.center_mesh()
    far = np.hypot(Rc, Zc - 5.0) > 1.3
    assert Q[far].max() < 1e-8 * Q.max()
    assert L2[far].min() > -1e-8 * np.abs(L2).max()


def test_q_criterion_equals_half_pressure_laplacian(ring_case):
    """Identité des écoulements incompressibles : ∇²p/ρ = 2Q (vérifiée dans le cœur de l'anneau)."""
    s, snap = ring_case
    Q = ax.q_criterion(snap)
    # Laplacien discret de la pression : divergence du gradient (mêmes opérateurs que la projection).
    lap = s.divergence(*s.gradient(snap["p"]))
    # Zone où Q dépasse 5 % de son maximum ; erreur relative en norme L2 (np.linalg.norm).
    core = Q > 0.05 * Q.max()
    err = np.linalg.norm((0.5 * lap - Q)[core]) / np.linalg.norm(Q[core])
    assert err < 1e-2


def test_dissipation_integral_equals_enstrophy(ring_case):
    """∫Φ dV = μ ∫ω² dV (fluide incompressible, paroi glissante) ; Φ ≥ 0 ; tr ∇u = 0."""
    s, snap = ring_case
    phi = ax.dissipation_rate(snap)
    d = s.diagnostics()
    # d["dissipation"] = ν ∫ω² dV calculé aux coins par le solveur (ρ = 1 → μ = ν).
    assert ax.volume_integral(snap, phi) == pytest.approx(d["dissipation"], rel=1e-5)
    # La densité d'enstrophie ½ω² intégrée redonne ∫ω² dV / 2.
    assert ax.volume_integral(snap, ax.enstrophy_density(snap)) == pytest.approx(d["dissipation"] / (2 * s.nu), rel=1e-8)
    # Φ = 2μ S:S est une forme quadratique positive.
    assert phi.min() >= 0.0
    # Trace du tenseur des gradients ∂u_r/∂r + ∂u_z/∂z + u_r/r = div u = 0 (projection exacte).
    gt = ax.velocity_gradient_tensor(snap)
    assert np.abs(gt["a"] + gt["d"] + gt["e"]).max() < 1e-10


# =============================================================================================
# Suivi des anneaux et profils de Hovmöller
# =============================================================================================
def test_track_rings_identifies_two_rings(two_rings):
    first = two_rings["first"]
    # Deux pics distincts trouvés ; tableau (2, 3) de colonnes (Γ, R, Z).
    assert two_rings["n_peaks0"] == 2
    assert first.shape == (2, 3)
    # Circulation de chaque anneau ≈ 1 (le seuil de masse de 2 % écarte ~1,5 % des queues gaussiennes).
    np.testing.assert_allclose(first[:, 0], 1.0, rtol=0.03)
    # Rayon de Saffman d'un cœur gaussien : R² = R0² + a²/2.
    np.testing.assert_allclose(first[:, 1], np.sqrt(1.0 + 0.155**2 / 2.0), rtol=2e-3)
    # Anneau 0 = le plus en arrière (z = 2.0), anneau 1 = l'anneau avant (z = 2.6).
    np.testing.assert_allclose(first[:, 2], [2.0, 2.6], atol=5e-3)


def test_run_records_profiles_and_ring_tracks(two_rings):
    res = two_rings["res"]
    s = two_rings["solver"]
    prof, rg = res["profiles"], res["rings"]
    nt, nz = len(prof["t"]), s.g.Nz
    # Profils de Hovmöller : une ligne par diagnostic, une colonne par position axiale.
    assert prof["omega_max"].shape == (nt, nz) and prof["p_min"].shape == (nt, nz)
    assert np.all(np.isfinite(prof["p_min"])) and np.all(prof["omega_max"] >= -1e-12)
    # La dépression maximale (p - p∞ < 0) se trouve dans les cœurs.
    assert prof["p_min"].min() < 0.0
    # Suivi : Γ, R, Z de forme (nt, 2), deux pics à chaque instant, anneaux restés ordonnés et avancés.
    assert rg["Z"].shape == (nt, 2) and np.all(rg["n_peaks"] == 2)
    assert np.all(rg["Z"][:, 0] < rg["Z"][:, 1])
    assert np.all(np.diff(rg["Z"], axis=0) > 0.0)


# =============================================================================================
# Reconstruction 3D par révolution et isosurfaces
# =============================================================================================
def test_revolve_field_matches_analytic():
    """Balayage azimutal d'un champ f(r, z) périodique en z sur une grille radiale non uniforme."""
    # Grille radiale uniforme jusqu'à r = 2 puis étirée (np.cumsum : abscisses cumulées).
    r = np.concatenate([np.linspace(0.0, 2.0, 81), 2.0 + np.cumsum(0.025 * 1.05 ** np.arange(1, 30))])
    # Centres de cellules d'un domaine périodique de longueur Lz.
    Lz, Nz = 4.0, 160
    z = (np.arange(Nz) + 0.5) * Lz / Nz
    f = np.exp(-((r[:, None] - 1.0) ** 2) / 0.2) * np.cos(2.0 * np.pi * z[None, :] / Lz)
    # Grille cartésienne cible, débordant du domaine en z (test du repliement périodique).
    x = np.linspace(-2.5, 2.5, 41)
    y = np.linspace(-2.5, 2.5, 37)
    zq = np.linspace(-2.0, 6.0, 53)
    F = v3.revolve_field(r, z, f, x, y, zq, fill=-7.0)
    assert F.shape == (41, 37, 53) and F.dtype == np.float32
    # Valeur analytique f(√(x² + y²), z) aux mêmes points.
    X, Y, Zq = np.meshgrid(x, y, zq, indexing="ij")
    Rq = np.hypot(X, Y)
    exact = np.exp(-((Rq - 1.0) ** 2) / 0.2) * np.cos(2.0 * np.pi * Zq / Lz)
    inside = Rq <= r[-1]
    # Erreur d'interpolation bilinéaire O(h²) ; au-delà de r_max : valeur de remplissage.
    assert np.abs(F - exact)[inside].max() < 3e-3
    assert np.all(F[~inside] == -7.0)


def test_isosurface_of_revolution_is_torus():
    """L'isosurface f = ½ d'une gaussienne méridienne est un tore de petit rayon s √(ln 2)."""
    rr = np.linspace(0.0, 2.5, 126)
    zz = np.linspace(0.0, 4.0, 201)
    s2 = 0.09
    field = np.exp(-((rr[:, None] - 1.0) ** 2 + (zz[None, :] - 2.0) ** 2) / s2)
    # Coloration par le rayon r (np.broadcast_to : vue (Nr, Nz) sans copie) ; secteur de 270° à partir de 0.3 rad.
    theta0, span = 0.3, 1.5 * np.pi
    surfs = v3.isosurface_of_revolution(rr, zz, field, 0.5, color_field=np.broadcast_to(rr[:, None], field.shape),
                                        n_theta=40, theta0=theta0, theta_span=span)
    assert len(surfs) == 1
    sf = surfs[0]
    # Distance de chaque point du maillage au cercle central (r = 1, z = 2).
    rho = np.hypot(np.hypot(sf["X"], sf["Y"]) - 1.0, sf["Z"] - 2.0)
    np.testing.assert_allclose(rho, np.sqrt(s2 * np.log(2.0)), rtol=2e-3)
    # La couleur interpolée reproduit le rayon cylindrique du point.
    np.testing.assert_allclose(sf["C"], np.hypot(sf["X"], sf["Y"]), atol=1e-9)
    # Tous les angles sont dans [θ0, θ0 + span] (décalage de 1e-9 contre le repliement de -0 en 2π).
    th = np.mod(np.arctan2(sf["Y"], sf["X"]) - theta0 + 1e-9, 2.0 * np.pi) - 1e-9
    assert th.min() >= -1e-8 and th.max() <= span + 1e-8


# =============================================================================================
# Tubes de vorticité
# =============================================================================================
def test_tube_mesh_geometry_and_closure():
    # Cercle de rayon 1 (64 nœuds) : tous les points du tube sont à la distance « radius » du cercle.
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    circle = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    P = v3.tube_mesh(circle, 0.1, n_circle=16)
    # M = max(2 × 64, 48) = 128 sections (+1 répétée) de 16 + 1 points.
    assert P.shape == (129, 17, 3)
    dist = np.hypot(np.hypot(P[..., 0], P[..., 1]) - 1.0, P[..., 2])
    np.testing.assert_allclose(dist, 0.1, atol=1e-5)
    # Fermeture le long de la courbe (dernière section = première) et autour du cercle.
    assert np.array_equal(P[0], P[-1])
    np.testing.assert_allclose(P[:, 0], P[:, -1], atol=1e-12)
    # Nœud de trèfle (courbe gauche à torsion non nulle) : grâce à la correction d'holonomie, le pas entre
    # la dernière section et la première n'est pas plus grand que les autres (pas de « couture » vrillée).
    t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    knot = np.column_stack([(2 + np.cos(3 * t)) * np.cos(2 * t), (2 + np.cos(3 * t)) * np.sin(2 * t), np.sin(3 * t)])
    P = v3.tube_mesh(knot, 0.1, n_circle=12)
    # Déplacement maximal des points d'une section à la suivante (np.diff le long de la courbe).
    steps = np.linalg.norm(np.diff(P, axis=0), axis=2).max(axis=1)
    assert steps[-1] < 1.1 * np.median(steps)


def test_cut_tube_removes_sector():
    """Écorché : le secteur de 90° centré sur l'azimut -60° est retiré et les extrémités sont fermées."""
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    circle = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    P = v3.tube_mesh(circle, 0.1, n_circle=16)
    Qm, caps = v3._cut_tube(P, np.zeros(3), np.radians(-60.0), np.radians(90.0))
    # 3/4 des 128 sections conservées ; deux disques de 16 facettes (quadrilatères dégénérés).
    assert Qm.shape == (96, 17, 3) and caps.shape == (32, 4, 3)
    # Toutes les sections conservées sont hors du secteur (écart angulaire > 45°).
    C = Qm[:, :-1].mean(axis=1)
    dth = np.mod(np.degrees(np.arctan2(C[:, 1], C[:, 0])) + 60.0 + 180.0, 360.0) - 180.0
    assert np.abs(dth).min() > 45.0
    # Un tube entièrement hors du secteur est rendu intact (anneau décalé loin de l'axe du secteur).
    far = circle * 0.2 + np.array([5.0, 5.0, 0.0])
    P2 = v3.tube_mesh(far, 0.05, n_circle=8)
    same, none = v3._cut_tube(P2, np.zeros(3), np.radians(-120.0), np.radians(30.0))
    assert same is P2 and none is None


def test_gamma_colormap_modes():
    # Circulations des deux signes : palette divergente centrée sur 0.
    cmap, norm = v3._gamma_colormap([-0.5, 0.2, 1.0])
    assert cmap is viz.CMAP_DIV and isinstance(norm, TwoSlopeNorm) and norm.vcenter == 0.0
    # Un seul signe : demi-palette étirée entre |Γ|min et |Γ|max.
    cmap, norm = v3._gamma_colormap([0.1, 0.4, 0.2])
    assert isinstance(norm, Normalize) and norm.vmin == pytest.approx(0.1) and norm.vmax == pytest.approx(0.4)
    # Circulations toutes égales : plage [0, Γ] (pas de normalisation dégénérée vmin = vmax).
    _, norm = v3._gamma_colormap([0.3, 0.3])
    assert norm.vmin == 0.0 and norm.vmax == pytest.approx(0.3)
    # Circulations négatives : le filament le plus intense est à l'extrémité foncée (valeur normalisée 0).
    _, norm = v3._gamma_colormap([-0.2, -0.6])
    assert norm(-0.6) == pytest.approx(0.0) and norm(-0.2) == pytest.approx(1.0)


# =============================================================================================
# Rendu volumique
# =============================================================================================
def test_volume_render_beer_lambert_slab():
    """Dalle homogène vue de face : transmittance T = exp(-κ L) (loi de Beer–Lambert)."""
    # Dalle de 12 voxels d'épaisseur en z (pas 0.05 → L = 0.6), large en x et y.
    vol = np.zeros((30, 30, 24))
    vol[3:27, 3:27, 6:18] = 1.0
    # Palette noire (aucune émission) et fond blanc : la valeur d'un pixel est la transmittance du rayon.
    # elev = 90° : visée verticale ; image de dimensions impaires → pixel central (15, 20) au centre de la dalle.
    img = v3.volume_render(vol, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05), elev=90.0, azim=-90.0, image_size=(41, 31),
                           density=2.5, gamma=1.0, cmap=ListedColormap(["black"]), background=("#ffffff", "#ffffff"))
    assert img.shape == (31, 41, 3)
    # Les rampes d'interpolation trilinéaire aux deux faces ajoutent chacune ½ voxel : L_eff = 12 × 0.05.
    assert img[15, 20, 0] == pytest.approx(np.exp(-2.5 * 0.6), rel=5e-3)
    # Volume vide : l'image est le fond (ici rouge uni).
    img0 = v3.volume_render(np.zeros((5, 5, 5)), (0, 0, 0), (1, 1, 1), image_size=(8, 6), background=("#ff0000", "#ff0000"))
    np.testing.assert_allclose(img0, np.broadcast_to([1.0, 0.0, 0.0], (6, 8, 3)))


# =============================================================================================
# Filaments : densité axiale de vorticité et reconstruction 3D (« splatting »)
# =============================================================================================
def test_vorticity_line_density_integral():
    """∫λ dz = Σ |Γ| × longueur du filament (gaussiennes normalisées)."""
    ring = vp.make_ring(1.0, 1.0, 64, delta=0.1)
    frame = [f.nodes for f in ring]
    lam = vp.vorticity_line_density(frame, [f.gamma for f in ring], [f.delta for f in ring], np.linspace(-1.5, 1.5, 601))
    # Périmètre du polygone (64 segments) : somme des longueurs des côtés (np.roll ferme le polygone).
    perimeter = np.linalg.norm(np.roll(frame[0], -1, axis=0) - frame[0], axis=1).sum()
    assert np.trapezoid(lam, np.linspace(-1.5, 1.5, 601)) == pytest.approx(perimeter, rel=1e-6)
    assert perimeter == pytest.approx(2.0 * np.pi, rel=1e-3)


def test_splat_vorticity_flux_equals_circulation():
    """Le flux de ω reconstruit à travers un demi-plan méridien vaut Γ (queues du noyau exceptées)."""
    ring = vp.make_ring(1.0, 1.0, 64, delta=0.1)
    frame = [f.nodes for f in ring]
    # Grille de pas 0.03 ; le plan y = 0 correspond exactement à l'indice 50.
    h, origin, shape = 0.03, (-1.5, -1.5, -0.9), (101, 101, 61)
    om = vp.splat_vorticity(frame, [f.gamma for f in ring], [f.delta for f in ring], origin, (h, h, h), shape, support=8.0)
    assert om.shape == (3,) + shape and om.dtype == np.float32
    jy = int(round((0.0 - origin[1]) / h))
    xs = origin[0] + h * np.arange(shape[0])
    # Flux de ω_y à travers le demi-plan x > 0 (le filament y est dirigé selon +y) et composante ω_x nulle.
    flux = om[1][xs > 0, jy, :].sum() * h * h
    assert abs(om[0][xs > 0, jy, :].sum() * h * h) < 1e-6
    # Noyau de Rosenhead–Moore à queue lente : la fraction du flux au-delà de ρ vaut δ²/(δ² + ρ²) ; la
    # troncature (support 8δ) et le bord du domaine (5δ) en retirent quelques pour cent, jamais plus.
    assert 0.95 < flux <= 1.0


def test_filament_enstrophy_volume_antialiasing():
    """Grille grossière : le δ de rendu est porté à 2 mailles et le volume vaut ½|ω|² du splatting correspondant."""
    ring = vp.make_ring(1.0, 1.0, 32, delta=0.05)
    frame = [f.nodes for f in ring]
    gammas, deltas = [f.gamma for f in ring], [f.delta for f in ring]
    # upsample=1 : les nœuds sont utilisés tels quels (comparaison directe avec splat_vorticity).
    vol, org, sp = v3.filament_enstrophy_volume(frame, gammas, deltas, resolution=30, upsample=1)
    h = sp[0]
    # Pas cubique et δ physique plus petit que 2 mailles : l'anti-crénelage est actif.
    assert np.allclose(sp, h) and deltas[0] < 2.0 * h
    # Référence : splatting direct avec δ = 2h sur la même grille (vol.shape = forme de la grille).
    om = vp.splat_vorticity(frame, gammas, [2.0 * h], org, sp, vol.shape)
    np.testing.assert_allclose(vol, 0.5 * (om**2).sum(axis=0), rtol=1e-5, atol=1e-6 * vol.max())
    # min_delta_cells=0 : δ physique conservé, volume plus piqué (noyau plus étroit, maximum plus élevé).
    vol0, _, _ = v3.filament_enstrophy_volume(frame, gammas, deltas, resolution=30, upsample=1, min_delta_cells=0.0)
    assert vol0.max() > vol.max()


# =============================================================================================
# Figures : test de fumée (chaque fonction de tracé s'exécute et produit une figure)
# =============================================================================================
def test_advanced_figures_smoke(ring_case, two_rings):
    s, snap = ring_case
    res = two_rings["res"]
    figs = []
    # Bilan d'énergie : le dictionnaire doit contenir ν dans « params ».
    figs.append(viz.plot_energy_enstrophy_budget({"diagnostics": res["diagnostics"], "params": {"nu": two_rings["solver"].nu}}))
    figs.append(viz.plot_dissipation_map([snap], [2.5]))
    rg = res["rings"]
    figs.append(viz.plot_hovmoller(res["profiles"], tracks=[(rg["t"], rg["Z"][:, i], f"anneau {i}") for i in range(2)]))
    # Portrait de phase à partir de deux orbites synthétiques (cercle et spirale).
    t = np.linspace(0.0, 6.0, 120)
    orbit = {"label": "orbite", "t": t, "R": np.column_stack([0.3 * np.sin(t), -0.3 * np.sin(t)]),
             "Z": np.column_stack([0.3 * np.cos(t), -0.3 * np.cos(t)])}
    spiral = {"label": "spirale", "t": t, "R": orbit["R"] * np.exp(-0.1 * t)[:, None],
              "Z": orbit["Z"] * np.exp(-0.1 * t)[:, None], "ls": "--"}
    figs.append(viz.plot_phase_diagram([orbit, spiral]))
    figs.append(viz.plot_hovmoller_density(np.linspace(-1, 1, 50), t, np.random.default_rng(0).random((120, 50))))
    # Isosurfaces Q et λ2 de révolution (fenêtre centrée sur l'anneau).
    figs.append(v3.plot_isosurfaces(snap, 2.5, n_theta=24))
    # Tubes d'un anneau à cœur épais (7 filaments) en écorché.
    ring = vp.make_ring(1.0, 1.0, 24, core_radius=0.15, n_core_layers=1)
    figs.append(v3.plot_vortex_tubes([[f.nodes for f in ring]], [f.gamma for f in ring], [[f.delta for f in ring]], [0.0],
                                     cutaway=90.0, n_circle=8))
    # Rendu volumique de l'enstrophie reconstruite par révolution (petite image).
    vol, org, sp = v3.axisym_enstrophy_volume(snap, 2.5, half_width=1.4, r_max=1.5, resolution=40)
    img = v3.volume_render(vol, org, sp, image_size=(60, 45), n_steps=60)
    assert img.shape == (45, 60, 3) and np.all((img >= 0.0) & (img <= 1.0))
    figs.append(v3.plot_volume_renderings([img], ["test"]))
    for fig in figs:
        # Chaque objet est une figure matplotlib ; canvas.draw() effectue réellement le rendu (tri en
        # profondeur des facettes 3D, légendes, barres de couleur) ; plt.close libère sa mémoire.
        assert isinstance(fig, plt.Figure)
        fig.canvas.draw()
        plt.close(fig)
