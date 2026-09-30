"""
Tests de la loi de Biot–Savart régularisée (approche lagrangienne, ``src/vortex_particles.py``).

Références analytiques :
  * centre d'un anneau :            u = Γ / (2R)
  * axe d'un anneau :               u_z(z) = Γ R² / (2 (R² + z²)^{3/2})
  * hors axe :                      intégrales elliptiques (theory.ring_induced_velocity)
  * filament rectiligne infini :    u = Γ / (2π d)            (singulier)
                                    u = Γ d / (2π (d² + δ²))  (Rosenhead–Moore)
  * anneau régularisé RM :          U = Γ/(4πR) [ln(8R/δ) - 1]

Rappels pytest : chaque fonction « test_* » est exécutée automatiquement par « pytest » ;
« assert condition » fait échouer le test si la condition est fausse ; pytest.approx(v, rel=...)
compare à une tolérance relative près ; np.testing.assert_allclose compare deux tableaux
élément par élément (rtol = tolérance relative, atol = tolérance absolue).
"""

# numpy : tableaux et fonctions de test numériques (np.testing).
import numpy as np
# pytest : cadre de tests (approx, mark.parametrize).
import pytest

# Modules testés.
from src import theory
from src import vortex_particles as vp


def _ring_sim(n=400, delta=1e-3, kernel="segment", **kw):
    # Utilitaire : simulation contenant un seul anneau (R = 1, Γ = 1) à n nœuds ;
    # **kw transmet d'éventuels paramètres supplémentaires à make_ring.
    return vp.VortexFilamentSimulation(vp.make_ring(1.0, 1.0, n, delta=delta, **kw), kernel=kernel)


# ---------------------------------------------------------------------------------------------
# Champs induits par un anneau
# ---------------------------------------------------------------------------------------------
# @pytest.mark.parametrize : le test est exécuté une fois pour chaque valeur de « kernel ».
@pytest.mark.parametrize("kernel", ["segment", "particle"])
def test_ring_center_velocity(kernel):
    # δ très petit : on retrouve le filament singulier (400 nœuds).
    sim = _ring_sim(kernel=kernel)
    # Vitesse au centre de l'anneau ([0] : premier et unique point évalué).
    u = sim.velocity_field([[0.0, 0.0, 0.0]])[0]
    # u_z = Γ/(2R) = 0.5 à 2e-4 près (erreur de discrétisation polygonale).
    assert u[2] == pytest.approx(0.5, rel=2e-4)
    # Composantes transverses nulles par symétrie.
    assert abs(u[0]) < 1e-12 and abs(u[1]) < 1e-12


def test_axial_velocity_profile():
    sim = _ring_sim()
    # 13 points sur l'axe z ∈ [-3, 3].
    z = np.linspace(-3.0, 3.0, 13)
    # np.column_stack : assemble les colonnes (x = 0, y = 0, z) en tableau (13, 3).
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    # Composante axiale de la vitesse calculée.
    uz = sim.velocity_field(pts)[:, 2]
    # Référence analytique Γ R² / (2 (R² + z²)^{3/2}).
    ref = 1.0 / (2.0 * (1.0 + z**2) ** 1.5)
    np.testing.assert_allclose(uz, ref, rtol=2e-4)


def test_off_axis_matches_elliptic_integrals():
    sim = _ring_sim(n=800)
    # Générateur de nombres aléatoires à graine fixe (tests reproductibles).
    rng = np.random.default_rng(0)
    # 20 points tirés uniformément en r, z et angle φ.
    r = rng.uniform(0.1, 2.5, 20)
    z = rng.uniform(-1.5, 1.5, 20)
    phi = rng.uniform(0, 2 * np.pi, 20)
    # Exclusion des points trop proches du filament (où δ ≠ 0 et la discrétisation comptent).
    keep = np.hypot(r - 1.0, z) > 0.25  # hors du voisinage immédiat du filament
    r, z, phi = r[keep], z[keep], phi[keep]
    # Coordonnées cartésiennes des points.
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    u = sim.velocity_field(pts)
    # Projection sur les directions radiale et azimutale locales.
    ur_num = u[:, 0] * np.cos(phi) + u[:, 1] * np.sin(phi)
    ut_num = -u[:, 0] * np.sin(phi) + u[:, 1] * np.cos(phi)
    # Formules exactes (intégrales elliptiques).
    ur, uz = theory.ring_induced_velocity(r, z, R=1.0, Gamma=1.0)
    np.testing.assert_allclose(u[:, 2], uz, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(ur_num, ur, rtol=2e-3, atol=2e-4)
    # Aucune composante azimutale (pas de swirl).
    assert np.abs(ut_num).max() < 1e-10  # pas de swirl


# ---------------------------------------------------------------------------------------------
# Filament rectiligne et régularisation
# ---------------------------------------------------------------------------------------------
def test_infinite_straight_filament_singular_and_regularized():
    # Segment très long (±1e5) selon z : équivalent d'un filament infini.
    Lhalf = 1e5
    a = np.array([[0.0, 0.0, -Lhalf]])
    b = np.array([[0.0, 0.0, Lhalf]])
    # Distances d'évaluation sur l'axe x.
    d = np.array([0.05, 0.3, 1.0, 4.0])
    pts = np.column_stack([d, np.zeros_like(d), np.zeros_like(d)])
    # Noyau singulier (δ = 0) : u_y = Γ/(2πd).
    u0 = vp.biot_savart_segments(pts, a, b, 1.0, 0.0)
    np.testing.assert_allclose(u0[:, 1], 1.0 / (2 * np.pi * d), rtol=1e-6)  # sens direct autour de +z
    # Noyau régularisé : u_y = Γ d / (2π (d² + δ²)).
    delta = 0.2
    ud = vp.biot_savart_segments(pts, a, b, 1.0, delta)
    np.testing.assert_allclose(ud[:, 1], d / (2 * np.pi * (d**2 + delta**2)), rtol=1e-6)


def test_regularized_velocity_is_bounded():
    """Avec δ > 0, la vitesse reste finie jusqu'au filament (max Γ/(4πδ) à la distance δ)."""
    Lhalf = 1e4
    a, b = np.array([[0.0, 0.0, -Lhalf]]), np.array([[0.0, 0.0, Lhalf]])
    delta = 0.1
    # 400 distances de 1e-8 à 10 réparties logarithmiquement (np.geomspace).
    d = np.geomspace(1e-8, 10.0, 400)
    pts = np.column_stack([d, np.zeros_like(d), np.zeros_like(d)])
    u = vp.biot_savart_segments(pts, a, b, 1.0, delta)
    # np.isfinite : aucune valeur infinie ou NaN.
    assert np.all(np.isfinite(u))
    # Le maximum Γ/(4πδ) est atteint à la distance δ.
    assert u[:, 1].max() == pytest.approx(1.0 / (4 * np.pi * delta), rel=1e-3)
    # sur le filament lui-même : vitesse nulle (pas d'auto-induction d'un segment rectiligne)
    on = vp.biot_savart_segments([[0.0, 0.0, 0.3]], a, b, 1.0, delta)
    assert np.linalg.norm(on) < 1e-12
    # particule : l'auto-contribution est nulle aussi (α × 0 = 0)
    up = vp.biot_savart_particles([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], delta)
    assert np.linalg.norm(up) == 0.0


def test_segment_and_particle_kernels_agree():
    """La règle du point milieu (particules) converge vers l'intégration exacte des segments."""
    rng = np.random.default_rng(1)
    # 30 points aléatoires (loi normale centrée réduite en 3D).
    pts = rng.normal(size=(30, 3))
    # Référence : segments exacts avec 128 nœuds ; comparaison : 4096 particules.
    exact = _ring_sim(n=128, delta=0.15, kernel="segment").velocity_field(pts)
    fine = _ring_sim(n=4096, delta=0.15, kernel="particle").velocity_field(pts)
    np.testing.assert_allclose(fine, exact, atol=2e-3 * np.abs(exact).max())


def test_linearity_and_orientation():
    pts = np.array([[0.3, 0.2, 0.1], [1.4, -0.3, 0.5]])
    # Un segment de référence, le même avec Γ doublé, et le même parcouru en sens inverse.
    base = vp.biot_savart_segments(pts, [[0, 0, 0.0]], [[1.0, 0.5, 0.2]], 1.0, 0.05)
    double = vp.biot_savart_segments(pts, [[0, 0, 0.0]], [[1.0, 0.5, 0.2]], 2.0, 0.05)
    reverse = vp.biot_savart_segments(pts, [[1.0, 0.5, 0.2]], [[0, 0, 0.0]], 1.0, 0.05)
    # Linéarité en Γ et changement de signe avec l'orientation.
    np.testing.assert_allclose(double, 2 * base, rtol=1e-12)
    np.testing.assert_allclose(reverse, -base, rtol=1e-12)


def test_induced_field_is_divergence_free():
    sim = _ring_sim(n=256, delta=0.1)
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.5, 1.5, size=(15, 3))
    # Pas des différences finies centrées.
    h = 1e-4
    div = np.zeros(len(pts))
    grad_scale = np.zeros(len(pts))
    # Somme des dérivées ∂u_k/∂x_k (k = x, y, z).
    for k in range(3):
        # Vecteur de décalage h selon la direction k.
        e = np.zeros(3)
        e[k] = h
        up = sim.velocity_field(pts + e)[:, k]
        um = sim.velocity_field(pts - e)[:, k]
        div += (up - um) / (2 * h)
        # Échelle de référence des gradients (pour une tolérance relative).
        grad_scale += np.abs(up - um) / (2 * h)
    # Divergence négligeable devant les gradients (np.maximum évite une échelle trop petite).
    assert np.all(np.abs(div) < 1e-5 * np.maximum(grad_scale, 1.0))


# ---------------------------------------------------------------------------------------------
# Dynamique : vitesse propre, sens de propagation, invariants
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", ["segment", "particle"])
def test_self_induced_speed_rosenhead_moore(kernel):
    delta = 0.1
    sim = _ring_sim(n=512, delta=delta, kernel=kernel)
    # Vitesse axiale de chaque nœud.
    U = sim.velocity()[:, 2]
    # np.ptp (« peak to peak ») = max - min.
    assert np.ptp(U) < 1e-10  # symétrie : tous les nœuds ont la même vitesse
    # forme close exacte  Γ/(2π√(4R²+δ²)) [K(m) - E(m)]  et asymptote  Γ/(4πR)[ln(8R/δ) - 1]
    U_exact = theory.rosenhead_moore_ring_speed(1.0, 1.0, delta, exact=True)
    U_asym = theory.rosenhead_moore_ring_speed(1.0, 1.0, delta)
    assert U.mean() == pytest.approx(U_exact, rel=2e-3)
    assert U_asym == pytest.approx(U_exact, rel=5e-3)


def test_regularized_ring_field_closed_form():
    """Champ d'un anneau RM : champ singulier évalué en z_e = √(z² + δ²) (et u_r × z/z_e)."""
    delta = 0.3
    sim = _ring_sim(n=2048, delta=delta)
    # Points proches du cœur (dont un sur le filament lui-même) dans le plan y = 0.
    pts = np.array([[0.9, 0.0, 0.1], [1.0, 0.0, 0.05], [1.3, 0.0, -0.2], [0.2, 0.0, 0.5], [1.0, 0.0, 0.0]])
    u = sim.velocity_field(pts)
    ur, uz = theory.ring_induced_velocity(pts[:, 0], pts[:, 2], R=1.0, Gamma=1.0, delta=delta)
    np.testing.assert_allclose(u[:, 0], ur, atol=2e-5)
    np.testing.assert_allclose(u[:, 2], uz, atol=2e-5)


def test_rosenhead_moore_delta_reproduces_physical_core_speed():
    """δ = a·exp(C-1) : l'anneau RM a la vitesse de l'anneau à noyau gaussien de rayon a."""
    a = 0.08
    delta = theory.rosenhead_moore_delta(a, "gaussian")
    U_rm = theory.rosenhead_moore_ring_speed(1.0, 1.0, delta)
    # Égalité exacte des deux formules asymptotiques.
    assert U_rm == pytest.approx(theory.saffman_speed(1.0, 1.0, a), rel=1e-12)
    # Anneau construit directement à partir du rayon de cœur physique.
    sim = vp.VortexFilamentSimulation(vp.make_ring(1.0, 1.0, 768, core_radius=a, core_type="gaussian"))
    assert sim.velocity()[:, 2].mean() == pytest.approx(U_rm, rel=5e-3)


def test_ring_moves_along_its_normal():
    # Normale quelconque, normalisée.
    normal = np.array([1.0, -2.0, 0.5])
    normal /= np.linalg.norm(normal)
    fil = vp.make_ring(0.7, 1.3, 128, center=(0.2, 0.1, -0.4), normal=normal, delta=0.07)
    sim = vp.VortexFilamentSimulation(fil)
    # Vitesse moyenne des nœuds (moyenne selon l'axe 0 = sur les nœuds).
    u_mean = sim.velocity().mean(axis=0)
    # Propagation dans le sens +normale...
    assert np.dot(u_mean, normal) > 0
    # ... et sans composante transverse.
    assert np.linalg.norm(u_mean - np.dot(u_mean, normal) * normal) < 1e-8 * np.linalg.norm(u_mean)
    P = sim.impulse()  # P = π R² Γ n
    np.testing.assert_allclose(P, np.pi * 0.7**2 * 1.3 * normal, rtol=1e-3)


def test_invariants_conserved_during_leapfrog():
    sim = vp.VortexFilamentSimulation(vp.leapfrog_scenario(delta=0.12, separation=0.6, n_nodes=64))
    # Impulsion et énergie initiales.
    P0, E0 = sim.impulse(), sim.energy()
    # 60 pas de 0.05 (save_every=60 : une seule sauvegarde intermédiaire).
    sim.run(3.0, 0.05, save_every=60)
    P1, E1 = sim.impulse(), sim.energy()
    assert np.linalg.norm(P1 - P0) < 1e-3 * np.linalg.norm(P0)
    assert abs(E1 - E0) < 2e-3 * abs(E0)


def test_leapfrog_matches_coaxial_ring_model():
    """Enveloppe des rayons (saute-mouton) comparée au modèle réduit d'anneaux minces coaxiaux."""
    delta, sep = 0.1, 0.6
    sim = vp.VortexFilamentSimulation(vp.leapfrog_scenario(delta=delta, separation=sep, n_nodes=96))
    hist = sim.run(6.0, 0.05, save_every=4)
    d = hist.as_arrays()
    # Modèle réduit intégré aux mêmes instants que les sauvegardes.
    model = theory.CoaxialRingsModel([1.0, 1.0], [0.0, sep], 1.0, delta, core="rosenhead_moore")
    res = model.integrate(6.0, n_out=len(d["t"]))
    # Rayon minimal de l'anneau A et maximal de l'anneau B (tolérance absolue 0.02).
    assert d["radius"][:, 0].min() == pytest.approx(res["R"][0].min(), abs=0.02)
    assert d["radius"][:, 1].max() == pytest.approx(res["R"][1].max(), abs=0.02)


def test_remesh_preserves_ring():
    """Le remaillage par spline périodique conserve l'anneau et rapproche le polygone du cercle."""
    sim = _ring_sim(n=100, delta=0.1)
    P_exact = np.pi  # π R² Γ
    P0 = sim.impulse()[2]  # polygone inscrit : π (1 - (2π/N)²/6)
    sim.filaments[0].spacing *= 0.5  # force un raffinement x2
    # remesh() renvoie True si un filament a été rééchantillonné.
    assert sim.remesh()
    assert abs(sim.n_nodes - 200) <= 1
    P1 = sim.impulse()[2]
    # Le polygone raffiné est plus proche du cercle exact.
    assert abs(P1 - P_exact) < abs(P0 - P_exact)
    assert P1 == pytest.approx(P_exact, rel=3e-4)
    # Les nouveaux nœuds restent sur le cercle de rayon 1 (norme des composantes x, y).
    radii = np.linalg.norm(sim.positions()[:, :2], axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=1e-5)
