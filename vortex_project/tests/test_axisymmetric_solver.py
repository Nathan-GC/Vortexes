"""
Tests du solveur Navier–Stokes axisymétrique (``src/solver_2d_axisymmetric.py``).
"""

# numpy et pytest (voir les rappels en tête de test_biot_savart.py).
import numpy as np
import pytest
# trapezoid : intégrale par la méthode des trapèzes (bilan d'énergie).
from scipy.integrate import trapezoid

# Modules testés.
from src import theory
from src import solver_2d_axisymmetric as ax


# @pytest.fixture : fonction qui fabrique un objet partagé par les tests qui la demandent en
# argument ; scope="module" → l'objet n'est construit qu'une fois pour tout ce fichier.
@pytest.fixture(scope="module")
def ring_solver():
    # Grille étirée : Δr = 0.05 jusqu'à r = 2, croissance 1.06 jusqu'à r = 6 ; Lz = 8, Nz = 160.
    grid = ax.AxisymmetricGrid.stretched(0.05, 2.0, 6.0, 1.06, 8.0, 160)
    s = ax.AxisymmetricNavierStokesSolver(grid, nu=1e-3)
    # Anneau gaussien de rayon 1 et de cœur 0.25 centré en z = 2.
    s.init_lamb_oseen_ring(R=1.0, a=0.25, Gamma=1.0, z0=2.0)
    return s


def test_stretched_grid_geometry():
    g = ax.AxisymmetricGrid.stretched(0.05, 2.0, 6.0, 1.06, 8.0, 160)
    # Première face sur l'axe, dernière exactement sur la paroi.
    assert g.rf[0] == 0.0 and g.rf[-1] == pytest.approx(6.0)
    # Zone uniforme : les 40 premières cellules ont la largeur 0.05 (np.allclose : égalité approchée).
    assert np.allclose(g.drc[: int(round(2.0 / 0.05))], 0.05)
    # Largeurs croissantes (np.diff = différences successives).
    assert np.all(np.diff(g.drc) >= -1e-12)  # étirement monotone
    # Les volumes duaux recouvrent exactement [0, L_r].
    assert g.drf[1:-1].sum() + g.drf[0] + g.drf[-1] == pytest.approx(6.0)


def test_initial_field_is_solenoidal_and_reproduces_vorticity(ring_solver):
    # « ring_solver » est fourni automatiquement par la fixture de même nom.
    s = ring_solver
    # Divergence nulle à la précision machine.
    assert np.abs(s.divergence(s.u, s.w)).max() < 1e-12
    # Vorticité de référence aux coins (ω nulle sur l'axe et la paroi).
    Rg, Zg = s.g.corner_mesh()
    om_ref = theory.lamb_oseen_ring_vorticity(Rg, s.g.periodic_dz(Zg, 2.0), 1.0, 0.25, 1.0, 0.0)
    om_ref[0] = om_ref[-1] = 0.0
    # Le rotationnel discret du champ initial reproduit la vorticité imposée.
    np.testing.assert_allclose(s.vorticity(), om_ref, atol=1e-9 * om_ref.max())
    d = s.diagnostics()
    assert d["circulation"] == pytest.approx(1.0, rel=1e-6)
    # R² = ∫ω r² / ∫ω = R0² + a²/2 pour un cœur gaussien
    assert d["R"] == pytest.approx(np.sqrt(1.0 + 0.25**2 / 2), rel=1e-4)
    # Rayon de cœur mesuré ≈ a.
    assert d["sigma"] == pytest.approx(0.25, rel=0.03)


def test_stream_function_consistency(ring_solver):
    """ψ intégré depuis w coïncide avec la solution de E²ψ = -rω (paroi : ψ = 0)."""
    s = ring_solver
    # ψ par intégration radiale de w, puis par résolution de l'équation de Stokes–Poisson.
    psi_int = s.stream_function()
    psi_sol = s.solve_stream_function(s.vorticity())
    np.testing.assert_allclose(psi_int, psi_sol, atol=1e-10 * np.abs(psi_sol).max())


def test_projection_is_exact_and_idempotent(ring_solver):
    s = ring_solver
    rng = np.random.default_rng(3)
    # Champ aléatoire (u nul sur l'axe et la paroi, conformément aux conditions aux limites).
    fu = rng.normal(size=s.u.shape)
    fu[0] = fu[-1] = 0.0
    fw = rng.normal(size=s.w.shape)
    # Projection : le résultat doit être à divergence nulle...
    pu, pw, _ = s.project(fu, fw)
    assert np.abs(s.divergence(pu, pw)).max() < 1e-10
    # ... et projeter une seconde fois ne doit rien changer (P² = P).
    pu2, pw2, _ = s.project(pu, pw)
    assert np.abs(pu2 - pu).max() < 1e-11 and np.abs(pw2 - pw).max() < 1e-11


def test_hill_vortex_initial_velocity():
    """Le solveur de Stokes–Poisson reproduit le champ analytique du vortex de Hill."""
    # Grille fine près du vortex, paroi repoussée à r = 10.
    g = ax.AxisymmetricGrid.stretched(0.025, 1.6, 10.0, 1.05, 10.0, 400)
    s = ax.AxisymmetricNavierStokesSolver(g, nu=0.0)
    s.init_hill_vortex(a=1.0, U=1.0, z0=5.0)
    # Champs aux centres des cellules.
    f = s.fields()
    Rc, Zc = g.center_mesh()
    # Champ analytique dans le repère du laboratoire.
    ur, uz = theory.hill_velocity(Rc, Zc - 5.0, 1.0, 1.0, frame="lab")
    # Masque : points éloignés du saut de vorticité (|ρ - a| > 0.1) et proches du vortex.
    far = np.abs(np.hypot(Rc, Zc - 5.0) - 1.0) > 0.1  # loin de la discontinuité de vorticité
    win = (Rc < 1.8) & (np.abs(Zc - 5.0) < 1.8) & far
    # erreur O(h) localisée près du saut de vorticité (ρ = a), faible en moyenne quadratique
    assert np.abs(f["u_z"] - uz)[win].max() < 0.035
    assert np.abs(f["u_r"] - ur)[win].max() < 0.035
    # Erreur quadratique moyenne (np.sqrt(np.mean(...²))).
    assert np.sqrt(np.mean((f["u_z"] - uz)[win] ** 2)) < 0.01
    # Vitesse au centre du vortex (première ligne = cellule la plus proche de l'axe).
    assert f["u_z"][0, np.argmin(np.abs(g.zc - 5.0))] == pytest.approx(2.5, rel=0.01)  # 5U/2 au centre


def test_inviscid_invariants_short_run():
    grid = ax.AxisymmetricGrid.stretched(0.05, 2.0, 6.0, 1.06, 8.0, 160)
    # ν = 0 : Euler (circulation, impulsion et énergie doivent être conservées).
    s = ax.AxisymmetricNavierStokesSolver(grid, nu=0.0)
    s.init_lamb_oseen_ring(R=1.0, a=0.3, Gamma=1.0, z0=2.0)
    d0 = s.diagnostics()
    res = s.run(2.0, diag_interval=0.5)
    d1 = s.diagnostics()
    # Γ_h = Δz Σ (w[0] - w[paroi]) est évalué à r = Δr/2 : dérive infime (~1e-6)
    assert d1["circulation"] == pytest.approx(d0["circulation"], rel=1e-5)
    assert d1["impulse"] == pytest.approx(d0["impulse"], rel=2e-3)
    assert d1["energy"] == pytest.approx(d0["energy"], rel=2e-3)
    # Divergence restée nulle tout au long du calcul.
    assert res["diagnostics"]["max_div"].max() < 1e-10


def test_energy_budget_and_saffman_speed():
    """Bilan d'énergie exact (dE/dt = -ν∫ω²) et vitesse proche de Saffman + Fukumoto–Moffatt."""
    # Simulation courte et grossière (quelques secondes).
    res = ax.run_vortex_ring(R=1.0, a=0.25, Gamma=1.0, Re=800.0, t_end=6.0, dr=0.05, Lr=6.0, r_uniform=2.0,
                             growth=1.06, Lz=8.0, z0=2.0)
    d = res["diagnostics"]
    # Énergie perdue = énergie dissipée par viscosité (aucune dissipation numérique).
    lost = d["energy"][0] - d["energy"][-1]
    assert lost == pytest.approx(trapezoid(d["dissipation"], d["t"]), rel=2e-3)
    # Diagnostic le plus proche de t = 5.
    k = np.argmin(np.abs(d["t"] - 5.0))
    assert res["U_sim"][k] == pytest.approx(res["U_saffman_fm"][k], rel=0.03)
    assert res["U_sim"][k] < res["U_saffman"][k]  # Saffman surestime la vitesse d'un cœur épais


def test_suivi_des_anneaux_de_circulations_opposees():
    """Collision frontale : le suivi « signé » identifie les deux anneaux (Γ > 0 et Γ < 0), qui se rapprochent
    symétriquement et s'élargissent en approchant du plan de symétrie."""
    # Deux anneaux identiques de sens opposés (z0 = 3 et 5), grille grossière : calcul de quelques secondes.
    rings = [{"R": 1.0, "a": 0.2, "Gamma": 1.0, "z0": 3.0}, {"R": 1.0, "a": 0.2, "Gamma": -1.0, "z0": 5.0}]
    res = ax.run_coaxial_rings(rings, Re=1000.0, t_end=3.0, dr=0.06, diag_interval=0.1)
    rg = res["rings"]
    # Deux pics distincts tout au long du calcul ; circulations de signes opposés et quasi conservées.
    assert rg["n_peaks"].min() == 2
    assert np.all(rg["Gamma"][:, 0] > 0.9) and np.all(rg["Gamma"][:, 1] < -0.9)
    # Rapprochement : l'écart axial diminue ; symétrie : le milieu des deux anneaux reste en z = 4.
    ecart = rg["Z"][:, 1] - rg["Z"][:, 0]
    assert ecart[-1] < ecart[0] - 0.8
    assert np.allclose(0.5 * (rg["Z"][:, 0] + rg["Z"][:, 1]), 4.0, atol=0.02)
    # Élargissement des deux anneaux (méthode des images), de façon identique.
    assert rg["R"][-1, 0] > rg["R"][0, 0] + 0.1
    assert rg["R"][-1, 0] == pytest.approx(rg["R"][-1, 1], rel=1e-2)


def test_vortex_de_hill_visqueux():
    """run_hill_vortex : le vortex démarre à la vitesse de Hill U, puis la viscosité le ralentit lentement ;
    la divergence reste nulle et l'énergie décroît selon le bilan visqueux."""
    res = ax.run_hill_vortex(a=1.0, U=1.0, Re=5000.0, t_end=2.0, dr=0.05)
    d = res["diagnostics"]
    # Vitesse initiale = U (à 1 % près), puis ralentissement modéré.
    assert res["U_sim"][1] == pytest.approx(1.0, rel=0.01)
    assert 0.9 < res["U_sim"][-1] < res["U_sim"][1]
    # Circulation 5Ua et incompressibilité.
    assert d["circulation"][0] == pytest.approx(5.0, rel=0.02)
    assert d["max_div"].max() < 1e-10
    # Énergie perdue = énergie dissipée par viscosité.
    assert d["energy"][0] - d["energy"][-1] == pytest.approx(trapezoid(d["dissipation"], d["t"]), rel=5e-3)
