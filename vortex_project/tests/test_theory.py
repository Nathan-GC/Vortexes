"""
Tests des modèles analytiques (``src/theory.py``).
"""

# numpy et pytest (voir les rappels en tête de test_biot_savart.py).
import numpy as np
import pytest

# Module testé.
from src import theory


def test_core_constants_from_profiles():
    # Noyau uniforme (indicatrice de s < 1, via np.where) : C = 1/4.
    assert theory.core_constant_from_profile(lambda s: np.where(s < 1.0, 1.0, 0.0), n=400001) == pytest.approx(0.25, abs=2e-4)
    # Noyau gaussien exp(-s²) : C = 1/2 + (ln 2 - γ_E)/2 ≈ 0.558.
    assert theory.core_constant_from_profile(lambda s: np.exp(-s**2)) == pytest.approx(theory.GAUSSIAN_CORE_CONSTANT, abs=2e-4)
    # vorticité 2D équivalente au noyau de Rosenhead–Moore : ω ∝ δ²/(s² + δ²)²  →  C = 1
    # (profil à décroissance algébrique : borne d'intégration et nombre de points élevés).
    assert theory.core_constant_from_profile(lambda s: 1.0 / (1.0 + s**2) ** 2, s_max=4000.0,
                                             n=4000001) == pytest.approx(1.0, abs=2e-3)


def test_kelvin_benjamin_consistency():
    """U = ∂E/∂I à Γ et volume de cœur (a² R) fixés."""
    Gamma, R, a = 1.0, 1.0, 0.1
    # Pas de la dérivée numérique centrée.
    h = 1e-6
    Rp, Rm = R + h, R - h
    # Rayons de cœur correspondants à volume de tore constant (a² R = cste).
    ap, am = a * np.sqrt(R / Rp), a * np.sqrt(R / Rm)
    # Variations d'énergie et d'impulsion.
    dE = theory.ring_energy(Gamma, Rp, ap) - theory.ring_energy(Gamma, Rm, am)
    dI = theory.ring_impulse(Gamma, Rp) - theory.ring_impulse(Gamma, Rm)
    # dE/dI doit égaler la vitesse de Kelvin (relation hamiltonienne).
    assert dE / dI == pytest.approx(theory.ring_speed(Gamma, R, a, "uniform"), rel=1e-6)


def test_hill_vortex_invariants_and_fields():
    inv = theory.hill_invariants(1.0, 1.0)
    # Γ = 5Ua, masse ajoutée 1/2, énergie adimensionnelle ≈ 0.160.
    assert inv["circulation"] == pytest.approx(5.0)
    assert inv["added_mass_coefficient"] == pytest.approx(0.5)
    assert inv["dimensionless_energy"] == pytest.approx(0.1601, abs=2e-4)
    # continuité de la vitesse et de la pression sur la sphère
    # Points juste à l'intérieur (1 - 1e-10) et juste à l'extérieur (1 + 1e-10) de la sphère.
    th = np.linspace(0.1, np.pi - 0.1, 9)
    ri, zi = (1 - 1e-10) * np.sin(th), (1 - 1e-10) * np.cos(th)
    ro, zo = (1 + 1e-10) * np.sin(th), (1 + 1e-10) * np.cos(th)
    np.testing.assert_allclose(theory.hill_velocity(ri, zi), theory.hill_velocity(ro, zo), atol=1e-8)
    np.testing.assert_allclose(theory.hill_pressure(ri, zi), theory.hill_pressure(ro, zo), atol=1e-8)
    # Pression minimale au centre du cœur : -29/32 ρU².
    assert theory.hill_pressure(1 / np.sqrt(2), 0.0) == pytest.approx(-29 / 32)
    # vorticité = -(1/r) E²ψ (différences finies)
    r, z, h = 0.5, 0.2, 1e-4
    # Fonction de courant en fonction de (r, z) (lambda = fonction anonyme).
    psi = lambda rr, zz: theory.hill_stream_function(rr, zz)
    # E²ψ = ψ_rr - ψ_r / r + ψ_zz par différences finies centrées.
    e2 = ((psi(r + h, z) - 2 * psi(r, z) + psi(r - h, z)) / h**2 - (psi(r + h, z) - psi(r - h, z)) / (2 * h * r)
          + (psi(r, z + h) - 2 * psi(r, z) + psi(r, z - h)) / h**2)
    assert -e2 / r == pytest.approx(float(theory.hill_vorticity(r, z)), rel=1e-5)


def test_ring_stream_function_matches_velocity():
    # Quelques points du plan méridien.
    r = np.array([0.3, 0.8, 1.3, 2.0])
    z = np.array([0.4, -0.7, 0.2, 1.5])
    h = 1e-6
    psi = lambda rr, zz: theory.ring_stream_function(rr, zz)
    # u_z = (1/r) ∂ψ/∂r et u_r = -(1/r) ∂ψ/∂z par différences finies centrées.
    uz = (psi(r + h, z) - psi(r - h, z)) / (2 * h * r)
    ur = -(psi(r, z + h) - psi(r, z - h)) / (2 * h * r)
    # Comparaison avec les formules directes de la vitesse.
    ur_a, uz_a = theory.ring_induced_velocity(r, z)
    np.testing.assert_allclose(uz, uz_a, rtol=1e-6)
    np.testing.assert_allclose(ur, ur_a, rtol=1e-6)
    # champ lointain dipolaire : ψ ≈ Γ R² r² / (4 ρ³)
    # Point à la distance ρ = √(30² + 40²) = 50.
    rr, zz = 30.0, 40.0
    assert theory.ring_stream_function(rr, zz) == pytest.approx(rr**2 / (4 * 50.0**3), rel=2e-3)


def test_formation_number_and_slug_model():
    # F* = √(π/2)/0.33 ≈ 3.798.
    assert theory.formation_number(0.33) == pytest.approx(3.798, abs=1e-3)
    # Énergie adimensionnelle calculée à partir du dictionnaire slug (D = 1, L = 4, U_p = 1)...
    s = theory.slug_model(1.0, 4.0, 1.0)
    alpha = theory.dimensionless_energy(s["energy"], s["impulse"], s["circulation"])
    # ... identique à la formule fermée √(π/2)/(L/D).
    assert alpha == pytest.approx(theory.slug_dimensionless_energy(4.0))


def test_saffman_formula_limits():
    # La formule temporelle (σ² = σ0² + 4νt) coïncide avec la formule en σ évaluée pour ce σ.
    assert theory.saffman_viscous_speed(1.0, 1.0, 1e-3, 2.0, sigma0=0.1) == pytest.approx(
        theory.saffman_speed(1.0, 1.0, np.sqrt(0.01 + 4e-3 * 2.0)))
    # la correction de Fukumoto–Moffatt réduit la vitesse
    assert theory.saffman_speed(1, 1, 0.3, second_order=True) < theory.saffman_speed(1, 1, 0.3)


def test_coaxial_model_single_ring_and_impulse_invariant():
    # Un seul anneau : translation uniforme à la vitesse de Kelvin.
    m = theory.CoaxialRingsModel([1.0], [0.0], 1.0, 0.1, core="uniform")
    res = m.integrate(5.0, 11)
    # Position finale (anneau 0, dernier instant) = 5 × vitesse.
    assert res["Z"][0, -1] == pytest.approx(5.0 * theory.ring_speed(1.0, 1.0, 0.1, "uniform"), rel=1e-8)
    # l'impulsion totale Σ π Γ_i R_i² est un invariant du système coaxial
    m2 = theory.CoaxialRingsModel([1.0, 0.8], [0.0, 0.5], [1.0, 1.3], 0.1)
    res2 = m2.integrate(10.0, 201)
    # Impulsion à chaque instant (somme sur les anneaux, axe 0).
    P = np.pi * (m2.Gamma[:, None] * res2["R"] ** 2).sum(axis=0)
    # Variation totale (np.ptp = max - min) négligeable.
    assert np.ptp(P) < 1e-7 * P[0]
