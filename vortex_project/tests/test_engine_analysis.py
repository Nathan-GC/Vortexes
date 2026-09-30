"""
Tests du module d'analyse des moteurs à vortex (``src/engine_analysis.py``).
"""

# numpy et pytest (voir les rappels en tête de test_biot_savart.py).
import numpy as np
import pytest

# Module testé.
from src import engine_analysis as ea


# Fixture : chambre d'exemple construite une seule fois pour tout le fichier (scope="module").
@pytest.fixture(scope="module")
def chamber():
    return ea.BidirectionalVortexChamber()


def test_mean_residence_time_equals_volume_over_flow(chamber):
    rtd = chamber.residence_time_distribution()
    # Théorème de Danckwerts : temps de séjour moyen = V/Q.
    assert rtd["mean"] == pytest.approx(chamber.volume / chamber.Q, rel=1e-6)
    # Les poids de débit sont normalisés.
    assert np.sum(rtd["weight"]) == pytest.approx(1.0)
    # La moyenne discrète pondérée retrouve la moyenne exacte.
    assert np.sum(rtd["weight"] * rtd["t"]) == pytest.approx(rtd["mean"], rel=2e-3)


def test_trajectory_follows_streamline_and_closed_form_time(chamber):
    # Particule injectée à r0 = 0.88 a.
    tr = chamber.trajectory(0.88 * chamber.a)
    # ψ conservé le long de la trajectoire (écoulement stationnaire).
    assert tr["psi_drift"] < 1e-6
    # Durée d'intégration = temps de séjour en forme close.
    assert tr["t"][-1] == pytest.approx(tr["residence_time"], rel=1e-4)
    # Rayon de sortie = √(a² - r0²).
    assert tr["r"][-1] == pytest.approx(tr["r_exit"], rel=1e-4)
    # Profondeur de pénétration z_min = L sin(π r̃0²).
    assert tr["z_min"] == pytest.approx(chamber.L * np.sin(np.pi * 0.88**2), rel=1e-3)
    # Trajectoire hélicoïdale : plus d'un tour et parcours plus long que la chambre.
    assert tr["revolutions"] > 1 and tr["path_ratio"] > 1


def test_uniform_cross_mantle_feeding(chamber):
    # Débit à travers le manteau par unité de longueur = Q/L.
    assert chamber.cross_mantle_flux_per_length() == pytest.approx(chamber.Q / chamber.L, rel=1e-10)
    # Vitesse axiale nulle sur le manteau (« _ » ignore u_r et u_θ).
    _, _, uz = chamber.velocity(chamber.mantle_radius, 0.1)
    assert abs(uz) < 1e-10


def test_cold_wall_model():
    k = 0.03
    # Sans convection (Pe = 0) : flux de conduction pure, rapport 1.
    assert ea.cold_wall_heat_flux(0.0, k)["ratio"] == pytest.approx(1.0)
    # Le rapport décroît quand le Péclet augmente (np.diff < 0 partout).
    ratios = [ea.cold_wall_heat_flux(pe, k)["ratio"] for pe in (10, 100, 300, 1000)]
    assert np.all(np.diff(ratios) < 0)
    # Pour Pe·κ = 40, l'asymptote de Laplace est précise à 5 % près.
    res = ea.cold_wall_heat_flux(40.0 / k, k)
    assert res["ratio_asymptotic"] == pytest.approx(res["ratio"], rel=0.05)
    # Conditions aux limites du profil de température : 1 à la flamme, 0 à la paroi.
    assert res["theta"][0] == pytest.approx(1.0) and res["theta"][-1] == pytest.approx(0.0, abs=1e-12)


def test_reactor_conversion_ordering(chamber):
    rc = ea.reactor_conversion([0.3, 1.0, 3.0], chamber)
    # Réacteur agité ≤ vortex ≤ piston (ordre attendu pour une réaction d'ordre 1).
    assert np.all(rc["cstr"] <= rc["vortex"] + 1e-12) and np.all(rc["vortex"] <= rc["plug"] + 1e-12)


def test_spheroid_added_mass_limits():
    # Sphère exacte, puis sphéroïdes presque sphériques (continuité des deux formules).
    assert ea.spheroid_added_mass(1.0, 1.0) == pytest.approx(0.5)
    assert ea.spheroid_added_mass(1.0 + 1e-3, 1.0) == pytest.approx(0.5, abs=1e-3)
    assert ea.spheroid_added_mass(1.0 - 1e-3, 1.0) == pytest.approx(0.5, abs=1e-3)
    c = 1e-3  # disque : masse ajoutée 8ρa³/3
    # C_A × volume du sphéroïde très aplati → masse ajoutée du disque.
    assert ea.spheroid_added_mass(c, 1.0) * 4.0 / 3.0 * np.pi * c == pytest.approx(8.0 / 3.0, rel=5e-3)
    assert ea.spheroid_added_mass(20.0, 1.0) < 0.02  # corps élancé


def test_vortex_bubble_of_hill_like_ring():
    """Bulle d'un anneau mince : points d'arrêt sur l'axe conformes à u_axe(z_s) = U."""
    # Anneau Γ = 1, R = 1, translation à U = 0.25.
    b = ea.vortex_bubble(1.0, 1.0, 0.25)
    # Solution de Γ R²/(2 (R² + z²)^{3/2}) = U : z_s = √((Γ R²/(2U))^{2/3} - R²).
    z_s = np.sqrt((1.0 / 0.5) ** (2 / 3) - 1.0)
    assert b["attached_to_axis"] and b["z_stagnation"] == pytest.approx(z_s)
    # Le demi-axe mesuré sur la grille retrouve le point d'arrêt.
    assert b["half_axial"] == pytest.approx(z_s, abs=0.02)
    # Volume de bulle supérieur à celui d'un sphéroïde de demi-axe z_s (borne grossière).
    assert b["volume"] > 4 / 3 * np.pi * z_s**3 * 0.5


def test_thruster_formation_and_overpressure():
    thr = ea.VortexRingThruster()
    fs = thr.formation_number()
    ld = np.linspace(0.5, 8, 301)
    beta = thr.overpressure_fraction(ld)
    # Le maximum de la surpression est atteint au nombre de formation...
    assert ld[np.argmax(beta)] == pytest.approx(fs, abs=0.03)
    # ... et vaut β_max.
    assert beta.max() == pytest.approx(thr.beta_max, rel=1e-3)
    # Le co-écoulement réduit le nombre de formation.
    assert thr.formation_number(0.5) < fs
    # Anneau de tête pour L/D = 2 : R = D/√2, Γ = U_j L/2, bouclage d'impulsion proche de 1.
    ring = thr.leading_ring(2.0, 1.0)
    assert ring["radius"] == pytest.approx(thr.D / np.sqrt(2))
    assert ring["circulation"] == pytest.approx(0.5 * 1.0 * 2.0 * thr.D)
    assert 0.8 < ring["impulse_closure"] < 1.2


def test_thruster_steady_jet_limit_and_thrust_solver():
    # Sans surpression (β_max = 0), le rendement est celui de Froude d'un jet continu.
    thr = ea.VortexRingThruster(beta_max=0.0)
    p = thr.performance(3.0, 5.0, 2.0, U_inf=0.5)
    assert p["efficiency"] == pytest.approx(2.0 / (1.0 + 2.0 / 0.5))  # Froude
    assert p["gain"] == pytest.approx(1.0)
    # Le solveur de vitesse de jet produit exactement la poussée demandée.
    thr = ea.VortexRingThruster()
    T = 0.3
    Uj = thr.jet_velocity_for_thrust(4.0, 3.0, 0.5, T)
    assert thr.performance(4.0, 3.0, Uj, 0.5)["thrust"] == pytest.approx(T, rel=1e-8)
    # La formation d'anneaux améliore le rendement par rapport au jet continu équivalent.
    assert thr.performance(4.0, 3.0, Uj, 0.5)["gain"] > 1.0
    # Fréquence trop élevée (rapport cyclique > 1) : point infaisable (NaN).
    assert np.isnan(thr.jet_velocity_for_thrust(8.0, 200.0, 0.5, T))  # rapport cyclique > 1
