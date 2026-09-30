"""
Exemple 4 — Moteurs à vortex : chambre à vortex bidirectionnel et propulseur pulsé (unités SI).

Modifiez les dimensions de la chambre et du propulseur, puis lancez :
    .venv\\Scripts\\python exemples\\04_moteurs.py
Résultats : exemples/resultats/04_moteurs/ (chambre, carte de rendement) et chiffres clés dans la console.
Théorie associée : docs/vortex_engines_analysis.pdf (§ 1 chambres, § 2 propulsion pulsée).
"""

# Chemins et accès au paquet src/.
import sys
from pathlib import Path

PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJET))

import numpy as np  # noqa: E402
from src import engine_analysis as ea  # noqa: E402
from src import visualization as viz  # noqa: E402

# =============================================================================================
# PARAMÈTRES À MODIFIER (unités SI)
# =============================================================================================
# Chambre à vortex bidirectionnel (solution exacte de Vyas & Majdalani).
RAYON = 0.05          # rayon de la chambre a [m]
LONGUEUR = 0.15       # longueur L [m]
U_INJECTION = 40.0    # vitesse tangentielle d'injection à la paroi U [m/s]
DEBIT = 0.05          # débit volumique injecté Q [m³/s]
RHO_GAZ = 20.0        # masse volumique du gaz [kg/m³] (champ de pression)
# Propulseur à anneaux tourbillonnaires (dans l'eau).
D_TUYERE = 0.02       # diamètre de sortie de la tuyère [m]
U_CROISIERE = 0.5     # vitesse du véhicule [m/s]
D_VEHICULE = 0.1      # diamètre du véhicule [m] (calcul de la traînée = poussée requise)
CX = 0.3              # coefficient de traînée du véhicule
SORTIE = PROJET / "exemples" / "resultats" / "04_moteurs"

# =============================================================================================
# Chambre à vortex
# =============================================================================================
ch = ea.BidirectionalVortexChamber(a=RAYON, L=LONGUEUR, U=U_INJECTION, Q=DEBIT, rho=RHO_GAZ)
# κ = Q/(2π U a L) : rapport débit axial / swirl, qui fixe toute la cinématique (modèle valable pour κ ≪ 1).
print(f"Chambre : κ = {ch.kappa:.4f}")
# Distribution des temps de séjour (moyenne = V/Q) et trois trajectoires injectées à 0.75a, 0.85a, 0.95a.
rtd = ch.residence_time_distribution()
traj = [ch.trajectory(f * ch.a, theta0=k * 2.1) for k, f in enumerate((0.75, 0.85, 0.95))]
print(f"   temps de séjour moyen : {rtd['mean'] * 1e3:.2f} ms (≈ {rtd['n_tanks']:.1f} réacteurs agités en série)")
for tr in traj:
    print(f"   injection à r0 = {tr['r'][0] / ch.a:.2f} a : {tr['residence_time'] * 1e3:.1f} ms, "
          f"{tr['revolutions']:.1f} tours, parcours {tr['path_ratio']:.1f} L")
# Chute de pression paroi → axe, nombre de swirl en sortie, diamètre de coupure des gouttes.
print(f"   chute de pression paroi → axe : {ch.pressure_drop(np.array([0.0]))[0] / 1e5:.2f} bar ; "
      f"swirl en sortie : {ch.exit_swirl_number():.2f} ; gouttes piégées si d > {ch.droplet_cutoff_diameter() * 1e6:.1f} µm")
viz.save_figure(viz.plot_bidirectional_vortex(ch, traj, rtd), SORTIE / "chambre.png")

# =============================================================================================
# Propulseur pulsé
# =============================================================================================
thr = ea.VortexRingThruster(D=D_TUYERE)
# Poussée requise = traînée du véhicule à la vitesse de croisière.
poussee = ea.body_drag(U_CROISIERE, diameter=D_VEHICULE, cd=CX)
print(f"Propulseur : F* = {thr.formation_number():.2f} ; poussée requise = {poussee:.3f} N")
# Carte du rendement propulsif η en fonction du rapport d'aspect L/D et de la fréquence d'éjection f.
carte = thr.efficiency_map(np.linspace(0.5, 8.0, 24), np.geomspace(0.5, 40.0, 24), U_CROISIERE, poussee)
carte["formation_number"] = thr.formation_number()
# Meilleur rendement (np.nanargmax ignore les combinaisons infaisables, notées NaN).
i, j = np.unravel_index(np.nanargmax(carte["efficiency"]), carte["efficiency"].shape)
print(f"   rendement maximal η = {carte['efficiency'][i, j]:.3f} à L/D = {carte['L_over_D'][i]:.2f}, "
      f"f = {carte['frequency'][j]:.1f} Hz")
viz.save_figure(viz.plot_efficiency_map(carte), SORTIE / "rendement_propulseur.png")
print(f"Figures enregistrées dans {SORTIE}")
