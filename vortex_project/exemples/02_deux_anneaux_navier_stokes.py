"""
Exemple 2 — Deux anneaux coaxiaux visqueux (saute-mouton), solveur Navier–Stokes axisymétrique.

Modifiez la liste ANNEAUX (rayons, cœurs, circulations, positions) et les réglages numériques, puis lancez :
    .venv\\Scripts\\python exemples\\02_deux_anneaux_navier_stokes.py
Résultats : exemples/resultats/02_deux_anneaux/ (Hovmöller, bilan d'énergie, isosurfaces 3D, portrait de phase).
Théorie associée : docs/theoretical_background.pdf, § 2.4 (anneaux coaxiaux) et § 5.4 (saute-mouton visqueux).

Remarque : le suivi individuel des cœurs repère les maxima de vorticité positive ; il suppose des circulations
positives (anneaux de même sens). Une collision frontale (Γ de signes opposés) se simule, mais seuls les champs
et les bilans globaux sont alors exploitables.
"""

# Chemins et accès au paquet src/ (dossier du projet ajouté au chemin de recherche des modules).
import sys
from pathlib import Path

PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJET))

import numpy as np  # noqa: E402
from src import solver_2d_axisymmetric as ax  # noqa: E402
from src import visualization as viz  # noqa: E402
from src import visualization_3d as v3  # noqa: E402

# =============================================================================================
# PARAMÈTRES À MODIFIER
# =============================================================================================
# Un dictionnaire par anneau : rayon R, rayon de cœur gaussien a, circulation Gamma, position axiale z0.
ANNEAUX = [
    {"R": 1.0, "a": 0.15, "Gamma": 1.0, "z0": 2.0},   # anneau arrière
    {"R": 1.0, "a": 0.15, "Gamma": 1.0, "z0": 2.6},   # anneau avant (écart 0.6)
]
RE = 2000.0      # Re_Γ = max|Γ| / ν
T_END = 6.0      # durée simulée
DR = 0.03        # pas de maillage : garder a/DR ≥ 5 (0.02 pour a = 0.10)
SORTIE = PROJET / "exemples" / "resultats" / "02_deux_anneaux"

# =============================================================================================
# Simulation
# =============================================================================================
print(f"{len(ANNEAUX)} anneaux, Re_Γ = {RE:g}, t_end = {T_END}, Δr = {DR}")
# run_coaxial_rings : superpose les anneaux de Lamb–Oseen, intègre, enregistre les profils du Hovmöller et suit
# chaque cœur (moments de Saffman par partie du domaine).
res = ax.run_coaxial_rings(ANNEAUX, Re=RE, t_end=T_END, dr=DR, snapshot_times=(0.0, 0.5 * T_END, T_END), verbose=True)
rg = res["rings"]
print(f"Grille : {res['grid']} — calcul : {res['wall_time']:.1f} s")

# Fusion des cœurs : premier instant où un seul pic subsiste pendant 3 diagnostics consécutifs ; les courbes de
# suivi sont tronquées à cet instant (au-delà, les deux anneaux n'en forment plus qu'un).
seul = rg["n_peaks"] < len(ANNEAUX)
i_fusion = next((i for i in range(len(seul) - 2) if seul[i] and seul[i + 1] and seul[i + 2]), None)
garde = rg["t"] < rg["t"][i_fusion] if i_fusion is not None else np.ones_like(rg["t"], dtype=bool)
if i_fusion is not None:
    print(f"Fusion des cœurs vers t ≈ {rg['t'][i_fusion]:.1f}")
# Instants où l'écart axial z1 − z2 change de signe (un anneau traverse l'autre).
dz = rg["Z"][garde, 0] - rg["Z"][garde, 1]
k = np.nonzero(np.sign(dz[1:]) != np.sign(dz[:-1]))[0]
print("Traversées : " + (", ".join(f"t ≈ {rg['t'][garde][i]:.2f}" for i in k) or "aucune"))
d = res["diagnostics"]
print(f"Énergie perdue : {100 * (1 - d['energy'][-1] / d['energy'][0]):.1f} %")

# =============================================================================================
# Figures
# =============================================================================================
# Trajectoires suivies superposées aux diagrammes de Hovmöller.
traj = [(rg["t"][garde], rg["Z"][garde, i], f"anneau {i + 1}") for i in range(len(ANNEAUX))]
viz.save_figure(viz.plot_hovmoller(res["profiles"], tracks=traj), SORTIE / "hovmoller.png")
viz.save_figure(viz.plot_energy_enstrophy_budget(res), SORTIE / "bilan_energie.png")
# Isosurfaces Q reconstruites par révolution, centrées sur le centroïde de vorticité global.
Zc = [float(np.interp(s["t"], d["t"], d["Z"])) for s in res["snapshots"]]
viz.save_figure(v3.plot_isosurface_sequence(res["snapshots"], Zc, ncols=3), SORTIE / "isosurfaces.png")
# Portrait de phase (R1 − R2, z1 − z2) : orbite fermée en fluide parfait, spirale en visqueux.
if len(ANNEAUX) == 2:
    courbe = [{"label": f"Navier–Stokes, Re_Γ = {RE:g}", "t": rg["t"][garde], "R": rg["R"][garde], "Z": rg["Z"][garde]}]
    viz.save_figure(viz.plot_phase_diagram(courbe), SORTIE / "portrait_de_phase.png")
print(f"Figures enregistrées dans {SORTIE}")
