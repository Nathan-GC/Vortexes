"""
Exemple 3 — Anneaux en fluide parfait par la méthode lagrangienne 3D (filaments tourbillonnaires).

Choisissez un SCÉNARIO et ses paramètres, puis lancez :
    .venv\\Scripts\\python exemples\\03_filaments_3d.py
Résultats : exemples/resultats/03_filaments/ (instantanés 3D, tubes de vorticité, animation GIF).
Théorie associée : docs/theoretical_background.pdf, § 1.5–1.6 (Biot–Savart, Rosenhead–Moore) et § 4.1.
"""

# Chemins et accès au paquet src/.
import sys
from pathlib import Path

PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJET))

from src import visualization as viz  # noqa: E402
from src import visualization_3d as v3  # noqa: E402
from src import vortex_particles as vp  # noqa: E402

# =============================================================================================
# PARAMÈTRES À MODIFIER
# =============================================================================================
SCENARIO = "saute_mouton"   # "saute_mouton", "collision", "collision_decalee" ou "sur_mesure"
DELTA = 0.1                 # rayon de régularisation δ (≈ 0.64 × rayon d'un cœur gaussien)
N_NOEUDS = 64               # nœuds par anneau : précision ~1 % si 2πR/N ≤ δ/2 (N ≈ 128 pour δ = 0.1)
T_END = 10.0                # durée simulée
DT = 0.05                   # pas de temps (≤ 0.05 pour δ = 0.1)
PERTURBATION = 0.02         # saute-mouton : ondulation azimutale (mode 5) qui rend la dynamique 3D
SORTIE = PROJET / "exemples" / "resultats" / "03_filaments"

# =============================================================================================
# Construction des filaments
# =============================================================================================
if SCENARIO == "saute_mouton":
    # Deux anneaux identiques de même sens, écartés de 0.6 : l'arrière traverse l'avant, périodiquement.
    filaments = vp.leapfrog_scenario(delta=DELTA, separation=0.6, n_nodes=N_NOEUDS, perturbation_amplitude=PERTURBATION)
elif SCENARIO == "collision":
    # Deux anneaux de sens opposés face à face : ils ralentissent et s'étirent radialement.
    filaments = vp.head_on_collision_scenario(delta=DELTA, separation=3.0, n_nodes=N_NOEUDS)
elif SCENARIO == "collision_decalee":
    # Même chose avec des axes décalés de 0.9 : dynamique pleinement tridimensionnelle.
    filaments = vp.offset_collision_scenario(delta=DELTA, separation=3.0, offset=0.9, n_nodes=N_NOEUDS)
else:
    # Configuration libre : chaque make_ring donne un anneau (rayon, circulation, centre, normale = axe).
    filaments = (vp.make_ring(R=1.0, Gamma=1.0, n_nodes=N_NOEUDS, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                              delta=DELTA, ring_id=0)
                 + vp.make_ring(R=0.7, Gamma=1.0, n_nodes=N_NOEUDS, center=(0.3, 0.0, 0.8), normal=(0.3, 0.0, 1.0),
                                delta=DELTA, ring_id=1))

# =============================================================================================
# Simulation
# =============================================================================================
print(f"Scénario « {SCENARIO} » : δ = {DELTA}, {N_NOEUDS} nœuds par anneau, t_end = {T_END}, dt = {DT}")
# VortexFilamentSimulation : Biot–Savart régularisé (Rosenhead–Moore) intégré exactement sur les segments,
# Runge–Kutta 4 et remaillage automatique ; run renvoie l'historique (une sauvegarde tous les 4 pas).
sim = vp.VortexFilamentSimulation(filaments)
hist = sim.run(T_END, DT, save_every=4, verbose=True)
print(f"Calcul : {hist.wall_time:.1f} s ; nœuds : {hist.n_nodes[0]} → {hist.n_nodes[-1]}")

# =============================================================================================
# Figures et animation
# =============================================================================================
# Vue qui suit les anneaux pour le saute-mouton (sinon boîte fixe englobant tout le mouvement).
suivre = SCENARIO == "saute_mouton"
viz.save_figure(viz.plot_filament_snapshots(hist, n=4, follow=suivre), SORTIE / "instantanes.png")
# Tubes de vorticité (rayon ∝ δ, couleur = Γ) au début, au milieu et à la fin.
idx = [0, len(hist.frames) // 2, len(hist.frames) - 1]
viz.save_figure(v3.plot_vortex_tubes([hist.frames[i] for i in idx], hist.gammas, [hist.deltas[i] for i in idx],
                                     [hist.times[i] for i in idx], follow=suivre), SORTIE / "tubes.png")
# Animation GIF (lisible partout ; stride=2 : une sauvegarde sur deux).
chemin = viz.animate_filaments_3d(hist, str(SORTIE / "animation.gif"), fps=20, follow=suivre, stride=2)
print(f"Figures et animation enregistrées dans {SORTIE}")
