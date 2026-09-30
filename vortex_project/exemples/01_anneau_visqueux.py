"""
Exemple 1 — Un anneau tourbillonnaire visqueux (solveur Navier–Stokes axisymétrique).

Modifiez les paramètres du bloc « PARAMÈTRES À MODIFIER », puis lancez depuis le dossier vortex_project :
    .venv\\Scripts\\python exemples\\01_anneau_visqueux.py
Résultats : exemples/resultats/01_anneau_visqueux/ (figures PNG) et tableau des vitesses dans la console.
Théorie associée : docs/theoretical_background.pdf, § 2.3 (Saffman, Fukumoto–Moffatt) et § 4.2 (solveur).
"""

# sys.path : liste des dossiers où Python cherche les modules ; pathlib.Path : chemins de fichiers.
import sys
from pathlib import Path

# Dossier du projet (parent du dossier exemples/), ajouté en tête du chemin de recherche pour « import src ».
PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJET))

# numpy : calcul sur tableaux ; modules du projet : solveur axisymétrique et figures.
import numpy as np  # noqa: E402
from src import solver_2d_axisymmetric as ax  # noqa: E402
from src import visualization as viz  # noqa: E402

# =============================================================================================
# PARAMÈTRES À MODIFIER (unités adimensionnelles : longueurs en R, temps en R²/Γ)
# =============================================================================================
R = 1.0          # rayon de l'anneau
A = 0.25         # rayon du cœur gaussien (anneau mince : A/R ≲ 0.35)
GAMMA = 1.0      # circulation (Γ > 0 : l'anneau avance vers les z croissants)
RE = 1000.0      # nombre de Reynolds Re_Γ = Γ/ν (≲ 5000 à cette résolution)
T_END = 8.0      # durée simulée
DR = 0.04        # pas de maillage près de l'anneau : garder A/DR ≥ 6 (0.025 = résolution du rapport)
SORTIE = PROJET / "exemples" / "resultats" / "01_anneau_visqueux"

# =============================================================================================
# Simulation
# =============================================================================================
# Zone de maillage uniforme jusqu'à R + 1.2 (au moins 2.2) et paroi à 8 R : le domaine suit la taille de l'anneau.
r_uniforme = max(2.2, R + 1.2)
print(f"Anneau : R = {R}, a = {A}, Γ = {GAMMA}, Re_Γ = {RE:g}, t_end = {T_END}, Δr = {DR}")
# run_vortex_ring : construit la grille, initialise l'anneau de Lamb–Oseen, intègre et compare à la théorie.
# snapshot_times : instants où les champs complets sont enregistrés ; record_profiles : profils du Hovmöller.
res = ax.run_vortex_ring(R=R, a=A, Gamma=GAMMA, Re=RE, t_end=T_END, dr=DR, Lr=8.0 * R, r_uniform=r_uniforme,
                         snapshot_times=(0.0, T_END), record_profiles=True, verbose=True)
# Séries temporelles des diagnostics (t, Z, R, σ, énergie, dissipation, divergence...).
d = res["diagnostics"]
print(f"Grille : {res['grid']} — calcul : {res['wall_time']:.1f} s")

# Tableau de comparaison à la théorie, après le transitoire initial (t > 2).
print("    t      U_simulée   Saffman (écart)    Saffman + Fukumoto–Moffatt (écart)")
for tt in np.linspace(min(2.0, T_END), T_END, 4):
    # Indice du diagnostic le plus proche de tt.
    k = int(np.argmin(np.abs(d["t"] - tt)))
    u, us, ufm = res["U_sim"][k], res["U_saffman"][k], res["U_saffman_fm"][k]
    print(f"  {d['t'][k]:5.1f}   {u:.4f}      {us:.4f} ({u / us - 1:+.1%})    {ufm:.4f} ({u / ufm - 1:+.1%})")
# Contrôles de fiabilité : divergence (incompressibilité) et part de l'énergie perdue.
print(f"Divergence max : {d['max_div'].max():.1e} ; énergie perdue : {100 * (1 - d['energy'][-1] / d['energy'][0]):.1f} %")

# =============================================================================================
# Figures
# =============================================================================================
# Position de l'anneau aux instants enregistrés (interpolation linéaire de Z(t)).
Z_s = [np.interp(s["t"], d["t"], d["Z"]) for s in res["snapshots"]]
# Champs finaux (vorticité, lignes de courant, vitesse, pression), vitesse comparée à la théorie,
# diagramme de Hovmöller et bilan d'énergie.
viz.save_figure(viz.plot_axisymmetric_fields(res["snapshots"][-1], Z_ring=Z_s[-1], U_ring=float(np.mean(res["U_sim"][-10:]))),
                SORTIE / "champs_final.png")
viz.save_figure(viz.plot_ring_speed_validation(res), SORTIE / "vitesse_vs_theorie.png")
viz.save_figure(viz.plot_hovmoller(res["profiles"], tracks=[(d["t"], d["Z"], "centroïde simulé")]), SORTIE / "hovmoller.png")
viz.save_figure(viz.plot_energy_enstrophy_budget(res), SORTIE / "bilan_energie.png")
print(f"Figures enregistrées dans {SORTIE}")
