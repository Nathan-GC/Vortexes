"""
Package ``src`` — Vortex toroïdaux (anneaux tourbillonnaires) & moteurs à vortex.

Modules
-------
theory                 : modèles analytiques de référence (Hill, Kelvin/Saffman, Rosenhead–Moore,
                         anneau filamentaire, modèle « slug », nombre de formation, anneaux coaxiaux).
vortex_particles       : approche A — méthode lagrangienne 3D (filaments / particules vortex,
                         Biot–Savart régularisé, RK4, saute-mouton, collision frontale).
solver_2d_axisymmetric : approche B — Navier–Stokes incompressible axisymétrique (r, z),
                         grille MAC, méthode de projection, noyau de Lamb–Oseen.
engine_analysis        : phase 3 — chambres de combustion à vortex et propulseurs pulsés.
visualization          : phase 4 — champs, courbes de validation, animations 3D (MP4/GIF).
"""

# Ce fichier fait du dossier « src » un paquet Python : on importe ses modules par
# « from src import theory », « from src import vortex_particles as vp », etc.
# Numéro de version du projet (convention : chaîne « majeur.mineur.correctif »).
__version__ = "1.0.0"
