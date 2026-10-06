# Vortex toroïdaux & moteurs à vortex

Étude théorique, simulateur Python et analyse physique des **anneaux tourbillonnaires** et de leurs
applications propulsives (chambres de combustion à vortex, propulseurs pulsés).

| Phase | Contenu | Fichiers |
|---|---|---|
| 1. Théorie | vorticité, fonction de courant de Stokes, Biot–Savart, régularisation de Rosenhead–Moore, vortex de Hill, formules de Kelvin / Saffman / Fukumoto–Moffatt, nombre de formation F* ≈ 4 | [`docs/theoretical_background.md`](docs/theoretical_background.md), `src/theory.py` |
| 2A. Lagrangien 3D | filaments / particules vortex, Biot–Savart régularisé (intégration exacte sur les segments), RK4, remaillage, saute-mouton, collisions | `src/vortex_particles.py` |
| 2B. Axisymétrique | Navier–Stokes incompressible (r, z), grille MAC, projection de Chorin à chaque étage RK4, Poisson exact (FFT + spectral), anneau de Lamb–Oseen | `src/solver_2d_axisymmetric.py` |
| 3. Moteurs | vortex bidirectionnel (trajectoires hélicoïdales, temps de séjour, paroi froide, mélange), propulseur pulsé (impulsion, masse ajoutée, rendement vs fréquence et L/D) | `src/engine_analysis.py`, [`docs/vortex_engines_analysis.md`](docs/vortex_engines_analysis.md) |
| 4. Visualisation | champs ω, ψ, u, p ; validation Saffman ; animations 3D MP4/GIF ; cartes de rendement | `src/visualization.py` |
| 5. Diagnostics avancés | isosurfaces Q / λ2 reconstruites par révolution, tubes de vorticité 3D, rendu volumique de l'enstrophie (« fumée »), cartes de dissipation Φ = 2μ S:S, bilan E_k / P_z / Ω, diagrammes de Hovmöller z–t, portrait de phase du saute-mouton visqueux | `src/visualization_3d.py`, `src/visualization.py`, [`docs/theoretical_background.md` § 5](docs/theoretical_background.md#5-diagnostics-et-visualisations-avancés) |

> **Pour débuter :** suivez le [tutoriel pas à pas](docs/tutoriel.md), aussi disponible en PDF (`docs/tutoriel.pdf`).
> Il couvre l'installation, les premiers lancements, la modification des paramètres de l'écoulement et un
> parcours de lecture de la théorie et du code. Les scripts du dossier `exemples/` permettent de changer
> l'écoulement en quelques lignes, et l'application web `app.py` réunit documentation et simulations
> paramétrables dans le navigateur (voir [Application web](#application-web-streamlit)).

## Installation

Python ≥ 3.10. Depuis ce dossier (Windows / PowerShell) :

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

L'export **MP4** utilise l'exécutable `ffmpeg` s'il est dans le `PATH`. Sinon, les animations sont
automatiquement écrites en **GIF** (Pillow).

## Utilisation

```bash
.venv\Scripts\python main.py all
```

Cette commande enchaîne les cinq volets en ≈ 8 min (≈ 12 min avec `--smoke-video`) et écrit les résultats dans `outputs/` :
`figures/*.png`, `animations/*.mp4`, `data/axisym_diagnostics.csv` et la synthèse chiffrée `summary.json`.
Options utiles :

```bash
.venv\Scripts\python main.py all --quick
```

```bash
.venv\Scripts\python main.py particles --scenario leapfrog --animate gif
```

```bash
.venv\Scripts\python main.py axisym --re 2000 --t-end 30
```

```bash
.venv\Scripts\python main.py engines
```

```bash
.venv\Scripts\python main.py advanced --smoke-video
```

`advanced` lance seul le volet des diagnostics avancés : saute-mouton visqueux (Navier–Stokes, Re_Γ = 3000),
portrait de phase, tubes de vorticité et rendus volumiques. `--smoke-video` y ajoute l'animation en rendu volumique
`animations/smoke_rings_leapfrog.mp4` (≈ 4 min de plus). `--docs` recopie les figures du rapport dans `docs/figures/`.

### Application web (Streamlit)

```bash
.venv\Scripts\python -m streamlit run app.py
```

Le navigateur s'ouvre sur l'application ; son adresse s'affiche aussi dans le terminal (en général
`http://localhost:8501`). Ctrl+C dans le terminal arrête le serveur. L'application compte deux onglets :

* **📚 Théorie & Documentation** : une page d'accueil (schéma annoté d'un anneau, animation, parcours conseillé et
  accès direct aux volets), ce README, les deux rapports, le tutoriel et un glossaire. Le panneau latéral sert à
  choisir la page, à suivre son sommaire et à télécharger sa version PDF.
* **🚀 Simulation Interactive** : cinq volets — anneau visqueux (ou vortex de Hill), deux anneaux coaxiaux
  (saute-mouton ou collision frontale), filaments 3D, moteurs à vortex et formules analytiques. Pour chacun :
  * des **scénarios guidés** qui règlent tous les paramètres et disent « ce qu'il faut observer », parfois avec une
    question « prédire puis vérifier » corrigée après le calcul ;
  * la formule clé, un extrait du rapport (« théorie associée ») et les limites du modèle ;
  * des garde-fous numériques (résolution a/Δr, nombre de Péclet de maille) et une durée estimée, recalibrée sur les
    calculs déjà faits sur le PC ;
  * le bouton **▶️ Lancer la simulation**, qui appelle directement les fonctions de `src/` dans un calcul en
    arrière-plan (barre de progression, bouton « Annuler », documentation consultable pendant ce temps) ;
  * les résultats : métriques avec mini-courbes, figures, animation et vue 3D interactives (Plotly, à faire tourner
    à la souris), courbes interactives, transposition en unités physiques, **comparaison** des derniers calculs,
    **vérification de convergence** (résolution affinée) et fiche de résultats HTML.

Les calculs restent disponibles quand on recharge la page, et un calcul relancé avec les mêmes paramètres
s'affiche instantanément. Pour alimenter la barre de progression, le solveur axisymétrique accepte une fonction de
rappel `callback`, appelée après chaque pas de temps, comme la méthode des filaments. Le code de l'interface est
rangé dans le paquet `appli/` (un module par rôle, un module par volet). Le fichier `.streamlit/config.toml` règle
les thèmes clair et sombre (menu ⋮ → Settings) et désactive l'envoi de statistiques d'utilisation.

### Documents PDF

Les deux rapports existent aussi en PDF : `docs/theoretical_background.pdf` et `docs/vortex_engines_analysis.pdf`.
Après une modification des `.md` (ou des figures), cette commande les régénère :

```bash
.venv\Scripts\python docs\build_pdf.py
```

Le script convertit le Markdown en HTML et les formules LaTeX en MathML, puis fait imprimer la page en PDF par
Chrome ou Edge en mode sans interface. Il ne demande aucune installation supplémentaire. La variable
d'environnement `CHROME_PATH` permet d'indiquer un autre navigateur.

## Tests

```bash
.venv\Scripts\python -m pytest
```

La suite comporte 73 tests et s'exécute en ≈ 1 min 20. Les 7 tests marqués « lent » (calculs complets de
l'application web) peuvent être écartés pour une vérification rapide (66 tests, ≈ 35 s) :

```bash
.venv\Scripts\python -m pytest -m "not lent"
```

Elle vérifie notamment :

* **Biot–Savart** (`tests/test_biot_savart.py`) : centre et axe d'un anneau (Γ/2R), champ hors axe
  (intégrales elliptiques), filament infini singulier et régularisé, bornitude du noyau, divergence nulle,
  vitesse propre de Rosenhead–Moore (forme close exacte), conservation de l'impulsion et de l'énergie,
  saute-mouton comparé au modèle coaxial ;
* **solveur axisymétrique** : divergence nulle à 1e-12, reproduction exacte de la vorticité imposée,
  projection idempotente, vortex de Hill (vitesse initiale, puis vortex visqueux), invariants non visqueux, bilan
  d'énergie exact, vitesse proche de Saffman–Fukumoto–Moffatt, suivi de deux anneaux de circulations opposées
  (collision frontale) ;
* **théorie** : constantes de noyau (1/4 et 0,558 recalculées), cohérence U = ∂E/∂I, invariants de Hill ;
* **moteurs** : temps de séjour moyen = V/Q, conservation de ψ le long des trajectoires, masse ajoutée
  (sphère → 1/2, disque → 8ρa³/3), limite de Froude du jet continu ;
* **diagnostics avancés** (`tests/test_advanced_diagnostics.py`) : signes de Q et λ2 (vortex de Hill),
  identité ∇²p/ρ = 2Q, ∫Φ dV = μ∫ω² dV, suivi de deux anneaux, reconstruction par révolution et isosurface
  torique exactes, géométrie et fermeture des tubes (nœud de trèfle sans couture), écorché, loi de
  Beer–Lambert du rendu volumique, flux de la vorticité reconstruite = Γ, anti-crénelage du « splatting » ;
* **conversion PDF** (`tests/test_build_pdf.py`) : aucune commande LaTeX inconnue dans les rapports, formules
  MathML bien formées, constructions de base (fractions, racines, délimiteurs, primes, `cases`), ancres du sommaire ;
* **application web** (`tests/test_app.py`, avec `streamlit.testing`) : rendu de toutes les pages (ancres du
  sommaire, aucune figure manquante, glossaire), boutons de la page d'accueil, scénarios guidés, valeurs du volet
  analytique, exécution complète de chaque volet avec des réglages légers (calcul en arrière-plan, figures,
  animation, mémoire des résultats), sous-onglets, comparaison et vérification de convergence. Ces tests sont
  ignorés si streamlit ou plotly ne sont pas installés, et n'altèrent pas la calibration des durées du PC.

## Exemple d'utilisation en Python

```python
from src import vortex_particles as vp, theory

sim = vp.VortexFilamentSimulation(vp.leapfrog_scenario(delta=0.1, separation=0.6, n_nodes=128))
hist = sim.run(t_end=40.0, dt=0.05, save_every=4)
model = theory.CoaxialRingsModel([1, 1], [0, 0.6], 1.0, 0.1).integrate(40.0)   # référence réduite
```

```python
from src import solver_2d_axisymmetric as ax

res = ax.run_vortex_ring(R=1.0, a=0.2, Gamma=1.0, Re=1000, t_end=20)
res["U_sim"], res["U_saffman"], res["U_saffman_fm"]      # vitesses simulée et théoriques
```

## Principaux résultats

* **Lagrangien 3D** : la vitesse propre discrète converge à l'ordre 2 en Δs/δ vers la forme close exacte
  de l'anneau de Rosenhead–Moore, U = Γ/(2π√(4R²+δ²))[K(m) − E(m)], établie dans ce projet. Le
  saute-mouton (13 dépassements) coïncide avec le modèle coaxial régularisé et conserve l'impulsion et
  l'énergie à ~1e-4.
* **Axisymétrique** : bilan d'énergie exact (aucune dissipation numérique), solution convergée en
  maillage (0,03 %) et en domaine (< 0,6 %). La vitesse de l'anneau suit la formule de Saffman et, à
  σ/R ≈ 0,35, reste à 2,7 % de sa correction de Fukumoto–Moffatt.
* **Moteurs** : forme close de la distribution des temps de séjour du vortex bidirectionnel ; loi de
  « paroi froide » Φ ≈ ln√2 · Peκ · e^(−0,24 Peκ) ; nombre de formation F* ≈ 3,8. Le propulseur pulsé
  gagne jusqu'à ~8 % de rendement sur le jet continu équivalent, avec un optimum à L/D ≈ F*.
* **Diagnostics avancés** : l'énergie simulée suit E₀ − ν∫Ω dt à 8·10⁻⁶ E₀ près ; la dissipation Φ se
  concentre sur une couronne de rayon 1,34 a autour de chaque cœur. En saute-mouton visqueux (Re_Γ = 3000),
  le portrait de phase, fermé en fluide parfait, devient une spirale : après deux traversées (t = 1,45 et 3,92),
  les cœurs fusionnent à t ≈ 5,4, quand le rapport séparation / rayon de cœur passe sous ≈ 4.

## Structure

```text
vortex_project/
├── app.py                            # application web Streamlit : point d'entrée (deux onglets)
├── appli/                            # code de l'application web
│   ├── commun.py, calibration.py, stockage.py, taches.py   # outils, durées estimées, historique, calcul en arrière-plan
│   ├── documentation.py, accueil.py, glossaire.py          # onglet de documentation
│   ├── simulation.py, resultats.py, pedagogie.py, vues3d.py # onglet de simulation, résultats, aides, vues 3D Plotly
│   └── volets/                       # un module par volet (anneau, saute_mouton, filaments, moteurs, theorie)
├── .streamlit/config.toml            # thèmes clair et sombre de l'application web
├── .gitignore                        # fichiers non versionnés (.venv, caches, sorties régénérables)
├── docs/
│   ├── theoretical_background.md     # Phase 1 — théorie (+ version .pdf)
│   ├── vortex_engines_analysis.md    # Phase 3 — rapport moteurs à vortex (+ version .pdf)
│   ├── tutoriel.md                   # tutoriel pas à pas : lancer, paramétrer, comprendre (+ version .pdf)
│   ├── build_pdf.py                  # conversion Markdown + LaTeX → PDF (MathML, Chrome/Edge)
│   └── figures/                      # figures du rapport (main.py --docs)
├── exemples/                         # scripts courts à paramètres modifiables (résultats dans exemples/resultats/)
│   ├── 01_anneau_visqueux.py
│   ├── 02_deux_anneaux_navier_stokes.py
│   ├── 03_filaments_3d.py
│   └── 04_moteurs.py
├── src/
│   ├── __init__.py
│   ├── theory.py                     # modèles analytiques (Hill, Saffman, RM, slug, coaxial)
│   ├── vortex_particles.py           # approche A : filaments / particules vortex 3D
│   ├── solver_2d_axisymmetric.py     # approche B : Navier–Stokes axisymétrique (projection)
│   ├── engine_analysis.py            # moteurs à vortex
│   ├── visualization.py              # figures 2D, bilans, Hovmöller, portrait de phase, animations
│   └── visualization_3d.py           # isosurfaces de révolution, tubes de vorticité, rendu volumique
├── tests/
│   ├── test_biot_savart.py
│   ├── test_axisymmetric_solver.py
│   ├── test_theory.py
│   ├── test_engine_analysis.py
│   ├── test_advanced_diagnostics.py
│   ├── test_build_pdf.py
│   └── test_app.py
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Limites connues

* Méthode filamentaire : pas de reconnexion ; au-delà d'une approche des filaments à ~2δ,
  la régularisation domine. Le coût est en O(M²) avec un calcul numpy vectorisé ; il n'y a pas d'arbre
  multipolaire.
* Solveur axisymétrique : sans swirl, avec une paroi glissante en r = L_r et une périodicité en z. Il est
  adapté aux régimes laminaires. Une étude du nombre de Péclet de maille (Pe_Δ = u_max Δr/ν) montre qu'il reste
  stable jusqu'à Re_Γ = 40 000 ; de petites oscillations (vorticité négative parasite > 0,2 %) n'apparaissent
  qu'avec un cœur mal résolu : au-delà de Pe_Δ ≈ 200 pour 4 ≤ a/Δr < 6, de Pe_Δ ≈ 70 pour a/Δr < 4, jamais pour
  a/Δr ≥ 6 dans la plage testée.
* Moteurs : modèles analytiques et semi-empiriques d'ordre de grandeur. La fermeture de surpression du
  propulseur pulsé est calibrée sur des mesures publiées (voir le rapport).
* Visualisations 3D : rendu matplotlib (tri des facettes par profondeur, sans z-buffer). Le rendu volumique
  est un lancer de rayons numpy sur CPU, d'environ 8 s par image de 420 × 330 pixels. Les isosurfaces par
  révolution ne s'appliquent qu'aux champs axisymétriques ; les champs des filaments passent par le rendu volumique.
* Application web : prévue pour un usage local. L'historique et la mémoire des résultats vivent dans la mémoire du
  serveur (perdus à son redémarrage) ; un seul calcul à la fois par session.
