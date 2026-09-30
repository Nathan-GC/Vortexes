"""
vortex_particles.py — Approche A : méthode lagrangienne 3D (filaments / particules vortex).

Principe
--------
Un anneau tourbillonnaire est discrétisé en un ou plusieurs filaments fermés (polygones de N
nœuds) portant chacun une circulation Γ. Les nœuds sont des marqueurs lagrangiens advectés par
la vitesse qu'induit l'ensemble des filaments (loi de Biot–Savart) :

    u(x) = (1/4π) Σ_k Γ_k ∫_{segment k} dl' × (x - x') / (|x - x'|² + δ_k²)^{3/2}

Le noyau de Rosenhead–Moore (δ > 0) supprime la singularité en r = 0 et modélise un cœur de
taille finie : un anneau circulaire se propage alors à  U = Γ/(4πR)[ln(8R/δ) - 1].

Deux discrétisations du noyau sont proposées :

* ``kernel="segment"``  (défaut) — intégration **exacte** du noyau régularisé le long de chaque
  segment rectiligne :
      u = Γ/(4π) · (r₁ × r₂) / (|r₁ × r₂|² + δ²|r₀|²) · r₀·( r₁/√(|r₁|²+δ²) - r₂/√(|r₂|²+δ²) )
  avec r₀ = B - A, r₁ = x - A, r₂ = x - B ;
* ``kernel="particle"`` — particules vortex (« vortex blobs ») placées au milieu des segments,
  de poids vectoriel α_k = Γ_k (B_k - A_k) (règle du point milieu) :
      u = (1/4π) Σ_k α_k × (x - x_k) / (|x - x_k|² + δ_k²)^{3/2} .

Les poids des particules étant recalculés à partir des nœuds, l'étirement tourbillonnaire est
automatiquement pris en compte et le théorème de Kelvin (Γ conservée) est satisfait.

Fonctionnalités : RK4, remaillage adaptatif par spline périodique, conservation optionnelle du
volume du cœur (δ² L = cste), diagnostics (impulsion, énergie, rayon et position des anneaux),
scénarios : anneau isolé, saute-mouton (*leapfrogging*), collision frontale, collision décalée.
"""

# Les annotations de type ne sont pas évaluées à l'exécution.
from __future__ import annotations

# time.perf_counter() : horloge haute résolution pour mesurer le temps de calcul.
import time
# dataclass : génère automatiquement __init__, __repr__... à partir des champs déclarés ;
# field(...) permet de préciser une valeur par défaut (ou une fabrique de valeur par défaut).
from dataclasses import dataclass, field

# numpy : tableaux et calcul vectoriel.
import numpy as np
# CubicSpline : interpolation par spline cubique (ici périodique) pour le remaillage des filaments.
from scipy.interpolate import CubicSpline

# Import relatif du module théorique du même paquet (formules δ ↔ rayon de cœur, etc.).
from . import theory

# Noms publics du module.
__all__ = [
    "biot_savart_segments",
    "biot_savart_particles",
    "Filament",
    "VortexFilamentSimulation",
    "SimulationHistory",
    "ring_nodes",
    "make_ring",
    "single_ring_scenario",
    "leapfrog_scenario",
    "head_on_collision_scenario",
    "offset_collision_scenario",
    "vorticity_line_density",
    "splat_vorticity",
]

# Constante 1/(4π) précalculée (préfacteur de la loi de Biot–Savart).
_INV_4PI = 1.0 / (4.0 * np.pi)
# Nombre maximal de couples (cible, source) traités d'un coup : limite la taille des tableaux
# temporaires (P × S éléments) pour ne pas saturer la mémoire.
_MAX_PAIRS_PER_CHUNK = 400_000  # contrôle la mémoire des tableaux temporaires (P × S)


# =============================================================================================
# Noyaux de Biot–Savart régularisés
# =============================================================================================
def _chunks(n_targets, n_sources):
    # Nombre de cibles par bloc pour que (cibles × sources) ≤ _MAX_PAIRS_PER_CHUNK ;
    # max(1, ...) garantit au moins une cible par bloc et évite une division par zéro.
    step = max(1, _MAX_PAIRS_PER_CHUNK // max(1, n_sources))
    # range(0, n, step) : indices de début de chaque bloc.
    for start in range(0, n_targets, step):
        # « yield » fait de cette fonction un générateur : elle fournit un objet slice
        # (tranche d'indices [start, fin)) à chaque itération de la boucle appelante.
        yield slice(start, min(n_targets, start + step))


def biot_savart_segments(targets, a, b, gamma, delta):
    """Vitesse induite en ``targets`` par des segments tourbillonnaires rectilignes régularisés.

    Paramètres
    ----------
    targets : (P, 3) points d'évaluation.
    a, b    : (S, 3) extrémités des segments (orientés de a vers b).
    gamma   : (S,) ou scalaire, circulation de chaque segment.
    delta   : (S,) ou scalaire, rayon de régularisation de Rosenhead–Moore (δ = 0 : singulier).

    Retour
    ------
    (P, 3) vitesses induites. Intégration exacte du noyau K_δ le long de chaque segment ; pour un
    point situé sur la droite porteuse d'un segment, la contribution de ce segment est nulle.
    """
    # np.atleast_2d : transforme un point isolé (3,) en tableau (1, 3) ; np.asarray convertit
    # les listes Python en tableaux numpy de réels.
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    # Nombre de segments sources (a.shape = (S, 3)).
    n_src = a.shape[0]
    # Γ/(4π) pour chaque segment ; np.broadcast_to étend un Γ scalaire aux S segments.
    g4 = _INV_4PI * np.broadcast_to(np.asarray(gamma, dtype=float), (n_src,))
    # le « tiny » évite 0/0 sans branchement : un terme nul (point sur la droite du segment
    # avec δ = 0, ou point confondu avec un nœud) donne alors une contribution exactement nulle
    # d2 = δ² par segment (+ 1e-300 pour ne jamais diviser exactement par zéro).
    d2 = np.broadcast_to(np.asarray(delta, dtype=float) ** 2, (n_src,)) + 1e-300
    # Composantes des extrémités a (début) et b (fin) des segments : tableaux (S,).
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    # r0 = b - a : vecteur directeur de chaque segment.
    r0x, r0y, r0z = bx - ax, by - ay, bz - az
    # Terme δ²|r0|² du dénominateur (indépendant de la cible, donc précalculé).
    d2r0sq = d2 * (r0x * r0x + r0y * r0y + r0z * r0z) + 1e-300
    # np.empty_like : tableau résultat (P, 3) non initialisé (chaque ligne sera remplie).
    out = np.empty_like(targets)
    # Traitement des cibles par blocs (limite mémoire).
    for sl in _chunks(targets.shape[0], n_src):
        # Coordonnées des cibles du bloc sous forme de colonnes (p, 1) : le « broadcasting »
        # numpy avec les lignes (S,) des sources produit des tableaux (p, S) de toutes les paires.
        tx = targets[sl, 0:1]
        ty = targets[sl, 1:2]
        tz = targets[sl, 2:3]
        # r1 = x - a (de l'origine du segment vers la cible), tableaux (p, S).
        r1x, r1y, r1z = tx - ax, ty - ay, tz - az
        # r2 = x - b (de l'extrémité du segment vers la cible).
        r2x, r2y, r2z = tx - bx, ty - by, tz - bz
        # Produit vectoriel c = r1 × r2 calculé composante par composante ; les opérateurs
        # « -= » travaillent en place (pas de tableau temporaire supplémentaire).
        cx = r1y * r2z
        cx -= r1z * r2y
        cy = r1z * r2x
        cy -= r1x * r2z
        cz = r1x * r2y
        cz -= r1y * r2x
        # n1 = √(|r1|² + δ²) ; np.sqrt(x, out=x) calcule la racine en place.
        n1 = r1x * r1x + r1y * r1y + r1z * r1z + d2
        np.sqrt(n1, out=n1)
        # n2 = √(|r2|² + δ²).
        n2 = r2x * r2x + r2y * r2y + r2z * r2z + d2
        np.sqrt(n2, out=n2)
        # Projection r0·(r1/n1 - r2/n2) : bornes de l'intégrale exacte le long du segment.
        proj = (r0x * r1x + r0y * r1y + r0z * r1z) / n1
        proj -= (r0x * r2x + r0y * r2y + r0z * r2z) / n2
        # Dénominateur |r1 × r2|² + δ²|r0|² (régularisé : jamais nul si δ > 0).
        denom = cx * cx + cy * cy + cz * cz + d2r0sq
        # Coefficient scalaire de chaque paire : Γ/(4π) × projection / dénominateur.
        proj *= g4
        proj /= denom
        # np.einsum("ij,ij->i", A, B) = Σ_j A[i,j]·B[i,j] : somme des contributions de tous les
        # segments (indice j) pour chaque cible (indice i), sans créer le produit A*B complet.
        out[sl, 0] = np.einsum("ij,ij->i", proj, cx)
        out[sl, 1] = np.einsum("ij,ij->i", proj, cy)
        out[sl, 2] = np.einsum("ij,ij->i", proj, cz)
    return out


def biot_savart_particles(targets, positions, alphas, delta):
    """Vitesse induite par des particules vortex régularisées (noyau de Rosenhead–Moore).

    u(x) = (1/4π) Σ_k α_k × (x - x_k) / (|x - x_k|² + δ_k²)^{3/2}

    ``alphas`` : (S, 3) poids vectoriels α_k = ∫ ω dV ≈ Γ_k Δl_k.
    """
    # Conversion des entrées en tableaux 2D de réels.
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    alphas = np.atleast_2d(np.asarray(alphas, dtype=float))
    # Nombre de particules sources.
    n_src = positions.shape[0]
    # δ² par particule (scalaire étendu à toutes les particules si besoin).
    d2 = np.broadcast_to(np.asarray(delta, dtype=float) ** 2, (n_src,))
    # Composantes des positions des particules et de leurs poids vectoriels.
    px, py, pz = positions[:, 0], positions[:, 1], positions[:, 2]
    wx, wy, wz = alphas[:, 0], alphas[:, 1], alphas[:, 2]
    # Tableau résultat.
    out = np.empty_like(targets)
    # Boucle sur les blocs de cibles.
    for sl in _chunks(targets.shape[0], n_src):
        # d = x - x_k pour toutes les paires (p, S).
        dx = targets[sl, 0:1] - px
        dy = targets[sl, 1:2] - py
        dz = targets[sl, 2:3] - pz
        # 1/(4π (|d|² + δ²)^{3/2}) : noyau régularisé de Rosenhead–Moore.
        inv3 = _INV_4PI * (dx * dx + dy * dy + dz * dz + d2) ** -1.5
        # Composantes du produit vectoriel α × d pondérées, puis np.sum(..., axis=1) : somme
        # sur toutes les particules sources pour chaque cible.
        out[sl, 0] = np.sum((wy * dz - wz * dy) * inv3, axis=1)
        out[sl, 1] = np.sum((wz * dx - wx * dz) * inv3, axis=1)
        out[sl, 2] = np.sum((wx * dy - wy * dx) * inv3, axis=1)
    return out


# =============================================================================================
# Filaments
# =============================================================================================
# @dataclass génère le constructeur Filament(nodes, gamma, delta, ring_id, spacing, delta0, length0).
@dataclass
class Filament:
    """Filament tourbillonnaire fermé (polygone de N nœuds, segment i : nœud i → nœud i+1)."""

    # Coordonnées des nœuds, tableau (N, 3).
    nodes: np.ndarray
    # Circulation portée par le filament.
    gamma: float
    # Rayon de régularisation de Rosenhead–Moore actuel.
    delta: float
    # Identifiant de l'anneau auquel appartient le filament (plusieurs filaments par anneau possible).
    ring_id: int = 0
    spacing: float = field(default=None)   # pas de remaillage cible
    delta0: float = field(default=None)    # pour la conservation du volume du cœur
    # Longueur initiale du filament (référence pour la conservation du volume du cœur).
    length0: float = field(default=None)

    # __post_init__ est appelé automatiquement par le constructeur généré par @dataclass.
    def __post_init__(self):
        # np.array(..., dtype=float) : copie des nœuds en tableau de réels (indépendant de l'entrée).
        self.nodes = np.array(self.nodes, dtype=float)
        # Vérification de la forme : 2 dimensions, 3 coordonnées, au moins 3 nœuds (un polygone).
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3 or self.nodes.shape[0] < 3:
            raise ValueError("nodes doit être un tableau (N >= 3, 3)")
        # Valeurs par défaut calculées à partir de la géométrie initiale.
        if self.length0 is None:
            self.length0 = self.length()
        if self.delta0 is None:
            self.delta0 = float(self.delta)
        if self.spacing is None:
            # Pas cible = longueur moyenne d'un segment au départ.
            self.spacing = self.length0 / self.n

    # Nombre de nœuds (= nombre de segments, le polygone étant fermé).
    @property
    def n(self):
        return self.nodes.shape[0]

    def segment_vectors(self):
        # np.roll(nodes, -1, axis=0) décale les lignes d'un cran vers le haut : la ligne i contient
        # le nœud i+1 (et la dernière ligne le nœud 0, ce qui ferme le polygone).
        # La différence donne le vecteur de chaque segment (nœud i → nœud i+1).
        return np.roll(self.nodes, -1, axis=0) - self.nodes

    def length(self):
        # np.linalg.norm(..., axis=1) : norme euclidienne de chaque vecteur segment ;
        # .sum() : longueur totale du polygone.
        return float(np.linalg.norm(self.segment_vectors(), axis=1).sum())

    def copy(self):
        # Copie profonde des nœuds (nodes.copy()) pour que les deux filaments soient indépendants.
        return Filament(self.nodes.copy(), self.gamma, self.delta, self.ring_id, self.spacing,
                        self.delta0, self.length0)


def _orthonormal_basis(normal):
    # Normalisation du vecteur normal au plan de l'anneau (np.linalg.norm = norme euclidienne).
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    # Vecteur d'essai non colinéaire à n (on évite e_x si n est presque parallèle à e_x).
    trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    # Orthogonalisation de Gram–Schmidt : on retire la composante de trial selon n
    # (np.dot = produit scalaire).
    e1 = trial - np.dot(trial, n) * n
    # Normalisation de e1.
    e1 /= np.linalg.norm(e1)
    # np.cross = produit vectoriel ; e2 = n × e1 complète la base.
    e2 = np.cross(n, e1)  # (e1, e2, n) direct : e1 × e2 = n
    return e1, e2, n


def ring_nodes(R, n_nodes, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), perturbation_mode=0,
               perturbation_amplitude=0.0, phase=0.0, axial_offset=0.0, radial_offset=0.0):
    """Nœuds d'un anneau circulaire (éventuellement perturbé) de rayon R.

    Les nœuds sont parcourus dans le sens direct autour de ``normal`` : avec Γ > 0 l'anneau se
    propage selon +normal. Perturbation radiale :  r(θ) = (R + radial_offset)(1 + A cos(mθ + φ)).
    ``axial_offset`` décale le cercle le long de la normale (utile pour les cœurs multi-filaments).
    """
    # Base orthonormée directe (e1, e2) du plan de l'anneau et normale n.
    e1, e2, n = _orthonormal_basis(normal)
    # np.arange(N) = [0, 1, ..., N-1] : angles θ_k = 2πk/N régulièrement répartis (sans répéter 2π).
    theta = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
    # Rayon local de chaque nœud, avec la perturbation azimutale de mode m (np.cos vectoriel).
    rad = (R + radial_offset) * (1.0 + perturbation_amplitude * np.cos(perturbation_mode * theta + phase))
    # Centre de l'anneau en tableau numpy.
    c = np.asarray(center, dtype=float)
    # Position = centre + rayon (cos θ e1 + sin θ e2) + décalage axial selon n ;
    # « [None, :] » et « [:, None] » ajoutent des axes pour le broadcasting (N, 3).
    return (c[None, :] + rad[:, None] * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2)
            + axial_offset * n[None, :])


def make_ring(R=1.0, Gamma=1.0, n_nodes=128, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
              delta=None, core_radius=None, core_type="gaussian", n_core_layers=0, ring_id=0,
              perturbation_mode=0, perturbation_amplitude=0.0, phase=0.0):
    """Construit un anneau tourbillonnaire sous forme de liste de filaments.

    * ``n_core_layers = 0`` : un seul filament de paramètre δ. Si ``delta`` n'est pas donné, il
      est déduit du rayon de cœur physique ``core_radius`` (``core_type`` = 'gaussian' ou
      'uniform') pour que la vitesse de propagation soit celle de l'anneau réel
      (cf. :func:`theory.rosenhead_moore_delta`).
    * ``n_core_layers = n > 0`` : cœur épais représenté par 1 + 6 + 12 + … + 6n filaments placés
      sur des couronnes concentriques du plan méridien, de circulations pondérées par un profil
      gaussien de rayon ``core_radius`` et régularisés par δ = pas des couronnes.
    """
    # --- Cas 1 : un seul filament.
    if n_core_layers == 0:
        # δ non fourni : on le déduit du rayon de cœur physique.
        if delta is None:
            if core_radius is None:
                raise ValueError("donner delta ou core_radius")
            # δ = a·exp(C - 1) (même vitesse de propagation que l'anneau à cœur physique).
            delta = theory.rosenhead_moore_delta(core_radius, core_type)
        # Coordonnées des nœuds du cercle.
        nodes = ring_nodes(R, n_nodes, center, normal, perturbation_mode, perturbation_amplitude, phase)
        # Liste contenant un unique Filament.
        return [Filament(nodes, Gamma, float(delta), ring_id)]

    # --- Cas 2 : cœur épais multi-filaments.
    if core_radius is None:
        raise ValueError("core_radius est requis pour un cœur multi-filaments")
    # Pas radial h entre couronnes concentriques (on couvre le cœur jusqu'à ~2σ).
    h = 2.0 * core_radius / (n_core_layers + 0.5)  # couronnes jusqu'à ~2σ
    # Filament central (décalage (0, 0)) et son aire représentative π(h/2)².
    offsets, weights = [(0.0, 0.0)], [np.pi * (0.5 * h) ** 2]
    # Couronnes successives : 6, 12, 18... filaments (réseau quasi hexagonal).
    for layer in range(1, n_core_layers + 1):
        # Rayon de la couronne dans le plan méridien.
        rho = layer * h
        # Nombre de filaments de la couronne.
        count = 6 * layer
        # Aire de l'anneau de largeur h partagée entre les filaments de la couronne.
        area = 2.0 * np.pi * rho * h / count
        for k in range(count):
            # Angle du filament (décalé d'un demi-pas une couronne sur deux).
            phi = 2.0 * np.pi * (k + 0.5 * (layer % 2)) / count
            # Décalage (radial, axial) du filament par rapport au centre du cœur.
            offsets.append((rho * np.cos(phi), rho * np.sin(phi)))
            weights.append(area)
    # Conversion de la liste en tableau (nf, 2).
    offsets = np.array(offsets)
    # Distance au centre du cœur au carré (somme des carrés sur chaque ligne).
    rho2 = (offsets**2).sum(axis=1)
    # Poids gaussien × aire représentative.
    w = np.array(weights) * np.exp(-rho2 / core_radius**2)
    # Circulation de chaque filament, normalisée pour que la somme vaille Γ.
    gammas = Gamma * w / w.sum()
    # Régularisation des filaments : pas des couronnes (recouvrement des noyaux) sauf δ imposé.
    fil_delta = h if delta is None else delta
    filaments = []
    # zip parcourt en parallèle les décalages et les circulations.
    for (dr, dz), g in zip(offsets, gammas):
        # Cercle de rayon R + dr décalé de dz le long de la normale.
        nodes = ring_nodes(R, n_nodes, center, normal, perturbation_mode, perturbation_amplitude,
                           phase, axial_offset=dz, radial_offset=dr)
        filaments.append(Filament(nodes, float(g), float(fil_delta), ring_id))
    return filaments


# =============================================================================================
# Historique de simulation
# =============================================================================================
@dataclass
class SimulationHistory:
    """Résultats d'une simulation lagrangienne.

    ``frames[k]`` : liste des tableaux de nœuds (un par filament) à l'instant ``times[k]``.
    Diagnostics (échantillonnés à chaque sauvegarde) : ``diag_time``, ``impulse`` (n, 3),
    ``energy`` (n,), ``ring_centroid`` (n, n_rings, 3), ``ring_radius`` (n, n_rings),
    ``ring_axial`` (n, n_rings) (position le long de l'axe de symétrie initial).
    """

    # field(default_factory=list) : chaque instance reçoit sa propre liste vide
    # (une valeur par défaut mutable partagée serait un piège classique en Python).
    times: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    ring_ids: list = field(default_factory=list)
    gammas: list = field(default_factory=list)
    diag_time: list = field(default_factory=list)
    impulse: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    ring_centroid: list = field(default_factory=list)
    ring_radius: list = field(default_factory=list)
    ring_axial: list = field(default_factory=list)
    n_nodes: list = field(default_factory=list)
    # δ de chaque filament à chaque sauvegarde (varie si le volume du cœur est conservé).
    deltas: list = field(default_factory=list)
    # Durée réelle du calcul (secondes).
    wall_time: float = 0.0
    # Axe de référence pour la position axiale des anneaux (fabrique lambda : nouveau tableau par instance).
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def as_arrays(self):
        """Convertit les diagnostics en tableaux numpy (dict)."""
        # np.asarray transforme chaque liste de valeurs (ou de tableaux) en tableau numpy.
        return {
            "t": np.asarray(self.diag_time),
            "impulse": np.asarray(self.impulse),
            "energy": np.asarray(self.energy),
            "centroid": np.asarray(self.ring_centroid),
            "radius": np.asarray(self.ring_radius),
            "axial": np.asarray(self.ring_axial),
            "n_nodes": np.asarray(self.n_nodes),
        }


# =============================================================================================
# Simulation
# =============================================================================================
class VortexFilamentSimulation:
    """Évolution lagrangienne d'un ensemble de filaments tourbillonnaires fermés.

    Paramètres
    ----------
    filaments : liste de :class:`Filament`.
    kernel    : 'segment' (intégration exacte du noyau RM sur les segments) ou 'particle'.
    conserve_core_volume : si vrai, δ_f = δ0_f √(L0_f / L_f) (volume du tube conservé).
    remesh    : active le remaillage (spline périodique) quand un segment dépasse
                ``remesh_max`` × pas cible ou descend sous ``remesh_min`` × pas cible.
    """

    def __init__(self, filaments, kernel="segment", conserve_core_volume=False, remesh=True,
                 remesh_max=1.6, remesh_min=0.35, min_nodes=16):
        # Contrôle du type de noyau demandé.
        if kernel not in ("segment", "particle"):
            raise ValueError("kernel doit valoir 'segment' ou 'particle'")
        # Copie des filaments (la simulation ne modifie pas les objets de l'appelant).
        self.filaments = [f.copy() for f in filaments]
        # Options de calcul.
        self.kernel = kernel
        self.conserve_core_volume = conserve_core_volume
        self.remesh_enabled = remesh
        self.remesh_max = remesh_max
        self.remesh_min = remesh_min
        self.min_nodes = min_nodes
        # Temps physique courant et nombre de pas effectués.
        self.time = 0.0
        self.n_steps = 0
        # Construction des tableaux d'indices (connectivité des segments).
        self._rebuild_topology()

    # ------------------------------------------------------------------ topologie / données
    def _rebuild_topology(self):
        # Nombre de nœuds de chaque filament.
        counts = np.array([f.n for f in self.filaments])
        # np.cumsum = somme cumulée : _offsets[k] est l'indice du premier nœud du filament k dans
        # le tableau global concaténé (et _offsets[-1] le nombre total de nœuds).
        self._offsets = np.concatenate([[0], np.cumsum(counts)])
        # Liste des indices « nœud suivant » de chaque filament.
        nxt = []
        # zip associe chaque filament à son indice de départ.
        for f, off in zip(self.filaments, self._offsets[:-1]):
            # Indices globaux des nœuds du filament.
            idx = np.arange(f.n) + off
            # np.roll(idx, -1) : le suivant du dernier nœud est le premier (filament fermé).
            nxt.append(np.roll(idx, -1))
        # _next[i] = indice global du nœud qui suit le nœud i (le segment i va de i à _next[i]).
        self._next = np.concatenate(nxt)
        # np.full(n, valeur) : tableau de n copies de la valeur ; circulation de chaque segment.
        self._seg_gamma = np.concatenate([np.full(f.n, f.gamma) for f in self.filaments])
        # Numéro du filament propriétaire de chaque segment (enumerate donne (indice, filament)).
        self._seg_owner = np.concatenate([np.full(f.n, k) for k, f in enumerate(self.filaments)])
        # Identifiant d'anneau de chaque filament.
        self._ring_ids = np.array([f.ring_id for f in self.filaments])

    # Nombre total de nœuds (dernier élément des offsets).
    @property
    def n_nodes(self):
        return int(self._offsets[-1])

    def positions(self):
        """Tableau (M, 3) de tous les nœuds (concaténés filament par filament)."""
        # np.concatenate(..., axis=0) empile verticalement les tableaux de nœuds.
        return np.concatenate([f.nodes for f in self.filaments], axis=0)

    def _set_positions(self, X):
        # Redistribue le tableau global X aux filaments (tranches [i0, i1) de chaque filament).
        for f, i0, i1 in zip(self.filaments, self._offsets[:-1], self._offsets[1:]):
            f.nodes = X[i0:i1].copy()

    def _segment_deltas(self, X):
        """δ de chaque segment (éventuellement ajusté pour conserver le volume du cœur)."""
        # Sans conservation du volume : δ constant par filament, recopié pour chaque segment.
        if not self.conserve_core_volume:
            return np.concatenate([np.full(f.n, f.delta) for f in self.filaments])
        # Longueur de chaque segment de la configuration X.
        seg_len = np.linalg.norm(X[self._next] - X, axis=1)
        # np.bincount(indices, weights) : somme des longueurs par filament propriétaire.
        lengths = np.bincount(self._seg_owner, weights=seg_len, minlength=len(self.filaments))
        # δ² L = cste ⇒ δ = δ0 √(L0/L) pour chaque filament.
        deltas = np.array([f.delta0 * np.sqrt(f.length0 / L) for f, L in zip(self.filaments, lengths)])
        # Indexation « fancy » : attribue à chaque segment le δ de son filament.
        return deltas[self._seg_owner]

    # ------------------------------------------------------------------ vitesses
    def velocity(self, X=None, targets=None):
        """Vitesse induite aux nœuds (ou en ``targets`` si fourni) par la configuration X."""
        # Configuration par défaut : positions actuelles des nœuds.
        X = self.positions() if X is None else X
        # Points d'évaluation par défaut : les nœuds eux-mêmes.
        targets = X if targets is None else targets
        # Extrémités des segments : a = nœud i, b = nœud suivant.
        a, b = X, X[self._next]
        # δ de chaque segment.
        deltas = self._segment_deltas(X)
        # Noyau « segment » : intégration exacte sur chaque segment rectiligne.
        if self.kernel == "segment":
            return biot_savart_segments(targets, a, b, self._seg_gamma, deltas)
        # Noyau « particle » : une particule au milieu de chaque segment...
        mid = 0.5 * (a + b)
        # ... de poids vectoriel α = Γ (b - a) (« [:, None] » ajoute un axe pour multiplier (S,1)×(S,3)).
        alphas = self._seg_gamma[:, None] * (b - a)
        return biot_savart_particles(targets, mid, alphas, deltas)

    def velocity_field(self, points):
        """Vitesse induite en des points quelconques (P, 3) — pour la visualisation."""
        # Évaluation en points arbitraires (np.atleast_2d accepte un point isolé).
        return self.velocity(targets=np.atleast_2d(points))

    # ------------------------------------------------------------------ intégration
    def step(self, dt):
        """Un pas de Runge–Kutta classique d'ordre 4 pour dX/dt = u(X)."""
        # État initial du pas.
        X0 = self.positions()
        # Quatre évaluations de la vitesse (pentes) du schéma RK4.
        k1 = self.velocity(X0)
        k2 = self.velocity(X0 + 0.5 * dt * k1)
        k3 = self.velocity(X0 + 0.5 * dt * k2)
        k4 = self.velocity(X0 + dt * k3)
        # Combinaison pondérée (1, 2, 2, 1)/6 : précision d'ordre 4 en dt.
        self._set_positions(X0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        # Mise à jour du δ stocké dans chaque filament (conservation du volume du cœur).
        if self.conserve_core_volume:
            for f in self.filaments:
                f.delta = f.delta0 * np.sqrt(f.length0 / f.length())
        # Avance du temps et du compteur de pas.
        self.time += dt
        self.n_steps += 1
        # Remaillage éventuel après le pas (jamais au milieu d'un pas RK4).
        if self.remesh_enabled:
            self.remesh()

    def remesh(self, force=False):
        """Rééchantillonne uniformément (spline cubique périodique) les filaments trop déformés."""
        # Indique si au moins un filament a été rééchantillonné.
        changed = False
        for f in self.filaments:
            # Longueur de chaque segment du filament.
            seg = np.linalg.norm(f.segment_vectors(), axis=1)
            # Rien à faire si tous les segments restent dans [remesh_min, remesh_max] × pas cible.
            if not force and seg.max() <= self.remesh_max * f.spacing and seg.min() >= self.remesh_min * f.spacing:
                continue
            # Abscisse curviligne cumulée de chaque nœud (0 pour le premier).
            s = np.concatenate([[0.0], np.cumsum(seg)])
            # Longueur totale du filament.
            total = s[-1]
            # np.vstack : ajoute le premier nœud à la fin (courbe fermée, valeur finale = initiale).
            closed = np.vstack([f.nodes, f.nodes[:1]])
            # Spline cubique périodique x(s) (bc_type="periodic" : dérivées continues à la jonction).
            spline = CubicSpline(s, closed, bc_type="periodic", axis=0)
            # Nouveau nombre de nœuds pour retrouver le pas cible (np.ceil = arrondi supérieur).
            n_new = max(self.min_nodes, int(np.ceil(total / f.spacing)))
            # Évaluation de la spline en n_new abscisses équiréparties (endpoint=False : sans doublon).
            f.nodes = spline(np.linspace(0.0, total, n_new, endpoint=False))
            changed = True
        # Les nombres de nœuds ont changé : on reconstruit les tableaux d'indices.
        if changed:
            self._rebuild_topology()
        return changed

    # ------------------------------------------------------------------ diagnostics
    def impulse(self):
        """Impulsion linéaire  P = ½ Σ Γ ∮ x × dl  (par unité de masse volumique ; exacte)."""
        # Positions, vecteurs segments et milieux des segments.
        X = self.positions()
        dl = X[self._next] - X
        mid = 0.5 * (X + X[self._next])
        # np.cross(mid, dl) : produit vectoriel ligne par ligne ; somme pondérée par Γ (exacte pour un
        # polygone car ∫_A^B x × dl = A × B = milieu × (B - A)).
        return 0.5 * np.sum(self._seg_gamma[:, None] * np.cross(mid, dl), axis=0)

    def energy(self):
        """Énergie cinétique régularisée  E = (1/8π) Σ_k Σ_l Γ_k Γ_l dl_k·dl_l / √(|x_k-x_l|²+δ_kl²)."""
        # Géométrie des segments.
        X = self.positions()
        dl = X[self._next] - X
        mid = 0.5 * (X + X[self._next])
        deltas = self._segment_deltas(X)
        # Vecteurs Γ dl de chaque segment.
        wdl = self._seg_gamma[:, None] * dl
        total = 0.0
        # Double somme calculée par blocs de lignes (mémoire bornée).
        for sl in _chunks(mid.shape[0], mid.shape[0]):
            # Différences de positions entre les milieux du bloc et tous les milieux : (p, M, 3).
            d = mid[sl, None, :] - mid[None, :, :]
            # Distance régularisée avec δ_kl² = moyenne des δ² des deux segments.
            dist = np.sqrt((d**2).sum(axis=2) + 0.5 * (deltas[sl, None] ** 2 + deltas[None, :] ** 2))
            # « @ » = produit matriciel : (wdl[sl] @ wdl.T)[k, l] = Γ_k Γ_l dl_k·dl_l.
            total += np.sum((wdl[sl] @ wdl.T) / dist)
        # Préfacteur 1/(8π).
        return total / (8.0 * np.pi)

    def ring_diagnostics(self, axis=(0.0, 0.0, 1.0)):
        """Pour chaque anneau (``ring_id``) : barycentre, rayon (via l'impulsion) et position axiale.

        Rayon :  R = √(|P_anneau| / (π |Γ_anneau|))  (exact pour un anneau plan circulaire).
        """
        # Axe unitaire de référence.
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        # Identifiants d'anneaux distincts, triés (set supprime les doublons, tolist → liste Python).
        ids = sorted(set(self._ring_ids.tolist()))
        centroids, radii, axial = [], [], []
        for rid in ids:
            # Filaments appartenant à cet anneau (compréhension de liste filtrée).
            fil = [f for f in self.filaments if f.ring_id == rid]
            # Accumulateurs : barycentre (numérateur, dénominateur), impulsion, circulation totale.
            num = np.zeros(3)
            den = 0.0
            P = np.zeros(3)
            gam = 0.0
            for f in fil:
                # Vecteurs segments, milieux et longueurs.
                dl = f.segment_vectors()
                mid = f.nodes + 0.5 * dl
                ln = np.linalg.norm(dl, axis=1)
                # Poids du barycentre : |Γ| × longueur du segment.
                w = abs(f.gamma) * ln
                num += (w[:, None] * mid).sum(axis=0)
                den += w.sum()
                # Impulsion du filament ½ Γ Σ milieu × dl.
                P += 0.5 * f.gamma * np.cross(mid, dl).sum(axis=0)
                gam += f.gamma
            # Barycentre de l'anneau.
            c = num / den
            centroids.append(c)
            # Rayon déduit de |P| = π R² |Γ| (NaN si circulation totale nulle).
            radii.append(np.sqrt(np.linalg.norm(P) / (np.pi * abs(gam))) if gam != 0 else np.nan)
            # Position axiale = projection du barycentre sur l'axe (np.dot = produit scalaire).
            axial.append(float(np.dot(c, axis)))
        # Conversion des listes en tableaux numpy.
        return np.array(centroids), np.array(radii), np.array(axial)

    def snapshot(self):
        # Copie des nœuds de chaque filament (instantané indépendant de l'état futur).
        return [f.nodes.copy() for f in self.filaments]

    # ------------------------------------------------------------------ boucle principale
    def run(self, t_end, dt, save_every=10, energy_every=0, axis=(0.0, 0.0, 1.0), verbose=False,
            callback=None):
        """Intègre jusqu'à ``t_end`` avec un pas ``dt`` ; sauvegarde toutes les ``save_every`` itérations.

        ``energy_every`` : calcule l'énergie (coûteuse, O(M²)) toutes les k sauvegardes (0 : jamais).
        Renvoie un :class:`SimulationHistory`.
        """
        # Conteneur des résultats.
        hist = SimulationHistory(axis=np.asarray(axis, dtype=float))
        hist.ring_ids = [f.ring_id for f in self.filaments]
        hist.gammas = [f.gamma for f in self.filaments]
        # Nombre de pas pour atteindre t_end (round : arrondi à l'entier le plus proche).
        n_steps = int(round((t_end - self.time) / dt))
        # Instant de départ pour mesurer la durée de calcul.
        t_start = time.perf_counter()

        # Fonction interne (fermeture) qui enregistre l'état courant dans « hist ».
        def record(k_save):
            hist.times.append(self.time)
            hist.frames.append(self.snapshot())
            # Barycentres, rayons et positions axiales des anneaux.
            c, r, ax = self.ring_diagnostics(axis)
            hist.diag_time.append(self.time)
            hist.impulse.append(self.impulse())
            hist.ring_centroid.append(c)
            hist.ring_radius.append(r)
            hist.ring_axial.append(ax)
            hist.n_nodes.append(self.n_nodes)
            # δ courant de chaque filament (rayon des tubes de vorticité en visualisation 3D).
            hist.deltas.append([f.delta for f in self.filaments])
            # Énergie seulement toutes les « energy_every » sauvegardes (calcul en O(M²)).
            if energy_every and k_save % energy_every == 0:
                hist.energy.append(self.energy())
            else:
                # np.nan marque une valeur non calculée.
                hist.energy.append(np.nan)

        # Enregistrement de l'état initial.
        record(0)
        k_save = 0
        # Boucle en temps.
        for it in range(1, n_steps + 1):
            # Un pas RK4 (+ remaillage éventuel).
            self.step(dt)
            # Sauvegarde périodique (et toujours au dernier pas).
            if it % save_every == 0 or it == n_steps:
                k_save += 1
                record(k_save)
                # Affichage de la progression toutes les 10 sauvegardes si « verbose ».
                if verbose and (k_save % 10 == 0 or it == n_steps):
                    el = time.perf_counter() - t_start
                    print(f"  t = {self.time:7.3f}  ({it}/{n_steps} pas, {self.n_nodes} nœuds, {el:6.1f} s)")
            # Fonction utilisateur optionnelle appelée après chaque pas.
            if callback is not None:
                callback(self)
        # Durée totale du calcul.
        hist.wall_time = time.perf_counter() - t_start
        return hist


# =============================================================================================
# Scénarios
# =============================================================================================
def single_ring_scenario(R=1.0, Gamma=1.0, delta=0.1, n_nodes=128, perturbation_mode=0,
                         perturbation_amplitude=0.0, n_core_layers=0, core_radius=None):
    """Anneau isolé, centré à l'origine, se propageant selon +z."""
    # Délègue à make_ring avec les paramètres fournis.
    return make_ring(R, Gamma, n_nodes, delta=delta, n_core_layers=n_core_layers,
                     core_radius=core_radius, perturbation_mode=perturbation_mode,
                     perturbation_amplitude=perturbation_amplitude)


def leapfrog_scenario(R=1.0, Gamma=1.0, delta=0.1, separation=0.6, n_nodes=128,
                      perturbation_amplitude=0.0, perturbation_mode=5):
    """Deux anneaux coaxiaux identiques de même sens (saute-mouton / *leapfrogging*).

    L'anneau arrière, accéléré et contracté par l'écoulement de l'anneau avant, le traverse ;
    les rôles s'échangent alors périodiquement. ``perturbation_amplitude`` > 0 ajoute une petite
    perturbation azimutale (mode ``perturbation_mode``) pour rendre la dynamique tridimensionnelle.
    """
    # Anneau A dans le plan z = 0.
    ring_a = make_ring(R, Gamma, n_nodes, center=(0.0, 0.0, 0.0), delta=delta, ring_id=0,
                       perturbation_mode=perturbation_mode, perturbation_amplitude=perturbation_amplitude)
    # Anneau B décalé de « separation » selon z, perturbation déphasée de π/m.
    ring_b = make_ring(R, Gamma, n_nodes, center=(0.0, 0.0, separation), delta=delta, ring_id=1,
                       perturbation_mode=perturbation_mode, perturbation_amplitude=perturbation_amplitude,
                       phase=np.pi / perturbation_mode if perturbation_mode else 0.0)
    # Concaténation des deux listes de filaments.
    return ring_a + ring_b


def head_on_collision_scenario(R=1.0, Gamma=1.0, delta=0.1, separation=3.0, n_nodes=128):
    """Collision frontale de deux anneaux coaxiaux de sens opposés.

    Les anneaux se ralentissent et s'étirent radialement à l'approche du plan de symétrie
    (équivalent inviscide d'un anneau approchant une paroi, par la méthode des images).
    """
    # Anneau A (Γ > 0, se déplace vers +z) en z = -separation/2.
    ring_a = make_ring(R, Gamma, n_nodes, center=(0.0, 0.0, -0.5 * separation), delta=delta, ring_id=0)
    # Anneau B (Γ < 0, se déplace vers -z) en z = +separation/2.
    ring_b = make_ring(R, -Gamma, n_nodes, center=(0.0, 0.0, 0.5 * separation), delta=delta, ring_id=1)
    return ring_a + ring_b


def offset_collision_scenario(R=1.0, Gamma=1.0, delta=0.1, separation=3.0, offset=0.8, n_nodes=128):
    """Collision décalée : deux anneaux de sens opposés dont les axes sont parallèles mais
    décalés de ``offset`` — dynamique pleinement tridimensionnelle (inclinaison, étirement)."""
    # Mêmes anneaux que la collision frontale, mais axes décalés de ±offset/2 selon x.
    ring_a = make_ring(R, Gamma, n_nodes, center=(-0.5 * offset, 0.0, -0.5 * separation), delta=delta, ring_id=0)
    ring_b = make_ring(R, -Gamma, n_nodes, center=(0.5 * offset, 0.0, 0.5 * separation), delta=delta, ring_id=1)
    return ring_a + ring_b


# =============================================================================================
# Champs dérivés pour la visualisation avancée
# =============================================================================================
def vorticity_line_density(frame, gammas, deltas, z_grid, axis=(0.0, 0.0, 1.0)):
    """Densité axiale lissée de vorticité  λ(z) = Σ_k |Γ_k| ℓ_k G_δ(z - z_k)  (diagramme de Hovmöller).

    ``frame`` : liste des tableaux de nœuds (un par filament) ; ``gammas``, ``deltas`` : Γ et δ de
    chaque filament ; G_δ est une gaussienne normalisée d'écart-type δ. ∫λ dz = Σ |Γ| × longueur.
    """
    # Axe unitaire de projection (axe de propagation des anneaux).
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    # Densité nulle au départ, de même forme que la grille axiale (float : accumulation en réels).
    lam = np.zeros_like(z_grid, dtype=float)
    for nodes, g, d in zip(frame, gammas, deltas):
        # Vecteurs segments, milieux et longueurs du filament (np.roll ferme le polygone).
        seg = np.roll(nodes, -1, axis=0) - nodes
        # Milieu de chaque segment (nœud + demi-segment).
        mid = nodes + 0.5 * seg
        # Longueur de chaque segment (norme ligne par ligne).
        ln = np.linalg.norm(seg, axis=1)
        # Position axiale des milieux (produit matriciel « @ » = produit scalaire avec l'axe).
        zk = mid @ axis
        # Somme de gaussiennes pondérées par |Γ| ℓ : tableau (Nz, n_segments) sommé sur les segments.
        kern = np.exp(-0.5 * ((z_grid[:, None] - zk[None, :]) / d) ** 2) / (np.sqrt(2.0 * np.pi) * d)
        # Poids |Γ| ℓ_k de chaque segment ([None, :] : ligne diffusée sur les Nz positions), somme sur k.
        lam += (abs(g) * ln[None, :] * kern).sum(axis=1)
    return lam


def splat_vorticity(frame, gammas, deltas, origin, spacing, shape, support=4.0):
    """Champ de vorticité 3D reconstruit sur une grille à partir des filaments (« splatting »).

    Chaque segment k devient une particule de poids α_k = Γ_k Δl_k répartie selon la distribution de
    vorticité associée au noyau de Rosenhead–Moore :
        ω(x) = Σ_k α_k ζ_δ(|x - x_k|),   ζ_δ(ρ) = 3/(4πδ³) (1 + ρ²/δ²)^{-5/2}   (∫ζ_δ dV = 1),
    tronquée à ``support`` × δ. ``origin``, ``spacing`` (3,) et ``shape`` (3,) définissent la grille
    régulière ; renvoie un tableau (3, nx, ny, nz) (composantes ω_x, ω_y, ω_z) en float32.
    """
    # Origine et pas de la grille en tableaux de réels ; dimensions converties en tuple d'entiers.
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    shape = tuple(int(n) for n in shape)
    # np.zeros((3,) + shape) : une composante par direction.
    omega = np.zeros((3,) + shape, dtype=np.float32)
    for nodes, g, d in zip(frame, gammas, deltas):
        # Segments du filament, milieux (positions des particules) et poids vectoriels α = Γ Δl.
        seg = np.roll(nodes, -1, axis=0) - nodes
        # Positions des particules (milieux des segments).
        mid = nodes + 0.5 * seg
        # Poids vectoriel α_k = Γ Δl_k (orienté selon le segment).
        alpha = g * seg
        # Rayon de troncature et constante de normalisation du noyau.
        rad = support * d
        # 3/(4πδ³) : normalisation qui donne ∫ζ_δ dV = 1.
        coef = 3.0 / (4.0 * np.pi * d**3)
        for xk, ak in zip(mid, alpha):
            # Boîte d'indices [lo, hi) couvrant la sphère de rayon rad autour de la particule
            # (np.floor/np.ceil arrondissent vers le bas/haut, np.clip borne aux dimensions de la grille).
            lo = np.clip(np.floor((xk - rad - origin) / spacing).astype(int), 0, shape)
            # Borne supérieure exclue (+1 pour inclure le dernier nœud couvert).
            hi = np.clip(np.ceil((xk + rad - origin) / spacing).astype(int) + 1, 0, shape)
            # Boîte vide (particule hors de la grille) : rien à ajouter.
            if np.any(hi <= lo):
                continue
            # Coordonnées des nœuds de la boîte selon chaque axe.
            xs = origin[0] + spacing[0] * np.arange(lo[0], hi[0])
            ys = origin[1] + spacing[1] * np.arange(lo[1], hi[1])
            zs = origin[2] + spacing[2] * np.arange(lo[2], hi[2])
            # Distance² à la particule par broadcasting (tableau 3D de la taille de la boîte).
            r2 = (xs[:, None, None] - xk[0]) ** 2 + (ys[None, :, None] - xk[1]) ** 2 + (zs[None, None, :] - xk[2]) ** 2
            # Noyau de Rosenhead–Moore ζ_δ(ρ) évalué sur la boîte.
            zeta = coef * (1.0 + r2 / d**2) ** -2.5
            # Tranche (slice) de la boîte dans la grille globale, puis ajout α ζ pour chaque composante.
            sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
            for comp in range(3):
                # (comp,) + sl : indice (composante, tranche x, tranche y, tranche z) du sous-bloc.
                omega[(comp,) + sl] += ak[comp] * zeta
    return omega
