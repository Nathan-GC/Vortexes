"""
theory.py — Modèles analytiques de référence pour les anneaux tourbillonnaires.

Contenu
-------
* Anneaux minces : vitesse de Kelvin / Saffman selon la structure du noyau, énergie, impulsion,
  correspondance avec le noyau régularisé de Rosenhead–Moore.
* Anneau visqueux (noyau gaussien qui diffuse) : formule de Saffman (1970) et correction
  d'ordre supérieur de Fukumoto & Moffatt (2000).
* Vortex sphérique de Hill : fonction de courant, vitesse, vorticité, pression, invariants.
* Anneau filamentaire : fonction de courant et vitesse induite exactes (intégrales elliptiques).
* Modèle « slug » d'un générateur piston-cylindre et nombre de formation F*.
* Modèle réduit d'anneaux minces coaxiaux (saute-mouton, collision frontale).

Conventions
-----------
Coordonnées cylindriques (r, θ, z), écoulement axisymétrique sans swirl.
Fonction de courant de Stokes ψ :   u_r = -(1/r) ∂ψ/∂z ,   u_z = (1/r) ∂ψ/∂r .
Vorticité azimutale :  ω_θ = ∂u_r/∂z - ∂u_z/∂r = -(1/r) E²ψ ,
avec  E² = ∂²/∂r² - (1/r) ∂/∂r + ∂²/∂z² .
Un anneau de circulation Γ > 0 (ω_θ > 0) se propage vers les z croissants.
Les grandeurs dynamiques (impulsion, énergie) sont données pour une masse volumique ``rho``
(par défaut 1, c.-à-d. « par unité de masse volumique »).
"""

# « from __future__ import annotations » : les annotations de type ne sont pas évaluées à
# l'exécution (compatibilité et légère accélération du chargement du module).
from __future__ import annotations

# numpy : calcul vectoriel sur tableaux (np.sqrt, np.log, np.where, np.asarray, ...).
import numpy as np
# scipy.integrate : quadratures (trapezoid, cumulative_trapezoid) et intégrateur d'EDO (solve_ivp).
# scipy.special : fonctions spéciales, ici les intégrales elliptiques complètes K (ellipk) et E (ellipe).
from scipy import integrate, special

# __all__ : liste des noms « publics » exportés par « from src.theory import * ».
__all__ = [
    "GAUSSIAN_CORE_CONSTANT",
    "FUKUMOTO_MOFFATT_COEFF",
    "core_constant",
    "core_constant_from_profile",
    "ring_speed",
    "ring_energy",
    "ring_impulse",
    "rosenhead_moore_delta",
    "rosenhead_moore_ring_speed",
    "viscous_core_radius",
    "saffman_speed",
    "saffman_viscous_speed",
    "lamb_oseen_ring_vorticity",
    "hill_stream_function",
    "hill_velocity",
    "hill_vorticity",
    "hill_pressure",
    "hill_invariants",
    "ring_stream_function",
    "ring_induced_velocity",
    "slug_model",
    "dimensionless_energy",
    "slug_dimensionless_energy",
    "formation_number",
    "CoaxialRingsModel",
]

# ---------------------------------------------------------------------------------------------
# Constantes de noyau :  U = Γ/(4πR) [ ln(8R/a) - C ]
# ---------------------------------------------------------------------------------------------
#: Constante du noyau gaussien ω ∝ exp(-s²/σ²) (a = σ) : C = 1/2 + (ln 2 - γ_E)/2 ≈ 0.5580.
# np.log(2.0) = ln 2 ≈ 0.6931 ; np.euler_gamma = constante d'Euler–Mascheroni γ_E ≈ 0.5772.
GAUSSIAN_CORE_CONSTANT = 0.5 + 0.5 * (np.log(2.0) - np.euler_gamma)

#: Coefficient de la correction de Fukumoto & Moffatt (2000) : -3.6716 νt/R² (noyau gaussien).
FUKUMOTO_MOFFATT_COEFF = 3.6716

# Dictionnaire « nom du type de noyau » → constante C (clés en minuscules ; plusieurs alias).
_CORE_CONSTANTS = {
    "uniform": 0.25,                      # Kelvin (1867) : vorticité uniforme (noyau de Rankine)
    "rankine": 0.25,                      # alias de « uniform »
    "gaussian": GAUSSIAN_CORE_CONSTANT,   # Saffman (1970) : noyau de Lamb–Oseen, a = σ
    "lamb_oseen": GAUSSIAN_CORE_CONSTANT,  # alias de « gaussian »
    "hollow": 0.5,                        # Hicks (1884) : noyau creux à pression constante
    "rosenhead_moore": 1.0,               # filament régularisé de Rosenhead–Moore, a = δ
}


def core_constant(core="uniform"):
    """Constante C de la formule  U = Γ/(4πR) [ln(8R/a) - C]  pour un type de noyau.

    ``core`` : 'uniform' (1/4), 'gaussian' (≈0.558), 'hollow' (1/2), 'rosenhead_moore' (1),
    ou directement un nombre.
    """
    # isinstance(x, (types...)) teste si x est un nombre (entier, réel Python ou réel numpy) :
    # dans ce cas l'utilisateur a fourni directement la constante C.
    if isinstance(core, (int, float, np.floating)):
        # float(...) convertit en réel Python standard.
        return float(core)
    # Sinon « core » est un nom : on le cherche dans le dictionnaire.
    try:
        # core.lower() met le nom en minuscules pour accepter « Gaussian », « GAUSSIAN », etc.
        return _CORE_CONSTANTS[core.lower()]
    # KeyError est levée si la clé n'existe pas : on la transforme en ValueError explicite.
    except KeyError as exc:  # pragma: no cover - message d'erreur explicite
        # list(dict) donne la liste des clés valides, affichée dans le message ; « from exc »
        # chaîne l'exception d'origine pour garder la trace complète.
        raise ValueError(f"Type de noyau inconnu : {core!r} ({list(_CORE_CONSTANTS)})") from exc


def core_constant_from_profile(omega, a=1.0, s_max=None, n=200001):
    """Calcule numériquement la constante C d'un profil de vorticité de cœur ω(s).

    Pour un anneau mince (a ≪ R), Saffman (1970) montre que
        U = Γ/(4πR) [ ln(8R/a) - 1/2 + A ],
        A = lim_{S→∞} [ (4π²/Γ²) ∫_0^S s v(s)² ds - ln(S/a) ],
    où v(s) est la vitesse azimutale du tourbillon rectiligne équivalent. On renvoie C = 1/2 - A.

    Paramètres
    ----------
    omega : callable, profil ω(s) (s ≥ 0), à support effectivement compact.
    a     : longueur de référence du noyau utilisée dans ln(8R/a).
    s_max : borne d'intégration (par défaut 12 a) ; au-delà, v = Γ/(2πs) (queue analytique).
    """
    # Borne supérieure d'intégration : 12 a par défaut (au-delà, un profil compact est nul).
    s_max = 12.0 * a if s_max is None else s_max
    # np.linspace(début, fin, n) : n points régulièrement espacés de 0 à s_max (inclus).
    s = np.linspace(0.0, s_max, n)
    # Évaluation vectorielle du profil de vorticité sur toute la grille.
    w = omega(s)
    # circulation cumulée Γ(s) = 2π ∫_0^s ω s' ds'
    # integrate.cumulative_trapezoid(y, x, initial=0) : intégrale cumulée par la méthode des
    # trapèzes ; « initial=0.0 » ajoute la valeur 0 en tête pour garder la même longueur que s.
    gam = 2.0 * np.pi * integrate.cumulative_trapezoid(w * s, s, initial=0.0)
    # Circulation totale du cœur = dernière valeur de la circulation cumulée.
    gamma_tot = gam[-1]
    # np.zeros_like(s) : tableau de zéros de même forme que s (v(0) = 0 sur l'axe du tourbillon).
    v = np.zeros_like(s)
    # Vitesse azimutale v(s) = Γ(s) / (2π s) (théorème de Stokes sur un cercle de rayon s) ;
    # on évite s = 0 (division par zéro) en ne remplissant que les indices 1..n-1.
    v[1:] = gam[1:] / (2.0 * np.pi * s[1:])
    # integrate.trapezoid(y, x) : intégrale définie par trapèzes de ∫ s v² ds sur [0, s_max].
    energy_int = integrate.trapezoid(s * v**2, s)
    # Constante A de Saffman ; au-delà de s_max, v = Γ/(2πs) exactement, donc la limite
    # « S → ∞ » est déjà atteinte : (4π²/Γ²)∫_{s_max}^S s v² ds = ln(S/s_max) compense le log.
    A = 4.0 * np.pi**2 / gamma_tot**2 * energy_int - np.log(s_max / a)
    # C = 1/2 - A : constante telle que U = Γ/(4πR)[ln(8R/a) - C].
    return 0.5 - A


def ring_speed(Gamma, R, a, core="uniform"):
    """Vitesse d'autopropulsion d'un anneau mince :  U = Γ/(4πR) [ln(8R/a) - C(core)].

    Formule de Kelvin (1867) pour un noyau uniforme (C = 1/4), généralisée par Saffman (1970).
    Valable pour a/R ≪ 1 (erreur O((a/R)² ln(a/R))).
    """
    # Constante C du type de noyau (voir core_constant ci-dessus).
    C = core_constant(core)
    # np.asarray(..., dtype=float) : accepte un scalaire, une liste ou un tableau et le convertit
    # en tableau numpy de réels (permet des calculs vectoriels sur plusieurs rayons à la fois).
    R = np.asarray(R, dtype=float)
    # Formule de l'anneau mince ; np.log est le logarithme népérien, appliqué élément par élément.
    return Gamma / (4.0 * np.pi * R) * (np.log(8.0 * R / a) - C)


def ring_energy(Gamma, R, a, core="uniform", rho=1.0):
    """Énergie cinétique d'un anneau mince :  E = ½ ρ Γ² R [ln(8R/a) - 2 + A],  A = 1/2 - C.

    (noyau uniforme : E = ½ρΓ²R [ln(8R/a) - 7/4]). Cohérent avec U = ∂E/∂I à Γ et volume de
    noyau fixés (principe variationnel de Kelvin–Benjamin).
    """
    # Constante A de Saffman déduite de C (A = 1/2 - C).
    A = 0.5 - core_constant(core)
    # Énergie de l'anneau mince (formule asymptotique, a ≪ R).
    return 0.5 * rho * Gamma**2 * R * (np.log(8.0 * R / a) - 2.0 + A)


def ring_impulse(Gamma, R, rho=1.0):
    """Impulsion d'un anneau mince :  I = π ρ R² Γ  (cas général : I = πρ ∬ ω_θ r² dr dz)."""
    # np.asarray(R, dtype=float) ** 2 : carré élément par élément du (des) rayon(s).
    return np.pi * rho * np.asarray(R, dtype=float) ** 2 * Gamma


def rosenhead_moore_delta(a, core="gaussian"):
    """Paramètre δ du noyau de Rosenhead–Moore reproduisant la vitesse d'un noyau physique.

    Un filament circulaire régularisé par  K_δ(x) = x / (|x|² + δ²)^{3/2}  se propage à
    U = Γ/(4πR)[ln(8R/δ) - 1]. L'égalité avec  Γ/(4πR)[ln(8R/a) - C]  donne  δ = a·exp(C - 1)
    (noyau uniforme : δ ≈ 0.472 a, Moore 1972 ; noyau gaussien : δ ≈ 0.643 σ).
    """
    # np.exp : exponentielle ; δ = a e^{C-1} égalise les deux formules de vitesse d'anneau.
    return a * np.exp(core_constant(core) - 1.0)


def rosenhead_moore_ring_speed(Gamma, R, delta, exact=False):
    """Vitesse d'un anneau filamentaire régularisé (Rosenhead–Moore).

    Asymptotique (δ ≪ R) :  U = Γ/(4πR)[ln(8R/δ) - 1].
    Exacte (``exact=True``) : le noyau RM revient à remplacer z² par z² + δ² dans le champ de
    l'anneau singulier ; la vitesse sur le filament vaut donc
        U = Γ/(2π√(4R² + δ²)) [K(m) - E(m)],   m = 4R²/(4R² + δ²).
    """
    # Version asymptotique : formule de l'anneau mince avec la constante C = 1 du noyau RM.
    if not exact:
        return ring_speed(Gamma, R, delta, core="rosenhead_moore")
    # Conversion en tableau de réels pour un calcul vectorisé.
    R = np.asarray(R, dtype=float)
    # Paramètre m = k² des intégrales elliptiques (point du filament, décalé de δ en z).
    m = 4.0 * R**2 / (4.0 * R**2 + delta**2)
    # special.ellipk(m) = K(m) = ∫_0^{π/2} dθ/√(1 - m sin²θ) ; special.ellipe(m) = E(m) =
    # ∫_0^{π/2} √(1 - m sin²θ) dθ (convention scipy : argument = paramètre m, pas le module k).
    return Gamma / (2.0 * np.pi * np.sqrt(4.0 * R**2 + delta**2)) * (special.ellipk(m) - special.ellipe(m))


def viscous_core_radius(sigma0, nu, t):
    """Rayon de cœur gaussien qui diffuse :  σ(t)² = σ0² + 4 ν t  (tourbillon de Lamb–Oseen)."""
    # np.sqrt : racine carrée élément par élément ; t peut être un tableau de temps.
    return np.sqrt(sigma0**2 + 4.0 * nu * np.asarray(t, dtype=float))


def saffman_speed(Gamma, R, sigma, second_order=False):
    """Vitesse d'un anneau à noyau gaussien ω ∝ exp(-s²/σ²) :

        U = Γ/(4πR) [ ln(8R/σ) - 0.558 ]                                  (Saffman 1970)
        U = Γ/(4πR) [ ln(8R/σ) - 0.558 - 3.6716 νt/R² ],  νt = σ²/4       (Fukumoto & Moffatt 2000)

    Le second terme (``second_order=True``) corrige l'épaississement fini du noyau.
    """
    # Conversion du rayon en tableau numpy de réels.
    R = np.asarray(R, dtype=float)
    # Crochet de la formule de Saffman : ln(8R/σ) - C_gauss.
    bracket = np.log(8.0 * R / sigma) - GAUSSIAN_CORE_CONSTANT
    # Correction de Fukumoto–Moffatt : -3.6716 νt/R² avec le temps virtuel νt = σ²/4.
    if second_order:
        bracket = bracket - FUKUMOTO_MOFFATT_COEFF * (np.asarray(sigma) ** 2 / 4.0) / R**2
    # Vitesse finale U = Γ/(4πR) × crochet.
    return Gamma / (4.0 * np.pi * R) * bracket


def saffman_viscous_speed(Gamma, R, nu, t, sigma0=0.0, second_order=False):
    """Formule de Saffman en fonction du temps :  σ² = σ0² + 4νt  puis  U = saffman_speed(...).

    Avec σ0 = 0 on retrouve  U = Γ/(4πR)[ln(8R/√(4νt)) - 0.558]  (Saffman 1970).
    """
    # Rayon de cœur à l'instant t (diffusion visqueuse).
    sigma = viscous_core_radius(sigma0, nu, t)
    # Vitesse de l'anneau pour ce rayon de cœur.
    return saffman_speed(Gamma, R, sigma, second_order=second_order)


def lamb_oseen_ring_vorticity(r, z, R, a, Gamma=1.0, z0=0.0):
    """Vorticité azimutale d'un anneau à noyau gaussien (Lamb–Oseen) de rayon R et de cœur a :

        ω_θ(r, z) = Γ/(π a²) · exp(-[(r - R)² + (z - z0)²] / a²) .
    """
    # Conversion des coordonnées en tableaux de réels (grilles 2D acceptées).
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    # Gaussienne centrée sur le cœur (r = R, z = z0), normalisée pour que ∬ω dr dz = Γ.
    return Gamma / (np.pi * a**2) * np.exp(-((r - R) ** 2 + (z - z0) ** 2) / a**2)


# ---------------------------------------------------------------------------------------------
# Vortex sphérique de Hill
# ---------------------------------------------------------------------------------------------
def _check_frame(frame):
    # Vérifie que le repère demandé est valide ; sinon lève une ValueError explicite.
    if frame not in ("body", "lab"):
        raise ValueError("frame doit valoir 'body' (repère lié au vortex) ou 'lab'")


def hill_stream_function(r, z, a=1.0, U=1.0, frame="body"):
    """Fonction de courant de Stokes du vortex sphérique de Hill (rayon a, vitesse U selon +z).

    Repère lié au vortex (écoulement uniforme -U à l'infini) :
        intérieur (ρ < a) : ψ = (3U/4) r² (1 - ρ²/a²)
        extérieur (ρ ≥ a) : ψ = -(U/2) r² (1 - a³/ρ³)          avec ρ² = r² + z².
    Repère du laboratoire : ψ_lab = ψ + U r²/2.
    """
    # Contrôle du nom de repère.
    _check_frame(frame)
    # Conversion des coordonnées en tableaux.
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    # ρ² = distance au centre du vortex au carré (coordonnées sphériques).
    rho2 = r**2 + z**2
    # Masque booléen : True à l'intérieur de la sphère de rayon a.
    inside = rho2 < a**2
    # np.errstate(...) supprime temporairement les avertissements numpy de division par zéro :
    # la formule extérieure est évaluée partout (même en ρ = 0) mais n'y est jamais utilisée.
    with np.errstate(divide="ignore", invalid="ignore"):  # branche extérieure inutilisée en ρ = 0
        # Solution extérieure : écoulement potentiel autour d'une sphère (uniforme + doublet).
        outside = -0.5 * U * r**2 * (1.0 - a**3 / rho2**1.5)
    # np.where(masque, A, B) choisit A là où le masque est vrai, B sinon (élément par élément).
    psi = np.where(inside, 0.75 * U * r**2 * (1.0 - rho2 / a**2), outside)
    # Changement de repère : ajouter l'écoulement uniforme +U (ψ_uniforme = U r²/2).
    if frame == "lab":
        psi = psi + 0.5 * U * r**2
    return psi


def hill_velocity(r, z, a=1.0, U=1.0, frame="body"):
    """Champ de vitesse (u_r, u_z) du vortex de Hill.

    Intérieur (repère du vortex) :  u_r = (3U/2a²) r z ,   u_z = (3U/2a²)(a² - 2r² - z²)
    Extérieur (repère du vortex) :  u_r = 3U a³ r z / (2ρ⁵) , u_z = -U + U a³ (2z² - r²)/(2ρ⁵)
    Repère du laboratoire : u_z → u_z + U.
    """
    # Contrôle du repère puis conversion des coordonnées.
    _check_frame(frame)
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    # ρ² et masque « intérieur de la sphère ».
    rho2 = r**2 + z**2
    inside = rho2 < a**2
    # Facteur commun des formules intérieures : 3U/(2a²).
    k = 1.5 * U / a**2
    # Branche extérieure (écoulement potentiel), calculée sans avertissement en ρ = 0.
    with np.errstate(divide="ignore", invalid="ignore"):  # branche extérieure inutilisée en ρ = 0
        # ρ⁵ = (ρ²)^{5/2}.
        rho5 = rho2**2.5
        # Vitesse radiale extérieure (dérivée de ψ par rapport à z, divisée par r).
        ur_out = 1.5 * U * a**3 * r * z / rho5
        # Vitesse axiale extérieure : écoulement uniforme -U + contribution du doublet.
        uz_out = -U + U * a**3 * (2.0 * z**2 - r**2) / (2.0 * rho5)
    # Sélection intérieur / extérieur élément par élément avec np.where.
    ur = np.where(inside, k * r * z, ur_out)
    uz = np.where(inside, k * (a**2 - 2.0 * r**2 - z**2), uz_out)
    # Dans le repère du laboratoire, le fluide lointain est au repos : on ajoute +U à u_z.
    if frame == "lab":
        uz = uz + U
    # Renvoie le couple (u_r, u_z) sous forme de tuple.
    return ur, uz


def hill_vorticity(r, z, a=1.0, U=1.0):
    """Vorticité azimutale du vortex de Hill :  ω_θ = (15U / 2a²) r  à l'intérieur, 0 dehors."""
    # Conversion des coordonnées.
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    # ω_θ/r = 15U/(2a²) = 7.5 U/a² à l'intérieur de la sphère, 0 ailleurs (écoulement irrotationnel).
    return np.where(r**2 + z**2 < a**2, 7.5 * U / a**2 * r, 0.0)


def hill_pressure(r, z, a=1.0, U=1.0, rho=1.0, p_inf=0.0):
    """Champ de pression du vortex de Hill (invariant galiléen : identique dans les deux repères).

    L'écoulement est stationnaire dans le repère du vortex ; la charge H = p/ρ + |u|²/2 vérifie
    ∇H = u × ω = -(ω_θ/r) ∇ψ. Comme ω_θ/r = K = 15U/(2a²) est constant à l'intérieur :
        extérieur :  p = p∞ + ½ρ (U² - |u|²)
        intérieur :  p = p∞ + ½ρ (U² - |u|²) - ρ K ψ
    (u et ψ exprimés dans le repère du vortex). Minimum au centre du cœur (r = a/√2, z = 0) :
    p_min = p∞ - (29/32) ρ U².
    """
    # Vitesse et fonction de courant dans le repère du vortex (écoulement stationnaire).
    ur, uz = hill_velocity(r, z, a, U, frame="body")
    psi = hill_stream_function(r, z, a, U, frame="body")
    # Conversion des coordonnées (pour construire le masque intérieur).
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    inside = r**2 + z**2 < a**2
    # K = ω_θ/r = 15U/(2a²), constant dans la sphère.
    K = 7.5 * U / a**2
    # Bernoulli (charge constante le long de tout l'écoulement potentiel extérieur).
    p = p_inf + 0.5 * rho * (U**2 - ur**2 - uz**2)
    # À l'intérieur, la charge varie comme -Kψ (H = H0 - Kψ) : on retranche ρKψ.
    return np.where(inside, p - rho * K * psi, p)


def hill_invariants(a=1.0, U=1.0, rho=1.0):
    """Invariants du vortex de Hill : Γ = 5Ua, I = 2πρa³U, E = (10/7)πρa³U², volume, etc.

    On vérifie aussi  I = (1 + 1/2) ρ V U  (masse ajoutée d'une sphère = 1/2) et l'énergie
    adimensionnelle  E/(ρ^{1/2} I^{1/2} Γ^{3/2}) ≈ 0.160 (borne inférieure de la famille de Norbury).
    """
    # Circulation = ∬ ω_θ dr dz sur le demi-disque méridien.
    Gamma = 5.0 * U * a
    # Impulsion = π ρ ∬ ω_θ r² dr dz.
    impulse = 2.0 * np.pi * rho * a**3 * U
    # Énergie cinétique totale (intérieur + extérieur), repère du laboratoire.
    energy = 10.0 / 7.0 * np.pi * rho * a**3 * U**2
    # Volume de la bulle sphérique.
    volume = 4.0 / 3.0 * np.pi * a**3
    # Dictionnaire des grandeurs caractéristiques.
    return {
        "circulation": Gamma,
        "impulse": impulse,
        "energy": energy,
        "volume": volume,
        # C_A tel que I = (1 + C_A) ρ V U → 1/2 (masse ajoutée d'une sphère).
        "added_mass_coefficient": impulse / (rho * volume * U) - 1.0,
        # Énergie adimensionnelle de Gharib et al. (≈ 0.160).
        "dimensionless_energy": dimensionless_energy(energy, impulse, Gamma, rho),
        # Vitesse du fluide au centre de la sphère dans le repère du laboratoire : 5U/2.
        "center_velocity_lab": 2.5 * U,
        # Rayon du « centre du cœur » (point d'arrêt dans le repère du vortex) : a/√2.
        "core_center_radius": a / np.sqrt(2.0),
        # Pression minimale (p_min - p∞)/(ρU²) = -29/32.
        "min_pressure_coefficient": -29.0 / 32.0,
    }


# ---------------------------------------------------------------------------------------------
# Anneau filamentaire (cœur infiniment mince) : intégrales elliptiques
# ---------------------------------------------------------------------------------------------
def ring_stream_function(r, z, R=1.0, Gamma=1.0):
    """Fonction de courant de Stokes d'un anneau filamentaire (rayon R, dans le plan z = 0) :

        ψ = (Γ/2π) (r₁ + r₂) [K(λ) - E(λ)],   λ = (r₂ - r₁)/(r₂ + r₁)   (Lamb, Hydrodynamics §161)

    r₁, r₂ : distances minimale et maximale du point au filament ; K, E intégrales elliptiques
    complètes (module λ). Loin de l'anneau : ψ ≈ Γ R² r² / (4 ρ³) (dipôle d'impulsion πΓR²).
    """
    # Conversion des coordonnées en tableaux.
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    # r1 : distance au point le plus proche du filament (dans le plan méridien, côté θ).
    r1 = np.sqrt((r - R) ** 2 + z**2)
    # r2 : distance au point le plus éloigné (côté opposé, θ + π).
    r2 = np.sqrt((r + R) ** 2 + z**2)
    # Somme des deux distances (préfacteur de la formule de Lamb).
    s = r1 + r2
    # λ = (r2 - r1)/(r2 + r1) réécrit comme 4rR/(r1 + r2)² pour éviter l'annulation numérique
    # quand r1 ≈ r2 (loin de l'anneau) ; np.maximum évite une division par zéro.
    lam = 4.0 * r * R / np.maximum(s, 1e-300) ** 2  # = (r2 - r1)/(r2 + r1) sans annulation
    # Paramètre m = λ² (np.clip borne λ dans [0, 1] pour absorber les erreurs d'arrondi).
    m = np.clip(lam, 0.0, 1.0) ** 2
    # Sur le filament m = 1 et K(1) = ∞ : on ignore l'avertissement correspondant.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Pour m très petit, K - E ≈ (π/4) m (1 + 3m/8) (développement limité, évite la perte
        # de précision de la différence de deux nombres presque égaux) ; sinon valeur exacte.
        kme = np.where(m < 1e-6, 0.25 * np.pi * m * (1.0 + 0.375 * m), special.ellipk(m) - special.ellipe(m))
    # Formule de Lamb pour ψ.
    return Gamma / (2.0 * np.pi) * s * kme


def ring_induced_velocity(r, z, R=1.0, Gamma=1.0, delta=0.0):
    """Vitesse (u_r, u_z) induite par un anneau filamentaire de rayon R situé dans le plan z = 0 :

        u_z = Γ/(2π√((r+R)²+z²)) [ K(m) + (R² - r² - z²)/((R-r)² + z²) E(m) ]
        u_r = Γ z/(2π r √((r+R)²+z²)) [ -K(m) + (R² + r² + z²)/((R-r)² + z²) E(m) ]
        m = k² = 4rR/((r+R)² + z²)

    (analogue exact du champ magnétique d'une spire). Sur l'axe : u_z = ΓR²/(2(R²+z²)^{3/2}).

    ``delta`` > 0 : anneau régularisé par le noyau de Rosenhead–Moore. Le dénominateur de
    Biot–Savart devient |x - x'|² + δ², ce qui revient exactement à évaluer le champ singulier en
    z_e = √(z² + δ²) :  u_z^δ(r, z) = u_z(r, z_e),  u_r^δ(r, z) = (z/z_e) u_r(r, z_e).
    """
    # Conversion des coordonnées.
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    # Cote « effective » z_e = √(z² + δ²) (identique à z si δ = 0).
    ze = np.sqrt(z**2 + delta**2)
    # Carré de la distance maximale au filament (évaluée avec z_e).
    r2sq = (r + R) ** 2 + ze**2
    # Carré de la distance minimale au filament (évaluée avec z_e).
    r1sq = (r - R) ** 2 + ze**2
    # Paramètre m = k² des intégrales elliptiques, borné dans [0, 1] par np.clip.
    m = np.clip(4.0 * r * R / r2sq, 0.0, 1.0)
    # Intégrales elliptiques complètes de première (K) et seconde (E) espèce.
    K = special.ellipk(m)
    E = special.ellipe(m)
    # Préfacteur commun Γ / (2π r2).
    pref = Gamma / (2.0 * np.pi * np.sqrt(r2sq))
    # Les divisions par r (axe) ou r1 (filament) peuvent produire inf/nan : avertissements masqués.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Vitesse axiale (formule de la spire de courant, z² remplacé par z_e²).
        uz = pref * (K + (R**2 - r**2 - ze**2) / r1sq * E)
        # Vitesse radiale : le facteur z (et non z_e) vient du produit vectoriel de Biot–Savart.
        ur = pref * (z / r) * (-K + (R**2 + r**2 + ze**2) / r1sq * E)
    # voisinage de l'axe : développement u_r ≈ -(r/2) ∂u_z/∂z|axe
    # Masque des points très proches de l'axe, où la formule générale perd sa précision.
    near_axis = r < 1e-4 * R
    # Développement limité exact au premier ordre en r (déduit de la continuité).
    ur_axis = 0.75 * Gamma * R**2 * r * z / (R**2 + ze**2) ** 2.5
    # Remplacement de u_r par son développement près de l'axe.
    ur = np.where(near_axis, ur_axis, ur)
    return ur, uz


# ---------------------------------------------------------------------------------------------
# Modèle « slug » et nombre de formation
# ---------------------------------------------------------------------------------------------
def slug_model(D, L, U_p, rho=1.0):
    """Modèle « slug » d'un générateur piston-cylindre (piston à vitesse U_p, course L, buse D).

    Γ = U_p L / 2 ,   I = ρ (πD²/4) L U_p ,   E = ½ ρ (πD²/4) L U_p² ,   V = (πD²/4) L .
    """
    # Section de la buse πD²/4.
    area = 0.25 * np.pi * D**2
    # Dictionnaire des grandeurs du « bouchon » de fluide éjecté.
    return {
        # Circulation : flux de circulation ½U_p² intégré sur la durée L/U_p.
        "circulation": 0.5 * U_p * L,
        # Impulsion = masse éjectée × vitesse.
        "impulse": rho * area * L * U_p,
        # Énergie cinétique = ½ masse × vitesse².
        "energy": 0.5 * rho * area * L * U_p**2,
        # Volume et masse éjectés.
        "volume": area * L,
        "mass": rho * area * L,
        # Durée de l'éjection.
        "duration": L / U_p,
    }


def dimensionless_energy(E, I, Gamma, rho=1.0):
    """Énergie adimensionnelle  α = E / (ρ^{1/2} I^{1/2} Γ^{3/2})  (Gharib, Rambod & Shariff 1998)."""
    # Combinaison sans dimension de l'énergie, de l'impulsion et de la circulation.
    return E / (np.sqrt(rho) * np.sqrt(I) * Gamma**1.5)


def slug_dimensionless_energy(L_over_D):
    """Énergie adimensionnelle délivrée par le modèle slug :  α = √(π/2) / (L/D)."""
    # Résultat exact du modèle slug (indépendant de U_p et de D).
    return np.sqrt(np.pi / 2.0) / np.asarray(L_over_D, dtype=float)


def formation_number(alpha_crit=0.33):
    """Nombre de formation limite F* = (L/D)* :

    un anneau stable (famille de Norbury) ne peut accepter une énergie adimensionnelle inférieure
    à α* ≈ 0.33 (valeur expérimentale). Le modèle slug fournissant α = √(π/2)/(L/D), l'anneau se
    détache (« pinch-off ») pour  F* = √(π/2)/α* ≈ 3.8 ≈ 4.
    """
    # Inversion de α(L/D) = √(π/2)/(L/D) pour α = α*.
    return np.sqrt(np.pi / 2.0) / alpha_crit


# ---------------------------------------------------------------------------------------------
# Modèle réduit : anneaux minces coaxiaux
# ---------------------------------------------------------------------------------------------
class CoaxialRingsModel:
    """N anneaux minces coaxiaux décrits par (R_i(t), Z_i(t)).

    dZ_i/dt = U_self(R_i, a_i) + Σ_{j≠i} u_z^{(j)}(R_i, Z_i - Z_j)
    dR_i/dt =                    Σ_{j≠i} u_r^{(j)}(R_i, Z_i - Z_j)

    avec la vitesse induite exacte d'un filament circulaire et U_self donné par la formule de
    l'anneau mince pour le type de noyau choisi. Si ``conserve_core_volume`` est vrai, le rayon de
    cœur suit  a_i² R_i = cste  (conservation du volume du tube tourbillonnaire).
    Pour ``core="rosenhead_moore"`` (défaut ``regularized=True``), vitesse propre et induction
    mutuelle utilisent le noyau régularisé exact (z² → z² + δ²) : le modèle est alors la réduction
    axisymétrique exacte de la méthode des filaments, y compris quand les anneaux se frôlent.
    Ce modèle (Dyson 1893, Hicks) sert de référence au solveur lagrangien 3D.
    """

    def __init__(self, R0, Z0, Gamma, a0, core="rosenhead_moore", conserve_core_volume=False, regularized=None):
        # np.atleast_1d : garantit un tableau d'au moins une dimension (un scalaire devient [x]),
        # pour traiter un ou plusieurs anneaux avec le même code.
        self.R0 = np.atleast_1d(np.asarray(R0, dtype=float))
        self.Z0 = np.atleast_1d(np.asarray(Z0, dtype=float))
        # Nombre d'anneaux.
        n = self.R0.size
        # np.broadcast_to(x, (n,)) étend une valeur scalaire à n anneaux (ou vérifie la taille) ;
        # .copy() crée un tableau modifiable indépendant (broadcast_to renvoie une vue en lecture seule).
        self.Gamma = np.broadcast_to(np.asarray(Gamma, dtype=float), (n,)).copy()
        self.a0 = np.broadcast_to(np.asarray(a0, dtype=float), (n,)).copy()
        # Contrôle de cohérence des tailles.
        if self.Z0.size != n:
            raise ValueError("R0 et Z0 doivent avoir la même taille")
        # Mémorisation des options du modèle.
        self.core = core
        self.conserve_core_volume = conserve_core_volume
        # Régularisation par défaut uniquement pour le noyau de Rosenhead–Moore.
        self.regularized = (core == "rosenhead_moore") if regularized is None else regularized

    # @property : « n » s'utilise comme un attribut (self.n) mais est recalculé à chaque accès.
    @property
    def n(self):
        # Nombre d'anneaux du modèle.
        return self.R0.size

    def core_radius(self, R):
        # Conservation du volume du tore : 2πR·πa² = cste ⇒ a = a0 √(R0/R).
        if self.conserve_core_volume:
            return self.a0 * np.sqrt(self.R0 / R)
        # Sinon rayon de cœur constant (cas d'un δ fixe dans la méthode des filaments).
        return self.a0

    def rhs(self, t, y):
        # Second membre de l'EDO dy/dt = f(t, y) avec y = [R_1..R_n, Z_1..Z_n].
        n = self.n
        # Découpage du vecteur d'état : n rayons puis n positions axiales.
        R, Z = y[:n], y[n:]
        # Rayon de cœur courant de chaque anneau.
        a = self.core_radius(R)
        # Vitesse propre de chaque anneau (forme close RM exacte, ou formule de l'anneau mince).
        if self.regularized and self.core == "rosenhead_moore":
            dZ = rosenhead_moore_ring_speed(self.Gamma, R, a, exact=True)
        else:
            dZ = ring_speed(self.Gamma, R, a, core=self.core)
        # np.zeros(n) : vitesses radiales initialisées à zéro (un anneau seul ne change pas de rayon).
        dR = np.zeros(n)
        # Double boucle sur les paires (i, j) : vitesse induite par l'anneau j sur l'anneau i.
        for i in range(n):
            for j in range(n):
                # Pas d'auto-induction ici (déjà comptée dans dZ).
                if i == j:
                    continue
                # Champ de l'anneau j évalué au cœur de l'anneau i (distance axiale Z_i - Z_j) ;
                # régularisé avec le δ de l'anneau source si demandé.
                ur, uz = ring_induced_velocity(R[i], Z[i] - Z[j], R=R[j], Gamma=self.Gamma[j],
                                               delta=a[j] if self.regularized else 0.0)
                # Accumulation des contributions radiale et axiale.
                dR[i] += ur
                dZ[i] += uz
        # np.concatenate : recolle [dR, dZ] en un seul vecteur de même structure que y.
        return np.concatenate([dR, dZ])

    def integrate(self, t_end, n_out=1001, method="DOP853", rtol=1e-9, atol=1e-11):
        """Intègre de 0 à ``t_end`` ; renvoie un dict {t, R (n,nt), Z (n,nt)}."""
        # Instants de sortie régulièrement espacés.
        t_eval = np.linspace(0.0, t_end, n_out)
        # Condition initiale y0 = [R0, Z0].
        y0 = np.concatenate([self.R0, self.Z0])
        # integrate.solve_ivp : intégrateur d'EDO de scipy ; « DOP853 » = Runge–Kutta explicite
        # d'ordre 8 à pas adaptatif (Dormand–Prince) ; rtol/atol = tolérances relative/absolue ;
        # t_eval = instants auxquels la solution est renvoyée (interpolation dense).
        sol = integrate.solve_ivp(self.rhs, (0.0, t_end), y0, t_eval=t_eval, method=method,
                                  rtol=rtol, atol=atol)
        # sol.success indique si l'intégration a abouti ; sinon on relaie le message d'erreur.
        if not sol.success:  # pragma: no cover
            raise RuntimeError(sol.message)
        # sol.t : instants ; sol.y : états (lignes = composantes, colonnes = instants).
        return {"t": sol.t, "R": sol.y[: self.n], "Z": sol.y[self.n:]}
