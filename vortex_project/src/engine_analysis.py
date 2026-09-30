"""
engine_analysis.py — Phase 3 : analyse physique des moteurs à vortex.

Partie 1 — Chambres de combustion à vortex (écoulement « bidirectionnel »)
------------------------------------------------------------------------
Modèle : solution eulérienne exacte de Vyas & Majdalani (2006) pour une chambre cylindrique
(rayon a, longueur L, fond fermé en z = 0, injection tangentielle près de z = L) :

    ψ = U a² κ z̃ sin(π r̃²) ,           r̃ = r/a,  z̃ = z/a,  κ = Q/(2π U a L)
    u_r = -U κ sin(π r̃²)/r̃ ,   u_θ = U/r̃ ,   u_z = 2π U κ z̃ cos(π r̃²)

Le fluide injecté descend en spirale le long de la paroi (vortex externe) vers le fond, franchit
le « manteau » r = a/√2 (où u_z = 0), puis remonte en spirale vers la tuyère (vortex interne).
On en déduit : trajectoires hélicoïdales, distribution des temps de séjour (forme close),
alimentation uniforme du cœur à travers le manteau, rétention centrifuge des gouttes, et un
modèle thermique 1D de « paroi froide » (analogue d'un refroidissement par transpiration).

Partie 2 — Propulseurs pulsés à anneaux tourbillonnaires
-------------------------------------------------------
Modèle « slug » + nombre de formation (Gharib, Rambod & Shariff 1998) + anneau mince de Kelvin
+ bulle tourbillonnaire (volume entraîné, masse ajoutée d'un sphéroïde équivalent) + impulsion de
surpression à la tuyère (Krueger & Gharib 2003, 2005) + interaction avec le train d'anneaux.
Rendement propulsif (Froude) en fonction de la fréquence d'éjection et du rapport L/D.

Toutes les grandeurs sont en unités SI.
"""

# Annotations non évaluées à l'exécution.
from __future__ import annotations

# @dataclass : génère automatiquement le constructeur à partir des attributs déclarés.
from dataclasses import dataclass

# numpy : calcul vectoriel.
import numpy as np
# scipy.integrate : quad (quadrature adaptative), trapezoid, cumulative_trapezoid, solve_ivp (EDO) ;
# scipy.optimize : brentq (recherche de racine par la méthode de Brent) ;
# scipy.special : sici (sinus et cosinus intégraux Si, Ci).
from scipy import integrate, optimize, special

# Module théorique du même paquet (nombre de formation, modèle slug, fonction de courant d'anneau).
from . import theory

# Noms publics du module.
__all__ = [
    "BidirectionalVortexChamber",
    "cold_wall_heat_flux",
    "reactor_conversion",
    "spheroid_added_mass",
    "vortex_bubble",
    "VortexRingThruster",
    "body_drag",
]


# =============================================================================================
# Partie 1 — Chambre à vortex bidirectionnel
# =============================================================================================
# @dataclass génère BidirectionalVortexChamber(a=..., L=..., U=..., Q=..., core_radius=..., rho=...).
@dataclass
class BidirectionalVortexChamber:
    """Chambre cylindrique à vortex bidirectionnel (Vyas & Majdalani, AIAA J. 44(10), 2006).

    Paramètres
    ----------
    a, L        : rayon et longueur de la chambre [m] ; fond fermé en z = 0, sortie en z = L.
    U           : vitesse tangentielle d'injection à la paroi [m/s].
    Q           : débit volumique injecté [m³/s].
    core_radius : rayon (adimensionnel, r/a) du noyau visqueux qui régularise u_θ = U/r̃ :
                  u_θ = (U/r̃)(1 - exp(-r̃²/r̃_c²)) / (1 - exp(-1/r̃_c²))  (type Burgers).
    rho         : masse volumique du gaz [kg/m³] (pour le champ de pression).
    """

    # Valeurs par défaut du cas d'exemple.
    a: float = 0.05
    L: float = 0.15
    U: float = 40.0
    Q: float = 0.05
    core_radius: float = 0.25
    rho: float = 20.0

    # ------------------------------------------------------------------ grandeurs globales
    # @property : s'utilise comme un attribut (chamber.kappa) mais est recalculé à chaque accès.
    @property
    def kappa(self):
        """Paramètre d'injection  κ = Q / (2π U a L)."""
        return self.Q / (2.0 * np.pi * self.U * self.a * self.L)

    @property
    def mantle_radius(self):
        """Rayon du manteau (u_z = 0) : r = a/√2."""
        # cos(π r̃²) = 0 pour r̃² = 1/2.
        return self.a / np.sqrt(2.0)

    @property
    def volume(self):
        # Volume du cylindre π a² L.
        return np.pi * self.a**2 * self.L

    @property
    def mean_residence_time(self):
        """Temps de séjour moyen = V/Q = a/(2κU) (tout le volume est balayé)."""
        return self.volume / self.Q

    def _swirl_factor(self, rt):
        # Facteur de régularisation du vortex libre : noyau visqueux de type Burgers,
        # normalisé pour valoir exactement 1 à la paroi (r̃ = 1).
        rc = self.core_radius
        return (1.0 - np.exp(-(rt / rc) ** 2)) / (1.0 - np.exp(-1.0 / rc**2))

    # ------------------------------------------------------------------ champ de vitesse
    def velocity(self, r, z):
        """(u_r, u_θ, u_z) en (r, z) [m, m/s]."""
        # Rayon adimensionnel r̃ = r/a, borné inférieurement pour éviter la division par zéro sur l'axe.
        rt = np.maximum(np.asarray(r, dtype=float) / self.a, 1e-12)
        # Cote adimensionnelle z̃ = z/a.
        zt = np.asarray(z, dtype=float) / self.a
        k, U = self.kappa, self.U
        # Vitesse radiale (centripète dans toute la chambre).
        ur = -U * k * np.sin(np.pi * rt**2) / rt
        # Swirl : vortex libre U/r̃ régularisé au centre.
        ut = U / rt * self._swirl_factor(rt)
        # Vitesse axiale : négative (vers le fond) pour r̃ > 1/√2, positive (vers la tuyère) au cœur.
        uz = 2.0 * np.pi * U * k * zt * np.cos(np.pi * rt**2)
        return ur, ut, uz

    def stream_function(self, r, z):
        """ψ = U a² κ z̃ sin(π r̃²)  [m³/s / 2π]."""
        rt = np.asarray(r, dtype=float) / self.a
        return self.U * self.a**2 * self.kappa * (np.asarray(z) / self.a) * np.sin(np.pi * rt**2)

    def pressure_drop(self, r):
        """p_paroi - p(r) = ρ ∫_r^a u_θ²/r' dr'  (équilibre radial, termes O(κ²) négligés)."""
        # np.atleast_1d : accepte un scalaire comme un tableau.
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty_like(r)
        # enumerate(r) fournit (indice, valeur) pour chaque rayon demandé.
        for i, ri in enumerate(r):
            # integrate.quad(f, a, b) : quadrature adaptative de ∫_a^b f ; renvoie (valeur, erreur),
            # d'où le [0]. « lambda s: ... » définit la fonction u_θ(s)²/s à intégrer ;
            # max(ri, ...) évite la borne r = 0 ; limit=200 : nombre maximal de sous-intervalles.
            out[i] = integrate.quad(lambda s: self.velocity(s, 0.0)[1] ** 2 / s, max(ri, 1e-9 * self.a), self.a,
                                    limit=200)[0]
        # Multiplication par ρ (équilibre radial dp/dr = ρ u_θ²/r).
        return self.rho * out

    def cross_mantle_flux_per_length(self):
        """Débit franchissant le manteau par unité de longueur : 2π r_m |u_r(r_m)| = Q/L (uniforme)."""
        rm = self.mantle_radius
        # « _ » : variables ignorées (seule u_r est utile ici).
        ur, _, _ = self.velocity(rm, 0.0)
        return 2.0 * np.pi * rm * abs(ur)

    def exit_swirl_number(self, n=2001):
        """Nombre de swirl en sortie S = ∫ u_z u_θ r² dr / (r_m ∫ u_z² r dr) sur le vortex interne."""
        # Rayons du vortex interne (de l'axe au manteau).
        r = np.linspace(1e-6, self.mantle_radius, n)
        # Vitesses dans le plan de sortie z = L.
        _, ut, uz = self.velocity(r, self.L)
        # Rapport flux de moment cinétique / (rayon × flux de quantité de mouvement axiale).
        return integrate.trapezoid(uz * ut * r**2, r) / (self.mantle_radius * integrate.trapezoid(uz**2 * r, r))

    # ------------------------------------------------------------------ temps de séjour
    def residence_time(self, r0):
        """Temps de séjour (forme close) du fluide entrant en z = L au rayon r0 (a/√2 < r0 < a) :

            t(r0) = a/(π κ U) · ln tan(π r̃0² / 2)

        obtenu en intégrant dt = dr/u_r entre r̃0 et le rayon de sortie r̃1 = √(1 - r̃0²).
        """
        # x0 = r̃0².
        x0 = (np.asarray(r0, dtype=float) / self.a) ** 2
        # np.tan et np.log appliqués élément par élément.
        return self.a / (np.pi * self.kappa * self.U) * np.log(np.tan(0.5 * np.pi * x0))

    def residence_time_distribution(self, n=4000):
        """Distribution des temps de séjour pondérée par le débit (lignes de courant entrantes).

        Renvoie un dict : r0, t (temps de séjour), weight (fraction de débit, somme = 1),
        mean, std, variance adimensionnelle σ_θ² = Var(t)/<t>², N_tanks = 1/σ_θ².
        """
        # x = r̃0² ∈ (1/2, 1), poids du débit ∝ |cos(π x)| dx
        # Bornes de n intervalles de x entre 1/2 (manteau) et 1 (paroi).
        edges = np.linspace(0.5, 1.0, n + 1)
        # Milieux des intervalles.
        x = 0.5 * (edges[1:] + edges[:-1])
        # Poids de débit de chaque intervalle : intégrale exacte de |cos πx| (primitive sin πx / π).
        w = np.abs(np.sin(np.pi * edges[1:]) - np.sin(np.pi * edges[:-1]))  # ∫|cos πx| dx exacte
        # Normalisation : somme des poids = 1.
        w /= w.sum()
        # Temps de séjour de chaque ligne de courant.
        t = self.a / (np.pi * self.kappa * self.U) * np.log(np.tan(0.5 * np.pi * x))
        # Échelle de temps a/(πκU).
        tau = self.a / (np.pi * self.kappa * self.U)
        # moments exacts par quadrature (l'intégrande est intégrable aux bornes)
        # f(x, p) = π|cos πx| (ln tan(πx/2))^p : densité de débit × (temps/τ)^p.
        f = lambda xx, p: np.pi * np.abs(np.cos(np.pi * xx)) * np.log(np.tan(0.5 * np.pi * xx)) ** p
        # Moments d'ordre 1 et 2 (args=(p,) transmet le paramètre p à f).
        m1 = integrate.quad(f, 0.5, 1.0, args=(1,), limit=200)[0] * tau
        m2 = integrate.quad(f, 0.5, 1.0, args=(2,), limit=200)[0] * tau**2
        # Variance du temps de séjour.
        var = m2 - m1**2
        return {"r0": self.a * np.sqrt(x), "t": t, "weight": w, "mean": m1, "std": np.sqrt(var),
                "sigma_theta2": var / m1**2, "n_tanks": m1**2 / var, "plug_flow_time": self.volume / self.Q}

    # ------------------------------------------------------------------ trajectoires 3D
    def trajectory(self, r0, theta0=0.0, n_points=4000):
        """Trajectoire hélicoïdale d'une particule fluide entrant en (r0, θ0, z = L).

        Intégration de  dr/dt = u_r, dθ/dt = u_θ/r, dz/dt = u_z  jusqu'au retour en z = L
        (sortie par le vortex interne). Renvoie t, r, θ, z, x, y et : temps de séjour, longueur
        de parcours, nombre de tours, profondeur de pénétration (z minimal), rayon de sortie.
        """
        a, L = self.a, self.L
        # Le point d'injection doit être dans le vortex externe.
        if not (self.mantle_radius < r0 < a):
            raise ValueError("r0 doit être compris entre a/√2 et a (vortex externe)")
        # Temps de séjour exact (sert à dimensionner l'intégration).
        t_res = float(self.residence_time(r0))
        # Départ juste sous z = L pour que l'événement de sortie ne se déclenche pas à t = 0.
        z_start = L * (1.0 - 1e-9)

        # Second membre de l'EDO en coordonnées cylindriques : y = (r, θ, z).
        def rhs(t, y):
            r, th, z = y
            ur, ut, uz = self.velocity(r, z)
            return [ur, ut / r, uz]

        # Fonction d'événement : s'annule quand la particule revient en z = L.
        def exit_event(t, y):
            return y[2] - L

        # Attributs lus par solve_ivp : arrêt à l'événement (terminal) et seulement quand z - L
        # passe du négatif au positif (direction = +1).
        exit_event.terminal = True
        exit_event.direction = 1.0
        # Instants de sortie de la solution.
        t_eval = np.linspace(0.0, t_res, n_points)
        # integrate.solve_ivp : intégration Runge–Kutta d'ordre 8 (DOP853) à tolérances serrées.
        sol = integrate.solve_ivp(rhs, (0.0, 1.05 * t_res), [r0, theta0, z_start], t_eval=t_eval,
                                  events=exit_event, rtol=1e-9, atol=1e-12, method="DOP853")
        # Composantes de la trajectoire (sol.y est un tableau (3, n)).
        r, th, z = sol.y
        # Passage en coordonnées cartésiennes pour le tracé 3D.
        x, y = r * np.cos(th), r * np.sin(th)
        # Vitesse le long de la trajectoire.
        ur, ut, uz = self.velocity(r, z)
        speed = np.sqrt(ur**2 + ut**2 + uz**2)
        # Longueur parcourue = ∫ |u| dt (méthode des trapèzes).
        path = integrate.trapezoid(speed, sol.t)
        return {
            "t": sol.t, "r": r, "theta": th, "z": z, "x": x, "y": y,
            "residence_time": t_res,
            "path_length": path,
            "path_ratio": path / L,
            # Nombre de tours = angle total parcouru / 2π.
            "revolutions": (th[-1] - th[0]) / (2.0 * np.pi),
            # Profondeur de pénétration vers le fond.
            "z_min": z.min(),
            # Rayon de sortie théorique r̃1 = √(1 - r̃0²).
            "r_exit": a * np.sqrt(1.0 - (r0 / a) ** 2),
            # Dérive relative de ψ (np.ptp = max - min) : contrôle de précision de l'intégration.
            "psi_drift": np.ptp(self.stream_function(r, z)) / max(self.stream_function(r0, L), 1e-300),
        }

    # ------------------------------------------------------------------ gouttes
    def droplet_cutoff_diameter(self, mu_gas=5e-5, rho_liquid=800.0):
        """Diamètre de coupure centrifuge au manteau (loi de Stokes) :

        une goutte de temps de réponse τ_p = ρ_l d²/(18 μ) dérive vers l'extérieur à τ_p u_θ²/r ;
        l'équilibre avec l'aspiration radiale du gaz |u_r| donne au manteau (vortex libre)
            τ_p = κ a / (2U)   ⇒   d_c = √(9 μ κ a / (ρ_l U)) .
        Les gouttes plus grosses restent piégées dans le vortex externe (ou atteignent la paroi).
        """
        return np.sqrt(9.0 * mu_gas * self.kappa * self.a / (rho_liquid * self.U))


def cold_wall_heat_flux(peclet, kappa, r_flame=1.0 / np.sqrt(2.0), n=2001):
    """Modèle thermique 1D de la « paroi froide » d'une chambre à vortex bidirectionnel.

    Entre la zone de combustion (manteau, r̃_f, température T_f) et la paroi (T_w), l'énergie est
    transportée par diffusion turbulente α_t et par la convection radiale *centripète* du vortex
    externe ũ_r = -κ sin(πr̃²)/r̃ :
        ũ_r dT/dr̃ = (1/Pe) (1/r̃) d/dr̃ (r̃ dT/dr̃),     Pe = U a / α_t .
    Avec q = r̃ dT/dr̃ :  q(r̃) = q_w exp(g(r̃)),  g(r̃) = (Pe κ / 2) [Si(π) - Si(π r̃²)]  (Si = sinus
    intégral), d'où le rapport entre flux pariétal et flux de conduction pure (Pe κ = 0) :
        Φ = q_w / q_w0 = ln(1/r̃_f) / ∫_{r̃_f}^1 e^{g(s)} ds / s .
    L'aspiration radiale agit comme une transpiration. Pour Pe κ ≫ 1, l'intégrale est dominée par
    le voisinage de r̃_f (méthode de Laplace) :
        Φ ≈ ln(1/r̃_f) · Pe κ sin(π r̃_f²) · exp(-g(r̃_f)),   soit  Φ ≈ ln√2 · Pe κ · e^{-0.2406 Pe κ}
    au manteau (r̃_f = 1/√2).

    Renvoie un dict : ratio Φ, son asymptote, profil adimensionnel θ(r̃) = (T - T_w)/(T_f - T_w),
    r̃ et g(r̃).
    """
    # Grille radiale adimensionnelle de la flamme (r̃_f) à la paroi (1).
    s = np.linspace(r_flame, 1.0, n)
    # special.sici(x) renvoie le couple (Si(x), Ci(x)) ; [0] sélectionne le sinus intégral Si.
    si_pi = special.sici(np.pi)[0]
    # Nombre de Péclet radial Pe·κ.
    pk = peclet * kappa
    # Exposant g(r̃) de la solution exacte.
    g = 0.5 * pk * (si_pi - special.sici(np.pi * s**2)[0])
    # Intégrande e^{g}/s normalisé par e^{g(r̃_f)} (évite les dépassements numériques pour Pe grand).
    integrand = np.exp(g - g[0]) / s  # normalisé par e^{g_f} pour éviter tout dépassement
    # Intégrale cumulée depuis la paroi : [::-1] inverse l'ordre des tableaux, on intègre de 1 vers r̃,
    # puis on remet dans l'ordre ; le facteur -1 corrige le sens d'intégration.
    cum = integrate.cumulative_trapezoid(integrand[::-1], s[::-1], initial=0.0)[::-1] * -1.0
    # Intégrale totale de r̃_f à 1.
    J = cum[0]
    # Profil de température adimensionnel (1 à la flamme, 0 à la paroi).
    theta = cum / J
    # ln(1/r̃_f) : intégrale de conduction pure (Pe κ = 0).
    log_rf = np.log(1.0 / r_flame)
    # Rapport des flux (on réintroduit le facteur e^{-g_f} retiré par la normalisation).
    ratio = log_rf * np.exp(-g[0]) / J
    # Asymptote de Laplace pour Pe κ grand (1 si pas de convection).
    asym = log_rf * pk * np.sin(np.pi * r_flame**2) * np.exp(-g[0]) if pk > 0 else 1.0
    return {"ratio": ratio, "ratio_asymptotic": asym, "r": s, "theta": theta, "g": g, "radial_peclet": pk}


def reactor_conversion(damkohler, chamber=None):
    """Conversion d'une réaction d'ordre 1 (temps chimique τ_c, Da = <t>/τ_c) selon le réacteur :

    piston : 1 - e^{-Da} ;  parfaitement agité : Da/(1+Da) ;
    vortex bidirectionnel (écoulement ségrégé, DTS exacte) : ∫ E(t) (1 - e^{-t/τ_c}) dt.
    """
    # Nombres de Damköhler demandés, en tableau.
    da = np.atleast_1d(np.asarray(damkohler, dtype=float))
    # Chambre par défaut si aucune n'est fournie.
    chamber = BidirectionalVortexChamber() if chamber is None else chamber
    # Échelle de temps de la forme close t(x) = τ ln tan(πx/2).
    tau = chamber.a / (np.pi * chamber.kappa * chamber.U)
    tmean = chamber.mean_residence_time
    vortex = np.empty_like(da)
    for i, d in enumerate(da):
        # Temps chimique correspondant au Damköhler d.
        tc = tmean / d
        # Densité de débit × conversion de chaque ligne de courant (modèle d'écoulement ségrégé).
        f = lambda x: np.pi * np.abs(np.cos(np.pi * x)) * (1.0 - np.exp(-tau * np.log(np.tan(0.5 * np.pi * x)) / tc))
        # Conversion moyenne = intégrale pondérée par le débit.
        vortex[i] = integrate.quad(f, 0.5, 1.0, limit=200)[0]
    return {"Da": da, "plug": 1.0 - np.exp(-da), "cstr": da / (1.0 + da), "vortex": vortex}


# =============================================================================================
# Partie 2 — Propulseurs pulsés à anneaux tourbillonnaires
# =============================================================================================
def spheroid_added_mass(c, a_eq):
    """Coefficient de masse ajoutée C_A d'un sphéroïde se déplaçant selon son axe de symétrie.

    c : demi-axe polaire (direction du mouvement) ; a_eq : demi-axe équatorial.
    C_A = α₀/(2 - α₀) (Lamb, Hydrodynamics §373) avec
      aplati (c < a_eq), e = √(1 - c²/a²) :  α₀ = (2/e²)[1 - √(1-e²) arcsin(e)/e]
      allongé (c > a_eq), e = √(1 - a²/c²) : α₀ = (2(1-e²)/e³)[½ ln((1+e)/(1-e)) - e]
    Sphère : 1/2 ; disque : → ∞ (masse ajoutée finie 8ρa³/3).
    """
    # Rapport d'aspect du sphéroïde.
    ratio = c / a_eq
    # Sphère (formules générales indéterminées en e = 0) : C_A = 1/2.
    if abs(ratio - 1.0) < 1e-6:
        return 0.5
    if ratio < 1.0:
        # Sphéroïde aplati : excentricité et coefficient de Lamb (np.arcsin = arc sinus).
        e = np.sqrt(1.0 - ratio**2)
        alpha0 = 2.0 / e**2 * (1.0 - np.sqrt(1.0 - e**2) * np.arcsin(e) / e)
    else:
        # Sphéroïde allongé.
        e = np.sqrt(1.0 - 1.0 / ratio**2)
        alpha0 = 2.0 * (1.0 - e**2) / e**3 * (0.5 * np.log((1.0 + e) / (1.0 - e)) - e)
    return alpha0 / (2.0 - alpha0)


def vortex_bubble(Gamma, R, U, n=500):
    """« Bulle » tourbillonnaire d'un anneau mince (fluide transporté avec l'anneau).

    Dans le repère de l'anneau, ψ' = ψ_anneau(r, z) - U r²/2 ; la bulle est la région ψ' > 0
    délimitée par la ligne de courant séparatrice. Renvoie volume, demi-axes (axial, radial),
    points d'arrêt sur l'axe et coefficient de masse ajoutée du sphéroïde équivalent.
    """
    # Étendue de la fenêtre de calcul (3 R de part et d'autre).
    ext = 3.0 * R
    # Centres de mailles en r (n valeurs) et en z (2n valeurs, fenêtre symétrique).
    r = (np.arange(n) + 0.5) * ext / n
    z = (np.arange(2 * n) + 0.5) * 2.0 * ext / (2 * n) - ext
    # Grilles 2D (indice 0 = r, indice 1 = z).
    Rg, Zg = np.meshgrid(r, z, indexing="ij")
    # ψ' dans le repère de l'anneau (singulier sur le filament : avertissements masqués).
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = theory.ring_stream_function(Rg, Zg, R=R, Gamma=Gamma) - 0.5 * U * Rg**2
    # Masque des mailles intérieures à la bulle.
    inside = psi > 0.0
    dr, dz = r[1] - r[0], z[1] - z[0]
    # Volume de révolution = Σ 2π r dr dz sur les mailles intérieures.
    volume = np.sum(2.0 * np.pi * Rg[inside]) * dr * dz
    # Demi-axes axial et radial de la bulle (inside.any() : au moins une maille intérieure).
    half_z = np.abs(Zg[inside]).max() + 0.5 * dz if inside.any() else 0.0
    half_r = Rg[inside].max() + 0.5 * dr if inside.any() else 0.0
    # points d'arrêt sur l'axe : u_axe(z) = Γ R²/(2 (R² + z²)^{3/2}) = U
    q = (Gamma * R**2 / (2.0 * U)) ** (2.0 / 3.0) - R**2
    # Bulle attachée à l'axe si des points d'arrêt existent (anneau pas trop mince).
    attached = q > 0
    z_stag = np.sqrt(q) if attached else 0.0
    return {"volume": volume, "half_axial": half_z, "half_radial": half_r, "z_stagnation": z_stag,
            "attached_to_axis": bool(attached), "added_mass": spheroid_added_mass(half_z, half_r)}


def body_drag(U, diameter=0.1, cd=0.3, rho=1000.0):
    """Traînée d'un corps profilé : D = ½ ρ C_D (π d²/4) U²."""
    return 0.5 * rho * cd * 0.25 * np.pi * diameter**2 * U**2


@dataclass
class VortexRingThruster:
    """Propulseur pulsé à anneaux tourbillonnaires (générateur piston-cylindre + tuyère D).

    Paramètres
    ----------
    D          : diamètre de sortie de la tuyère [m].
    rho        : masse volumique du fluide [kg/m³].
    alpha_crit : énergie adimensionnelle critique de détachement (≈ 0.33, Gharib et al. 1998).
    beta_max   : fraction maximale d'impulsion de surpression I_op/I_slug atteinte à L/D = F*
                 (ordre de grandeur des mesures de Krueger & Gharib 2003, 2005 : ~0.4).
    coflow_slope : pente de F*(λ) = F*₀ (1 - c λ) avec λ = U_coflow/U_j ; c ≈ 0.42 reproduit
                 F* ∈ [3, 4] pour λ < 0.6 (Krueger, Dabiri & Gharib 2006).
    entrainment_fraction : fraction de fluide ambiant dans la bulle de l'anneau (≈ 0.3–0.4
                 mesuré par Dabiri & Gharib 2004) ; le modèle slug seul ne peut pas la prédire.
    overpressure_work_factor : coût énergétique de la surpression = facteur × U_j × I_op
                 (1 : surpression générée à vitesse de piston constante ; ~0.5 si elle l'est
                 surtout pendant l'accélération du piston).
    """

    # Paramètres par défaut (tuyère de 2 cm dans l'eau).
    D: float = 0.02
    rho: float = 1000.0
    alpha_crit: float = 0.33
    beta_max: float = 0.40
    coflow_slope: float = 0.42
    entrainment_fraction: float = 0.35
    overpressure_work_factor: float = 1.0

    @property
    def area(self):
        # Section de sortie de la tuyère.
        return 0.25 * np.pi * self.D**2

    # ------------------------------------------------------------------ formation
    def formation_number(self, coflow_ratio=0.0):
        """F*(λ) = √(π/2)/α* · (1 - c λ)."""
        # np.clip borne λ dans [0, 0.95] (le modèle n'est pas défini au-delà).
        return theory.formation_number(self.alpha_crit) * (1.0 - self.coflow_slope * np.clip(coflow_ratio, 0.0, 0.95))

    def slug(self, L_over_D, U_j):
        """Modèle slug d'une impulsion (Γ, I, E, V, m, durée)."""
        return theory.slug_model(self.D, L_over_D * self.D, U_j, self.rho)

    def overpressure_fraction(self, L_over_D, coflow_ratio=0.0):
        """Fraction d'impulsion de surpression β = I_op / I_slug.

        Fermeture semi-empirique : pendant la formation, l'impulsion de surpression est celle de
        la masse ajoutée du fluide ambiant accéléré par l'anneau (Krueger & Gharib 2003) ; elle
        croît comme (volume de l'anneau ∝ L) × (vitesse ∝ Γ/R ∝ L), donc I_op ∝ L², puis se fige
        au détachement (L/D = F*). D'où β = β_max (L/D)/F* avant F*, β_max F*/(L/D) après, avec un
        maximum à L/D = F*. Le co-écoulement λ réduit la formation par le facteur (1-λ)²(1+λ)
        (flux de circulation ∝ 1-λ², volume de l'anneau ∝ 1-λ).
        """
        # Co-écoulement borné et nombre de formation correspondant.
        lam = np.clip(coflow_ratio, 0.0, 0.95)
        fs = self.formation_number(lam)
        ld = np.asarray(L_over_D, dtype=float)
        # np.where : croissance linéaire avant F*, décroissance en 1/(L/D) après.
        beta0 = np.where(ld <= fs, self.beta_max * ld / fs, self.beta_max * fs / ld)
        # Réduction par le co-écoulement.
        return beta0 * (1.0 - lam) ** 2 * (1.0 + lam)

    def leading_ring(self, L_over_D, U_j, U_co=0.0, bubble=True):
        """Anneau de tête formé par une impulsion (modèle slug avec co-écoulement U_co).

        L_r = min(L, F* D) ;  Γ = U_j L_r (1 - λ²)/2 ;  R = D/√(2(1+λ))  (via I = πρR²Γ) ;
        cœur a déduit de l'énergie E = ½ρΓ²R[ln(8R/a) - 7/4] (noyau uniforme) ;
        U_self = Γ/(4πR)[ln(8R/a) - 1/4] (Kelvin). Si ``bubble`` : volume V_b de la bulle,
        masse ajoutée C_A ρ V_b du sphéroïde équivalent, fluide entraîné ε V_b (ε empirique) et
        bouclage de l'impulsion  I_bulle = (1 + C_A) ρ V_b U_self  comparée à l'impulsion slug.
        """
        # Rapport de co-écoulement et nombre de formation associé.
        lam = U_co / U_j
        fs = self.formation_number(lam)
        # Course totale et partie de la course qui alimente l'anneau (le reste forme le jet traînant).
        L = L_over_D * self.D
        L_r = min(L, fs * self.D)
        # Circulation (flux ½(U_j² - U_co²) pendant la durée L_r/U_j).
        Gamma = 0.5 * U_j * L_r * (1.0 - lam**2)
        # Rayon de l'anneau déduit de l'impulsion I = πρR²Γ.
        R = self.D / np.sqrt(2.0 * (1.0 + lam))
        # Énergie cinétique en excès transmise à l'anneau.
        energy = 0.5 * self.rho * self.area * L_r * (U_j - U_co) ** 2
        # ln(8R/a) tiré de E = ½ρΓ²R[ln(8R/a) - 7/4].
        log_term = 2.0 * energy / (self.rho * Gamma**2 * R) + 1.75
        # Rayon de cœur correspondant.
        a = 8.0 * R * np.exp(-log_term)
        # Vitesse propre (formule de Kelvin, noyau uniforme).
        U_self = Gamma / (4.0 * np.pi * R) * (log_term - 0.25)
        # Dictionnaire des propriétés de l'anneau.
        out = {
            "L_ring": L_r, "L_trailing": max(0.0, L - L_r), "formation_number": fs,
            "circulation": Gamma, "radius": R, "core_radius": a, "core_ratio": a / R,
            "thin_core_valid": a / R < 0.5, "U_self": U_self, "energy": energy,
            "impulse": self.rho * self.area * L_r * (U_j - U_co), "slug_volume": self.area * L_r,
        }
        if bubble:
            # Bulle transportée par l'anneau (séparatrice ψ' = 0).
            b = vortex_bubble(Gamma, R, U_self)
            eps = self.entrainment_fraction
            # Impulsion de la bulle + masse ajoutée : (1 + C_A) ρ V_b U.
            bubble_impulse = (1.0 + b["added_mass"]) * self.rho * b["volume"] * U_self
            # dict.update ajoute ces entrées au dictionnaire existant.
            out.update({
                "bubble_volume": b["volume"], "bubble_to_slug_volume": b["volume"] / out["slug_volume"],
                "bubble_half_axial": b["half_axial"], "bubble_half_radial": b["half_radial"],
                "added_mass": b["added_mass"], "added_mass_kg": b["added_mass"] * self.rho * b["volume"],
                "entrained_volume": eps * b["volume"],
                "ejected_volume_in_ring": (1.0 - eps) * b["volume"],
                "total_moving_mass": (1.0 + b["added_mass"]) * self.rho * b["volume"],
                "bubble_impulse": bubble_impulse,
                # Rapport de bouclage : impulsion bulle / impulsion fournie par le piston.
                "impulse_closure": bubble_impulse / out["impulse"],
            })
        return out

    # ------------------------------------------------------------------ interaction du train
    # @staticmethod : méthode sans « self », appelable depuis la classe ou une instance.
    @staticmethod
    def ring_train_induced_velocity(Gamma, R, U_conv, period, T_form, L_ring, n_rings=400):
        """Vitesse axiale induite au centre de la tuyère par les anneaux précédents.

        L'anneau k (k ≥ 1) formé k périodes plus tôt se trouve, à mi-formation de l'anneau
        courant, à la distance d_k = L_r/2 + U_conv (k T - T_form/2) ;
        u = Σ_k Γ R² / (2 (R² + d_k²)^{3/2}).
        """
        # Numéros des anneaux précédents 1..n_rings.
        k = np.arange(1, n_rings + 1)
        # Distance de chaque anneau à la tuyère (np.maximum : au moins L_r/2).
        d = 0.5 * L_ring + U_conv * (k * period - 0.5 * T_form)
        d = np.maximum(d, 0.5 * L_ring)
        # Somme des vitesses sur l'axe d'anneaux de circulation Γ et de rayon R.
        return float(np.sum(Gamma * R**2 / (2.0 * (R**2 + d**2) ** 1.5)))

    # ------------------------------------------------------------------ performances
    def performance(self, L_over_D, frequency, U_j, U_inf=0.0, interaction=True, n_iter=8):
        """Poussée, puissance et rendement propulsif moyens d'un fonctionnement pulsé.

        Poussée   T = f [ m_s (U_j - U∞) + I_op ]
        Puissance P = f [ ½ m_s (U_j² - U∞²) + c_w U_j I_op ]   (travail du piston contre la
                  surpression, c_w = ``overpressure_work_factor``)
        Rendement η = T U∞ / P ; référence : jet continu de même débit moyen et même poussée,
        η_s = 2 U∞ / (U∞ + U_js),  U_js = U∞ + T/ṁ.
        """
        # Course, durée d'éjection, période et masse éjectée par impulsion.
        L = L_over_D * self.D
        T_p = L / U_j
        period = 1.0 / frequency
        m_s = self.rho * self.area * L
        # Vitesse induite par le train d'anneaux (point fixe itératif).
        u_ind = 0.0
        ring = None
        # « _ » : compteur de boucle inutilisé ; une seule itération sans interaction.
        for _ in range(n_iter if interaction else 1):
            # Co-écoulement effectif vu par l'anneau en formation (vitesse du véhicule + induction).
            lam_eff = min((U_inf + u_ind) / U_j, 0.95)
            ring = self.leading_ring(L_over_D, U_j, U_co=lam_eff * U_j, bubble=False)
            if not interaction:
                break
            # Durée de formation de l'anneau.
            T_form = ring["L_ring"] / U_j
            # Nouvelle estimation de la vitesse induite (anneaux convectés à U∞ + U_self).
            new = self.ring_train_induced_velocity(ring["circulation"], ring["radius"],
                                                   U_inf + ring["U_self"], period, T_form, ring["L_ring"])
            # Convergence du point fixe.
            if abs(new - u_ind) < 1e-6 * U_j:
                u_ind = new
                break
            u_ind = new
        # Co-écoulement final et fraction de surpression correspondante.
        lam_eff = min((U_inf + u_ind) / U_j, 0.95)
        beta = float(self.overpressure_fraction(L_over_D, lam_eff))
        # Impulsion de surpression par impulsion.
        I_op = beta * m_s * U_j
        # Poussée moyenne (quantité de mouvement nette par impulsion × fréquence).
        thrust = frequency * (m_s * (U_j - U_inf) + I_op)
        # Puissance moyenne (énergie cinétique fournie + travail contre la surpression).
        power = frequency * (0.5 * m_s * (U_j**2 - U_inf**2) + self.overpressure_work_factor * U_j * I_op)
        # Débit massique moyen.
        mdot = frequency * m_s
        # Rendement propulsif (nul si le véhicule est à l'arrêt).
        eta = thrust * U_inf / power if U_inf > 0 else 0.0
        # Jet continu équivalent : même débit, même poussée → vitesse U_js et rendement de Froude.
        U_js = U_inf + thrust / mdot
        eta_s = 2.0 * U_inf / (U_inf + U_js) if U_inf > 0 else 0.0
        return {
            "L_over_D": L_over_D, "frequency": frequency, "U_j": U_j, "U_inf": U_inf,
            "duty_cycle": frequency * T_p, "feasible": frequency * T_p <= 1.0,
            "mass_flow": mdot, "thrust": thrust, "power": power, "efficiency": eta,
            "efficiency_steady": eta_s, "gain": eta / eta_s if eta_s > 0 else np.nan,
            "beta": beta, "induced_velocity": u_ind, "coflow_ratio": lam_eff,
            "formation_number": self.formation_number(lam_eff), "ring": ring,
            "impulse_per_pulse": m_s * (U_j - U_inf) + I_op,
        }

    def jet_velocity_for_thrust(self, L_over_D, frequency, U_inf, thrust, U_max=200.0, **kw):
        """Vitesse de jet U_j donnant la poussée moyenne ``thrust`` (NaN si infaisable : rapport
        cyclique f L/U_j > 1 requis, ou poussée inaccessible)."""
        L = L_over_D * self.D
        # Borne inférieure : jet plus rapide que le véhicule et rapport cyclique f L/U_j ≤ 1.
        U_lo = max(U_inf * (1.0 + 1e-6), frequency * L * (1.0 + 1e-9), 1e-6)
        # Fonction dont on cherche la racine : poussée(U_j) - poussée requise ;
        # **kw transmet les options supplémentaires (interaction, n_iter) à performance().
        f = lambda Uj: self.performance(L_over_D, frequency, Uj, U_inf, **kw)["thrust"] - thrust
        # Si la poussée minimale réalisable dépasse déjà la cible, le point est infaisable.
        f_lo = f(U_lo)
        if f_lo > 0:
            return np.nan
        # Poussée requise inaccessible même à U_max.
        if f(U_max) < 0:
            return np.nan
        # optimize.brentq : racine de f sur [U_lo, U_max] (changement de signe garanti).
        return optimize.brentq(f, U_lo, U_max, xtol=1e-10 * U_max, rtol=1e-10)

    def efficiency_map(self, L_over_D_values, frequencies, U_inf, thrust, **kw):
        """Carte η(L/D, f) à poussée imposée (vitesse de croisière U_inf, traînée = poussée)."""
        LD = np.asarray(L_over_D_values, dtype=float)
        F = np.asarray(frequencies, dtype=float)
        shape = (LD.size, F.size)
        # Grandeurs cartographiées.
        keys = ("efficiency", "efficiency_steady", "gain", "U_j", "duty_cycle", "beta", "induced_velocity")
        # np.full(forme, np.nan) : cartes initialisées à NaN (= point infaisable).
        out = {k: np.full(shape, np.nan) for k in keys}
        for i, ld in enumerate(LD):
            for j, f in enumerate(F):
                # Vitesse de jet nécessaire pour la poussée requise.
                Uj = self.jet_velocity_for_thrust(ld, f, U_inf, thrust, **kw)
                # np.isfinite : faux pour NaN (point infaisable, on laisse NaN).
                if not np.isfinite(Uj):
                    continue
                perf = self.performance(ld, f, Uj, U_inf, **kw)
                for k in keys:
                    out[k][i, j] = perf[k] if k != "U_j" else Uj
        # Ajout des axes de la carte et des conditions de fonctionnement.
        out.update({"L_over_D": LD, "frequency": F, "U_inf": U_inf, "thrust": thrust})
        return out
