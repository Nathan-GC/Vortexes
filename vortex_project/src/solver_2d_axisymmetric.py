"""
solver_2d_axisymmetric.py — Approche B : Navier–Stokes incompressible axisymétrique en (r, z).

Équations (sans swirl, ρ = 1)
-----------------------------
    ∂u/∂t + (1/r)∂(r u²)/∂r + ∂(u w)/∂z = -∂p/∂r + ν [ ∂/∂r((1/r)∂(r u)/∂r) + ∂²u/∂z² ]
    ∂w/∂t + (1/r)∂(r u w)/∂r + ∂(w²)/∂z = -∂p/∂z + ν [ (1/r)∂/∂r(r ∂w/∂r)  + ∂²w/∂z² ]
    (1/r)∂(r u)/∂r + ∂w/∂z = 0
u = u_r, w = u_z. Le terme visqueux radial ∂/∂r((1/r)∂(ru)/∂r) = ∇²u - u/r².

Discrétisation
--------------
* Grille décalée MAC : p au centre des cellules, u sur les faces radiales, w sur les faces
  axiales, ω_θ et ψ aux coins. Maillage structuré, uniforme en z (périodique) et uniforme ou
  géométriquement étiré en r (repousse la paroi extérieure à moindre coût).
* Différences finies centrées d'ordre 2 en forme conservative.
* Conditions aux limites : axe r = 0 (u = 0, flux nuls), paroi glissante en r = L_r
  (u = 0, ∂w/∂r = 0, ω = 0), périodicité en z.
* Méthode de projection (Chorin) appliquée à chaque étage d'un Runge–Kutta 4 :
      du/dt = P[F(u)],   P = I - G L⁻¹ D,   L = D G,
  ce qui maintient div u = 0 à la précision machine. L'équation de Poisson pour la pression est
  résolue exactement : FFT en z, puis décomposition spectrale précalculée de l'opérateur radial
  tridiagonal (symétrisé), ce qui évite toute boucle Python.
* Initialisation : vorticité de Lamb–Oseen (ou de Hill) aux coins, puis résolution de
  l'équation « de Poisson » pour la fonction de courant de Stokes
      E²ψ = ∂²ψ/∂r² - (1/r)∂ψ/∂r + ∂²ψ/∂z² = -r ω_θ ,
  et u = -(1/r)∂ψ/∂z, w = (1/r)∂ψ/∂r : le champ initial est exactement à divergence nulle et
  son rotationnel discret reproduit exactement la vorticité imposée.

Rappel sur le stockage des tableaux (indice i en r, indice j en z)
------------------------------------------------------------------
    u[i, j]   (Nr+1, Nz) : vitesse radiale en (rf[i], zc[j])   — faces radiales
    w[i, j]   (Nr,   Nz) : vitesse axiale  en (rc[i], zf[j])   — faces axiales
    p[i, j]   (Nr,   Nz) : pression        en (rc[i], zc[j])   — centres
    ω, ψ      (Nr+1, Nz) : aux coins          (rf[i], zf[j])
"""

# Annotations non évaluées à l'exécution.
from __future__ import annotations

# itertools.permutations : appariement des anneaux d'un instant à l'autre (suivi d'identité).
import itertools
# time.perf_counter() : mesure du temps de calcul.
import time

# numpy : tableaux, FFT (np.fft), algèbre linéaire (np.linalg).
import numpy as np
# scipy.ndimage.maximum_filter : détection des maxima locaux de vorticité (un pic par cœur d'anneau).
from scipy import ndimage

# Module théorique du même paquet (profil de Lamb–Oseen, vortex de Hill, formule de Saffman).
from . import theory

# Noms publics du module.
__all__ = ["AxisymmetricGrid", "RadialSpectralSolver", "AxisymmetricNavierStokesSolver", "run_vortex_ring",
           "run_coaxial_rings", "velocity_gradient_tensor", "q_criterion", "lambda2_criterion",
           "dissipation_rate", "enstrophy_density", "volume_integral"]


def _zp(a):
    """a[:, j+1] (périodique)."""
    # np.roll(a, -1, axis=1) décale les colonnes d'un cran vers la gauche : la colonne j contient
    # l'ancienne colonne j+1 et la dernière reçoit la première (périodicité en z).
    return np.roll(a, -1, axis=1)


def _zm(a):
    """a[:, j-1] (périodique)."""
    # Décalage d'un cran vers la droite : la colonne j contient l'ancienne colonne j-1.
    return np.roll(a, 1, axis=1)


# =============================================================================================
# Grille
# =============================================================================================
class AxisymmetricGrid:
    """Grille MAC en (r, z).

    Attributs principaux
    --------------------
    rf  (Nr+1,) : positions des faces radiales (rf[0] = 0 = axe, rf[Nr] = L_r = paroi).
    rc  (Nr,)   : centres des cellules.  drc = largeurs des cellules.
    drf (Nr+1,) : distances entre centres voisins de part et d'autre de chaque face
                  (demi-cellules aux bords) = largeur des volumes duaux centrés sur les faces.
    zf, zc      : faces et centres en z (Nz points, période L_z).
    """

    def __init__(self, r_faces, Lz, Nz, z0=0.0):
        # Positions des faces radiales converties en tableau de réels.
        rf = np.asarray(r_faces, dtype=float)
        # np.diff(rf) = rf[i+1] - rf[i] ; on exige rf[0] = 0 (axe) et des faces strictement croissantes.
        if rf[0] != 0.0 or np.any(np.diff(rf) <= 0):
            raise ValueError("r_faces doit commencer à 0 et être strictement croissant")
        self.rf = rf
        # Nombre de cellules radiales.
        self.Nr = rf.size - 1
        # Centres des cellules : milieu de deux faces consécutives.
        self.rc = 0.5 * (rf[1:] + rf[:-1])
        # Largeur de chaque cellule.
        self.drc = np.diff(rf)
        # np.empty : tableau non initialisé, rempli juste après.
        self.drf = np.empty(self.Nr + 1)
        # Faces internes : distance entre les deux centres voisins.
        self.drf[1:-1] = np.diff(self.rc)
        # Face de l'axe : demi-cellule [0, rc[0]].
        self.drf[0] = self.rc[0]
        # Face de la paroi : demi-cellule [rc[-1], L_r].
        self.drf[-1] = rf[-1] - self.rc[-1]
        # Rayon de la paroi extérieure.
        self.Lr = float(rf[-1])
        # Nombre de points, période et pas en z.
        self.Nz = int(Nz)
        self.Lz = float(Lz)
        self.dz = self.Lz / self.Nz
        # Origine du domaine en z.
        self.z0 = float(z0)
        # Positions des faces axiales zf[j] = z0 + j Δz (np.arange(N) = 0..N-1).
        self.zf = self.z0 + self.dz * np.arange(self.Nz)
        # Centres des cellules en z : décalés d'un demi-pas.
        self.zc = self.zf + 0.5 * self.dz

    # @classmethod : constructeur alternatif appelé sur la classe (AxisymmetricGrid.uniform(...)).
    @classmethod
    def uniform(cls, Lr, Nr, Lz, Nz, z0=0.0):
        # Faces radiales régulièrement espacées de 0 à Lr (np.linspace inclut les deux bornes).
        return cls(np.linspace(0.0, Lr, Nr + 1), Lz, Nz, z0)

    @classmethod
    def stretched(cls, dr0, r_uniform, Lr, growth, Lz, Nz, z0=0.0):
        """Pas uniforme ``dr0`` jusqu'à ``r_uniform`` puis croissance géométrique jusqu'à ``Lr``."""
        # Nombre de cellules de la zone uniforme (round : entier le plus proche).
        n_u = int(round(r_uniform / dr0))
        # Faces de la zone uniforme : 0, dr0, 2 dr0, ..., n_u dr0 (list pour pouvoir ajouter).
        faces = list(dr0 * np.arange(n_u + 1))
        dr = dr0
        # Zone étirée : chaque cellule est « growth » fois plus large que la précédente.
        while faces[-1] < Lr:
            dr *= growth
            faces.append(faces[-1] + dr)
        # Conversion en tableau numpy.
        faces = np.array(faces)
        # ajuste la zone étirée pour finir exactement en Lr
        r_u = faces[n_u]
        # Mise à l'échelle affine de la zone étirée : la dernière face tombe exactement sur Lr.
        faces[n_u:] = r_u + (faces[n_u:] - r_u) * (Lr - r_u) / (faces[-1] - r_u)
        return cls(faces, Lz, Nz, z0)

    def corner_mesh(self):
        """(R, Z) aux coins (Nr+1, Nz) — emplacement de ω et ψ."""
        # np.meshgrid(..., indexing="ij") : grilles 2D où le premier indice suit r et le second z.
        return np.meshgrid(self.rf, self.zf, indexing="ij")

    def center_mesh(self):
        # Grilles 2D des centres de cellules (emplacement de p et des champs interpolés).
        return np.meshgrid(self.rc, self.zc, indexing="ij")

    def periodic_dz(self, z, z_ref):
        """Écart z - z_ref ramené dans [-Lz/2, Lz/2)."""
        # « % » (modulo) ramène l'écart dans [0, Lz) ; le décalage de Lz/2 le centre sur 0.
        return (z - z_ref + 0.5 * self.Lz) % self.Lz - 0.5 * self.Lz

    def __repr__(self):
        # Représentation textuelle affichée par print(grille).
        return (f"AxisymmetricGrid(Nr={self.Nr}, Nz={self.Nz}, Lr={self.Lr:.3g}, Lz={self.Lz:.3g}, "
                f"dr_min={self.drc.min():.4g}, dr_max={self.drc.max():.4g}, dz={self.dz:.4g})")


# =============================================================================================
# Solveur elliptique : FFT en z + décomposition spectrale de l'opérateur radial
# =============================================================================================
class RadialSpectralSolver:
    """Résout  (L_r + δ_zz) x = f  avec L_r tridiagonal (radial) et δ_zz périodique (FFT).

    L_r est donné par ses coefficients ``lower`` (de x[i-1]) et ``upper`` (de x[i+1]) et sa
    diagonale ``diag``. Il doit être symétrisable par une matrice diagonale W :
    W L_r symétrique. On précalcule  S = W^{1/2} L_r W^{-1/2} = Q Λ Qᵀ, puis pour chaque mode
    de Fourier k (valeur propre -λ_k de δ_zz) :  x̂ = W^{-1/2} Q (Λ - λ_k)^{-1} Qᵀ W^{1/2} f̂.
    Les couples (Λ ≈ 0, k = 0) (mode nul d'un problème de Neumann) sont annulés (jauge).
    """

    def __init__(self, lower, diag, upper, weight, Nz, dz):
        # Taille de l'opérateur radial.
        n = diag.size
        # Matrice tridiagonale dense : np.diag(v, k) place le vecteur v sur la k-ième diagonale
        # (k = 1 : sur-diagonale, k = -1 : sous-diagonale).
        A = np.diag(diag) + np.diag(upper[:-1], 1) + np.diag(lower[1:], -1)
        # Racine des poids de symétrisation W.
        ws = np.sqrt(weight)
        # S = W^{1/2} A W^{-1/2} (produit par des matrices diagonales = mise à l'échelle des lignes/colonnes).
        S = (ws[:, None] * A) / ws[None, :]
        # Symétrisation exacte (supprime les erreurs d'arrondi) : S ← (S + Sᵀ)/2.
        S = 0.5 * (S + S.T)
        # np.linalg.eigh : valeurs propres (croissantes) et vecteurs propres orthonormés d'une
        # matrice symétrique réelle : S = Q diag(Λ) Qᵀ.
        self.eigval, self.Q = np.linalg.eigh(S)
        self.ws = ws
        self.Nz = Nz
        # Nombres d'onde k = 0..Nz/2 conservés par la FFT réelle (rfft).
        k = np.arange(Nz // 2 + 1)
        # Valeur propre de la différence seconde centrée périodique : -λ_k = -(4/Δz²) sin²(πk/Nz).
        self.lam = (4.0 / dz**2) * np.sin(np.pi * k / Nz) ** 2
        # Dénominateurs (Λ_i - λ_k) pour tous les couples (mode radial i, mode de Fourier k).
        denom = self.eigval[:, None] - self.lam[None, :]
        # Échelle de référence pour détecter les dénominateurs numériquement nuls.
        scale = np.abs(self.eigval).max() + self.lam.max()
        # Masque du mode nul (problème de Neumann : pression définie à une constante près).
        null = np.abs(denom) < 1e-10 * scale
        # Inverses 1/(Λ - λ) ; 0 pour le mode nul (fixe la jauge : pression de moyenne nulle).
        # Le np.where intérieur évite la division par zéro avant la sélection.
        self.inv = np.where(null, 0.0, 1.0 / np.where(null, 1.0, denom))
        # Indique si l'opérateur possède un mode nul (vrai pour la pression, faux pour ψ).
        self.has_null_mode = bool(null.any())
        self.n = n

    def solve(self, f):
        # np.fft.rfft(f, axis=1) : transformée de Fourier réelle le long de z (Nz/2 + 1 modes complexes).
        fh = np.fft.rfft(f, axis=1)
        # Projection sur les vecteurs propres : Qᵀ W^{1/2} f̂ (« @ » = produit matriciel).
        coef = self.Q.T @ (self.ws[:, None] * fh)
        # Division par (Λ - λ_k) puis retour à la base physique en r : W^{-1/2} Q (...).
        xh = (self.Q @ (coef * self.inv)) / self.ws[:, None]
        # np.fft.irfft(..., n=Nz) : transformée inverse vers Nz points réels en z.
        return np.fft.irfft(xh, n=self.Nz, axis=1)


# =============================================================================================
# Solveur Navier–Stokes
# =============================================================================================
class AxisymmetricNavierStokesSolver:
    """Navier–Stokes incompressible axisymétrique (sans swirl) par projection sur grille MAC.

    Paramètres
    ----------
    grid : :class:`AxisymmetricGrid`.
    nu   : viscosité cinématique (Re_Γ = Γ/ν).
    cfl  : nombre de Courant cible  (|u|/Δr + |w|/Δz) Δt  (RK4 centré stable jusqu'à ~2.8).
    """

    def __init__(self, grid, nu, cfl=0.8, dt_max=None):
        # Grille et paramètres physiques/numériques.
        self.g = grid
        self.nu = float(nu)
        self.cfl = float(cfl)
        self.dt_max = dt_max
        g = grid
        # np.zeros(forme) : champs initialement nuls aux emplacements MAC.
        self.u = np.zeros((g.Nr + 1, g.Nz))
        self.w = np.zeros((g.Nr, g.Nz))
        self.p = np.zeros((g.Nr, g.Nz))
        # Temps courant, nombre de pas et position de référence de l'anneau (suivi périodique).
        self.time = 0.0
        self.n_steps = 0
        self._z_ref = None
        # géométrie en colonnes (broadcast sur z)
        # « [:, None] » transforme un vecteur (N,) en colonne (N, 1) pour multiplier des tableaux (N, Nz).
        self._rf = g.rf[:, None]
        self._rc = g.rc[:, None]
        self._vol_c = (g.rc * g.drc)[:, None]          # r dr des cellules
        # Distances entre centres pour les faces internes i = 1..Nr-1.
        self._drf_in = g.drf[1:-1, None]
        # Somme des largeurs des deux cellules voisines de chaque face interne.
        wr = g.drc[:-1] + g.drc[1:]
        self._wl = (g.drc[1:] / wr)[:, None]           # poids d'interpolation de w vers les faces
        self._wr = (g.drc[:-1] / wr)[:, None]
        # Précalcul des solveurs de Poisson (pression) et de Stokes (fonction de courant).
        self._build_elliptic_solvers()

    # ------------------------------------------------------------------ solveurs elliptiques
    def _build_elliptic_solvers(self):
        g = self.g
        n = g.Nr
        # Pression : centres, Neumann à l'axe (rf=0) et à la paroi.
        # Volume (par unité d'angle et de Δz) de chaque cellule : r dr.
        vol = g.rc * g.drc
        # Conductance r/Δr de chaque face ; nulle à l'axe (rf = 0) et à la paroi (Neumann).
        a_face = np.zeros(n + 1)
        a_face[1:-1] = g.rf[1:-1] / g.drf[1:-1]         # conductance des faces internes
        # Coefficients de p[i-1] (lower) et p[i+1] (upper) du laplacien radial discret D·G.
        lower = np.zeros(n)
        upper = np.zeros(n)
        lower[1:] = a_face[1:-1] / vol[1:]
        upper[:-1] = a_face[1:-1] / vol[:-1]
        # Diagonale : somme des flux sortants (lignes de somme nulle → mode constant nul).
        diag = -(lower + upper)
        # Solveur spectral ; poids de symétrisation W = volumes des cellules.
        self._p_solver = RadialSpectralSolver(lower, diag, upper, vol, g.Nz, g.dz)
        # Fonction de courant : coins internes i = 1..Nr-1, Dirichlet en 0 et en L_r.
        # Indices des coins internes.
        i = np.arange(1, n)
        # 1/(r dr) des cellules situées de part et d'autre du coin i.
        cm = 1.0 / (g.rc[i - 1] * g.drc[i - 1])
        cp = 1.0 / (g.rc[i] * g.drc[i])
        # Facteur r/Δr au coin.
        fac = g.rf[i] / g.drf[i]
        # Coefficients de ψ[i-1] et ψ[i+1] de l'opérateur de Stokes discret r ∂/∂r((1/r) ∂ψ/∂r).
        lo = fac * cm
        up = fac * cp
        dg = -(lo + up)
        # Coefficient du dernier coin vers la paroi (sert à imposer ψ = psi_wall au second membre).
        self._psi_upper_last = up[-1]
        # Copies modifiables puis suppression des couplages vers les nœuds de Dirichlet
        # (ψ = 0 sur l'axe, ψ = psi_wall sur la paroi, traités hors de la matrice).
        lo = lo.copy()
        up = up.copy()
        lo[0] = 0.0
        up[-1] = 0.0
        # Poids de symétrisation W = Δr/r au coin.
        self._psi_solver = RadialSpectralSolver(lo, dg, up, g.drf[i] / g.rf[i], g.Nz, g.dz)

    # ------------------------------------------------------------------ opérateurs discrets
    def divergence(self, u, w):
        # Flux radial r u aux faces.
        ru = self._rf * u
        # (1/(r dr)) [ (r u)_{i+1} - (r u)_i ] + (w_{j+1} - w_j)/Δz, aux centres des cellules.
        return (ru[1:] - ru[:-1]) / self._vol_c + (_zp(w) - w) / self.g.dz

    def gradient(self, phi):
        # Composante radiale du gradient aux faces internes ; nulle à l'axe et à la paroi.
        gu = np.zeros((self.g.Nr + 1, self.g.Nz))
        gu[1:-1] = (phi[1:] - phi[:-1]) / self._drf_in
        # Composante axiale aux faces axiales : (φ_j - φ_{j-1})/Δz (périodique).
        gw = (phi - _zm(phi)) / self.g.dz
        return gu, gw

    def solve_pressure(self, div):
        # Résolution exacte de L φ = div (L = D G) par le solveur spectral.
        return self._p_solver.solve(div)

    def project(self, fu, fw):
        """Projection de Leray discrète : retire le gradient qui rend (fu, fw) à divergence nulle."""
        # φ tel que L φ = D(f).
        phi = self.solve_pressure(self.divergence(fu, fw))
        # G φ.
        gu, gw = self.gradient(phi)
        # f - G φ est à divergence nulle car D(f - Gφ) = D f - L φ = 0 ; φ joue le rôle de p/ρ.
        return fu - gu, fw - gw, phi

    def vorticity(self, u=None, w=None):
        """ω_θ = ∂u/∂z - ∂w/∂r aux coins (nulle sur l'axe et sur la paroi glissante)."""
        # Champs courants par défaut.
        u = self.u if u is None else u
        w = self.w if w is None else w
        # np.zeros_like(u) : même forme que u (Nr+1, Nz) ; les lignes 0 et Nr restent nulles.
        om = np.zeros_like(u)
        # Coins internes : (u_j - u_{j-1})/Δz - (w_i - w_{i-1})/Δr.
        om[1:-1] = (u[1:-1] - _zm(u[1:-1])) / self.g.dz - (w[1:] - w[:-1]) / self._drf_in
        return om

    def stream_function(self, w=None):
        """ψ aux coins par intégration exacte de w = (1/r)∂ψ/∂r depuis l'axe (ψ = 0 sur l'axe)."""
        w = self.w if w is None else w
        # ψ = 0 sur l'axe (ligne 0).
        psi = np.zeros((self.g.Nr + 1, self.g.Nz))
        # np.cumsum(..., axis=0) : somme cumulée en r de r dr w, soit ψ(r) = ∫_0^r r' w dr'.
        psi[1:] = np.cumsum(self._vol_c * w, axis=0)
        return psi

    def solve_stream_function(self, omega, psi_wall=0.0):
        """Résout  E²_h ψ = -r ω  (coins internes), ψ = 0 sur l'axe, ψ = psi_wall sur la paroi."""
        g = self.g
        # Second membre -r ω aux coins internes (.copy() pour pouvoir le modifier).
        rhs = -(self._rf * omega)[1:-1].copy()
        # Transfert de la condition de Dirichlet de la paroi au second membre.
        rhs[-1] -= self._psi_upper_last * psi_wall
        # Assemblage de ψ : axe = 0, coins internes résolus, paroi = psi_wall.
        psi = np.zeros((g.Nr + 1, g.Nz))
        psi[1:-1] = self._psi_solver.solve(rhs)
        psi[-1] = psi_wall
        return psi

    def velocity_from_stream_function(self, psi):
        g = self.g
        # u = -(1/r) ∂ψ/∂z aux faces radiales internes (u = 0 sur l'axe et la paroi).
        u = np.zeros((g.Nr + 1, g.Nz))
        u[1:-1] = -(_zp(psi[1:-1]) - psi[1:-1]) / (self._rf[1:-1] * g.dz)
        # w = (1/r) ∂ψ/∂r aux faces axiales : (ψ_{i+1} - ψ_i)/(r dr).
        w = (psi[1:] - psi[:-1]) / self._vol_c
        return u, w

    # ------------------------------------------------------------------ initialisation
    def set_vorticity(self, omega, psi_wall=0.0):
        """Impose un champ de vorticité aux coins ; en déduit ψ puis (u, w)."""
        # Copie en tableau de réels (ne modifie pas l'argument de l'appelant).
        omega = np.array(omega, dtype=float)
        # ω = 0 sur l'axe (symétrie) et sur la paroi glissante.
        omega[0] = 0.0
        omega[-1] = 0.0
        # Résolution de E²ψ = -rω puis calcul de la vitesse : champ exactement solénoïdal.
        psi = self.solve_stream_function(omega, psi_wall)
        self.u, self.w = self.velocity_from_stream_function(psi)
        # Pression remise à zéro (« [:] » modifie le tableau en place).
        self.p[:] = 0.0

    def init_lamb_oseen_ring(self, R=1.0, a=0.2, Gamma=1.0, z0=None):
        """Anneau de Lamb–Oseen : ω = Γ/(πa²) exp(-[(r-R)² + (z-z0)²]/a²) (périodique en z)."""
        g = self.g
        # Position initiale par défaut : au quart du domaine.
        z0 = g.z0 + 0.25 * g.Lz if z0 is None else z0
        # Coordonnées des coins.
        Rg, Zg = g.corner_mesh()
        # Écart axial périodique au centre de l'anneau (l'anneau « voit » sa plus proche image).
        dzp = g.periodic_dz(Zg, z0)
        # Vorticité gaussienne de l'anneau.
        omega = theory.lamb_oseen_ring_vorticity(Rg, dzp, R, a, Gamma, 0.0)
        # Construction du champ de vitesse correspondant.
        self.set_vorticity(omega)
        # Référence de suivi de l'anneau et paramètres mémorisés.
        self._z_ref = z0
        self.ring_params = {"R0": R, "a0": a, "Gamma": Gamma, "z0": z0, "type": "lamb_oseen"}

    def init_hill_vortex(self, a=1.0, U=1.0, z0=None):
        """Vortex sphérique de Hill (repère du laboratoire) : ω = 15U r / (2a²) dans la sphère."""
        g = self.g
        z0 = g.z0 + 0.25 * g.Lz if z0 is None else z0
        Rg, Zg = g.corner_mesh()
        # Vorticité analytique de Hill aux coins.
        omega = theory.hill_vorticity(Rg, g.periodic_dz(Zg, z0), a, U)
        self.set_vorticity(omega)
        self._z_ref = z0
        self.ring_params = {"a": a, "U": U, "z0": z0, "Gamma": 5.0 * U * a, "type": "hill"}

    def init_rings(self, rings):
        """Superpose plusieurs anneaux de Lamb–Oseen coaxiaux (saute-mouton visqueux, collision...).

        ``rings`` : liste de dictionnaires {"R": rayon, "a": rayon de cœur, "Gamma": circulation,
        "z0": position axiale initiale}.
        """
        # Raccourci vers la grille.
        g = self.g
        # Coordonnées des coins (emplacement de ω).
        Rg, Zg = g.corner_mesh()
        # np.zeros_like : vorticité totale initialement nulle, de même forme que la grille des coins.
        omega = np.zeros_like(Rg)
        # Somme des vorticités gaussiennes de chaque anneau (écart axial périodique à son centre).
        for ring in rings:
            # Tourbillon gaussien de rayon R, cœur a et circulation Γ, centré en z0 (dernier argument 0 :
            # centre à l'origine de l'écart axial fourni) ; « += » accumule les anneaux.
            omega += theory.lamb_oseen_ring_vorticity(Rg, g.periodic_dz(Zg, ring["z0"]), ring["R"], ring["a"],
                                                      ring["Gamma"], 0.0)
        # Champ de vitesse solénoïdal correspondant (résolution de E²ψ = -rω).
        self.set_vorticity(omega)
        # Référence axiale globale = position moyenne des anneaux ; état de suivi réinitialisé.
        self._z_ref = float(np.mean([ring["z0"] for ring in rings]))
        # Aucun suivi antérieur : le prochain appel de track_rings classera les anneaux par z croissant.
        self._ring_state = None
        # dict(r) copie chaque dictionnaire (l'appelant peut modifier les siens sans effet ici).
        self.ring_params = {"type": "coaxial_rings", "rings": [dict(r) for r in rings]}

    def current_pressure(self):
        """Pression p/ρ cohérente avec le champ de vitesse courant (une projection du second membre)."""
        # _projected_rhs renvoie (F_u, F_w, φ) : φ résout L φ = D(F(u)), c'est la pression à l'instant courant.
        _, _, phi = self._projected_rhs(self.u, self.w)
        return phi

    def track_rings(self, n_rings=2, peak_threshold=0.2, mass_threshold=0.02, separation=0.2):
        """Identifie et suit individuellement ``n_rings`` anneaux dans le champ de vorticité.

        1. pics : maxima locaux de ω (ndimage.maximum_filter) supérieurs à ``peak_threshold`` × ω_max,
           distants d'au moins ``separation`` ;
        2. partition : chaque coin où ω > ``mass_threshold`` × ω_max est attribué au pic le plus proche ;
        3. moments de Saffman de chaque partie : Γ_i, R_i² = ∬ω r²/∬ω, Z_i = ∬ω r² z/∬ω r² ;
        4. identité : appariement avec l'instant précédent (distance minimale), Z déroulé (continu).
        Renvoie un tableau (n_rings, 3) de colonnes (Γ, R, Z) ; l'anneau 0 est initialement le plus en arrière.
        """
        g = self.g
        # Vorticité aux coins et aires duales associées.
        om = self.vorticity()
        area = g.drf[:, None] * g.dz
        om_max = om.max()
        # Demi-largeur (en mailles) du voisinage de recherche des maxima : environ 0.15 unité de longueur.
        h = min(g.drc.min(), g.dz)
        k = max(1, int(round(0.15 / h)))
        # ndimage.maximum_filter : maximum glissant sur une fenêtre (2k+1)² ; modes par axe :
        # « nearest » en r (pas de périodicité), « wrap » en z (domaine périodique).
        mx = ndimage.maximum_filter(om, size=2 * k + 1, mode=("nearest", "wrap"))
        # np.argwhere : indices (i, j) des coins qui sont des maxima locaux assez intenses.
        cand = np.argwhere((om >= mx) & (om > peak_threshold * om_max))
        # Tri des candidats par vorticité décroissante (np.argsort sur l'opposé).
        order = np.argsort(-om[cand[:, 0], cand[:, 1]])
        seeds = []
        for idx in order:
            i, j = cand[idx]
            r0, z0 = g.rf[i], g.zf[j]
            # On ne garde que des pics bien séparés (distance euclidienne, z périodique).
            if all(np.hypot(r0 - rs, g.periodic_dz(z0, zs)) > separation for rs, zs in seeds):
                seeds.append((r0, z0))
            if len(seeds) == n_rings:
                break
        # Nombre de pics distincts réellement trouvés (< n_rings : cœurs fusionnés ou trop proches).
        self._last_n_peaks = len(seeds)
        # État précédent (None au premier appel) ; getattr évite une erreur si l'attribut n'existe pas.
        prev = getattr(self, "_ring_state", None)
        # Pics fusionnés (anneaux trop proches) : on complète avec les positions précédentes.
        if len(seeds) < n_rings and prev is not None:
            for Rp, Zp in prev:
                if len(seeds) == n_rings:
                    break
                if all(np.hypot(Rp - rs, g.periodic_dz(Zp, zs)) > separation for rs, zs in seeds):
                    seeds.append((Rp, Zp))
        # Moins de pics que d'anneaux : résultat indéfini (NaN).
        if len(seeds) < n_rings:
            return np.full((n_rings, 3), np.nan)
        # Partition du domaine : distance² de chaque coin à chaque pic (np.stack empile les n tableaux).
        Rg, Zg = g.corner_mesh()
        d2 = np.stack([(Rg - rs) ** 2 + g.periodic_dz(Zg, zs) ** 2 for rs, zs in seeds])
        # np.argmin(..., axis=0) : numéro du pic le plus proche pour chaque coin.
        owner = np.argmin(d2, axis=0)
        # Seuls les coins de vorticité significative sont comptés.
        mask = om > mass_threshold * om_max
        rings = []
        for n, (rs, zs) in enumerate(seeds):
            # Vorticité × aire restreinte à la partie n.
            wA = np.where(mask & (owner == n), om, 0.0) * area
            G = wA.sum()
            m2 = (wA * Rg**2).sum()
            # Position axiale relative au pic (écart périodique), puis moment de Saffman.
            dzp = g.periodic_dz(Zg, zs)
            Z = zs + (wA * Rg**2 * dzp).sum() / m2
            rings.append([G, np.sqrt(m2 / G), Z])
        if prev is not None and len(prev) == len(rings):
            # itertools.permutations : toutes les façons d'associer les nouveaux anneaux aux anciens ;
            # min(..., key=...) choisit l'association de déplacement total minimal.
            best = min(itertools.permutations(range(len(rings))),
                       key=lambda perm: sum(np.hypot(rings[p][1] - prev[q][0], g.periodic_dz(rings[p][2], prev[q][1]))
                                            for q, p in enumerate(perm)))
            rings = [rings[p] for p in best]
            # Déroulement de Z : on ajoute au Z précédent le petit déplacement périodique.
            for q, rg in enumerate(rings):
                rg[2] = prev[q][1] + g.periodic_dz(rg[2], prev[q][1])
        else:
            # Premier appel : anneau 0 = le plus en arrière (Z croissant).
            rings.sort(key=lambda rg: rg[2])
        # Mémorisation (R, Z) pour l'appel suivant.
        self._ring_state = [(rg[1], rg[2]) for rg in rings]
        return np.array(rings)

    # ------------------------------------------------------------------ second membre
    def rhs(self, u, w):
        """Second membre F(u) (convection conservative + diffusion), sans pression."""
        # Raccourcis locaux.
        g = self.g
        nu = self.nu
        dz = g.dz
        rf, rc = self._rf, self._rc
        # Second membre radial (nul sur l'axe et la paroi où u est imposé).
        fu = np.zeros_like(u)
        # valeurs aux centres
        # u interpolé au centre des cellules (moyenne des deux faces radiales).
        uc = 0.5 * (u[:-1] + u[1:])
        # w interpolé au centre des cellules (moyenne des faces axiales j et j+1).
        wc = 0.5 * (w + _zp(w))
        # valeurs aux coins (r_f, z_f)
        # u interpolé en z vers le coin (moyenne de j-1 et j).
        u_cz = 0.5 * (u + _zm(u))
        # w interpolé en r vers le coin (moyenne pondérée par les largeurs de cellules).
        w_cr = np.empty_like(u)
        w_cr[1:-1] = self._wl * w[:-1] + self._wr * w[1:]
        w_cr[0] = w[0]
        w_cr[-1] = w[-1]
        # Flux croisé u·w aux coins.
        uw = u_cz * w_cr  # nul sur l'axe et la paroi (u = 0)
        # --- quantité de mouvement radiale, faces internes
        # (1/r) ∂(r u²)/∂r : différence du flux r u² entre centres voisins.
        rcu2 = rc * uc * uc
        adv = (rcu2[1:] - rcu2[:-1]) / (rf[1:-1] * self._drf_in)
        # + ∂(u w)/∂z : différence du flux u w entre les coins j+1 et j.
        adv += (_zp(uw[1:-1]) - uw[1:-1]) / dz
        # Terme visqueux radial ∂/∂r[(1/r) ∂(r u)/∂r] (= ∇²u - u/r²).
        dru = (rf[1:] * u[1:] - rf[:-1] * u[:-1]) / self._vol_c      # (1/r) ∂(r u)/∂r aux centres
        visc = (dru[1:] - dru[:-1]) / self._drf_in
        # + ∂²u/∂z² (différence seconde centrée périodique).
        ui = u[1:-1]
        visc += (_zp(ui) - 2.0 * ui + _zm(ui)) / dz**2
        # F_u = ν × diffusion - convection.
        fu[1:-1] = nu * visc - adv
        # --- quantité de mouvement axiale, faces axiales
        # (1/r) ∂(r u w)/∂r : différence du flux r u w entre les coins i+1 et i.
        rfuw = rf * uw
        adv_w = (rfuw[1:] - rfuw[:-1]) / self._vol_c
        # + ∂(w²)/∂z : différence du flux w² entre les centres j et j-1.
        wc2 = wc * wc
        adv_w += (wc2 - _zm(wc2)) / dz
        # Flux diffusif r ∂w/∂r aux faces internes ; nul à l'axe et à la paroi glissante.
        gw = np.zeros_like(u)
        gw[1:-1] = rf[1:-1] * (w[1:] - w[:-1]) / self._drf_in          # r ∂w/∂r (0 axe/paroi)
        # (1/r) ∂/∂r(r ∂w/∂r).
        visc_w = (gw[1:] - gw[:-1]) / self._vol_c
        # + ∂²w/∂z².
        visc_w += (_zp(w) - 2.0 * w + _zm(w)) / dz**2
        # F_w = ν × diffusion - convection.
        fw = nu * visc_w - adv_w
        return fu, fw

    def _projected_rhs(self, u, w):
        # Second membre sans pression, puis projection (ajoute implicitement -∇p).
        fu, fw = self.rhs(u, w)
        return self.project(fu, fw)

    # ------------------------------------------------------------------ avance en temps
    def stable_dt(self):
        g = self.g
        # Inverse du temps de traversée d'une maille : |u|max/Δr_min + |w|max/Δz.
        adv = np.abs(self.u).max() / g.drc.min() + np.abs(self.w).max() / g.dz
        # Pas convectif (max(..., 1e-12) évite une division par zéro si l'écoulement est au repos).
        dt = self.cfl / max(adv, 1e-12)
        if self.nu > 0:  # stabilité de RK4 sur l'axe réel : |λ Δt| < 2.78
            # Limite diffusive : ν Δt (4/Δr² + 4/Δz²) ≤ 2.
            dt = min(dt, 2.0 / (self.nu * (4.0 / g.drc.min() ** 2 + 4.0 / g.dz**2)))
        # Plafond optionnel fixé par l'utilisateur.
        if self.dt_max is not None:
            dt = min(dt, self.dt_max)
        return dt

    def step(self, dt):
        """Un pas de RK4 ; chaque étage est projeté (div u = 0 à chaque sous-étape)."""
        # État au début du pas.
        u0, w0 = self.u, self.w
        # Quatre étages RK4, chacun projeté (p1 = pression de l'étage 1, soit à l'instant t^n).
        k1u, k1w, p1 = self._projected_rhs(u0, w0)
        k2u, k2w, _ = self._projected_rhs(u0 + 0.5 * dt * k1u, w0 + 0.5 * dt * k1w)
        k3u, k3w, _ = self._projected_rhs(u0 + 0.5 * dt * k2u, w0 + 0.5 * dt * k2w)
        k4u, k4w, _ = self._projected_rhs(u0 + dt * k3u, w0 + dt * k3w)
        # Combinaison RK4 (poids 1, 2, 2, 1 divisés par 6).
        self.u = u0 + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        self.w = w0 + dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        self.p = p1  # pression (p/ρ) à l'instant t^n
        # Avance du temps et du compteur.
        self.time += dt
        self.n_steps += 1
        # Tous les 50 pas, projection de la vitesse elle-même (élimine l'accumulation d'arrondis).
        if self.n_steps % 50 == 0:  # nettoie la dérive d'arrondi de la divergence
            self.u, self.w, _ = self.project(self.u, self.w)

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self):
        """Circulation, impulsion, position/rayon de l'anneau (moments de Saffman), énergie, etc."""
        g = self.g
        # Vorticité aux coins.
        om = self.vorticity()
        # Aire de la cellule duale de chaque coin (Δr dual × Δz).
        area = g.drf[:, None] * g.dz                    # aires duales des coins
        r = self._rf
        # ω dA pour chaque coin.
        wA = om * area
        # Circulation Γ = ∬ ω dr dz.
        circ = wA.sum()
        # Moment ∬ ω r² dr dz (impulsion / π).
        m2 = (wA * r**2).sum()
        # Référence axiale pour le calcul périodique de la position (dernière position connue).
        z_ref = self._z_ref if self._z_ref is not None else g.z0 + 0.5 * g.Lz
        # Écart axial périodique de chaque colonne de coins à la référence.
        dzp = g.periodic_dz(g.zf[None, :], z_ref)
        # Position Z = ∬ ω r² z / ∬ ω r² (définition de Saffman).
        Zc = z_ref + (wA * r**2 * dzp).sum() / m2 if m2 != 0 else np.nan
        # Rayon R = √(∬ ω r² / ∬ ω).
        Rc = np.sqrt(m2 / circ) if circ * m2 > 0 else np.nan
        # taille de cœur : second moment de ω restreint au cœur (ω > f ω_max, ce qui exclut le
        # sillage), corrigé de la troncature pour un profil gaussien exp(-s²/σ²) :
        # <s²>_tronqué = σ² [1 - (1 + X) e^{-X}] / (1 - e^{-X}),  X = ln(1/f)
        # np.argmax donne l'indice « aplati » du maximum ; np.unravel_index le convertit en (i, j).
        ipk = np.unravel_index(np.argmax(om), om.shape)
        # Seuil relatif : 10 % du maximum de vorticité.
        frac = 0.1
        # Vorticité du cœur seul (valeurs sous le seuil mises à 0), multipliée par l'aire.
        core = np.where(om > frac * om[ipk], om, 0.0) * area
        sp = core.sum()
        # Barycentre radial du cœur.
        rbar = (core * r).sum() / sp
        # Écart axial périodique au pic de vorticité, puis barycentre axial.
        zrel = g.periodic_dz(g.zf[None, :], g.zf[ipk[1]])
        zbar = (core * zrel).sum() / sp
        # Facteur de correction de troncature d'une gaussienne coupée à ω = f ω_max.
        X = np.log(1.0 / frac)
        trunc = (1.0 - (1.0 + X) * np.exp(-X)) / (1.0 - np.exp(-X))
        # σ = √(<s²>/facteur), s² = distance au barycentre du cœur au carré.
        sigma = np.sqrt((core * ((r - rbar) ** 2 + (zrel - zbar) ** 2)).sum() / sp / trunc)
        # Énergie cinétique E = π ∬ (u² + w²) r dr dz (u sur ses volumes duaux, w sur les cellules).
        energy = np.pi * (self.u**2 * (g.rf * g.drf)[:, None]).sum() * g.dz \
            + np.pi * (self.w**2 * self._vol_c).sum() * g.dz
        # Taux de dissipation ν ∫ ω² dV = 2π ν ∬ ω² r dr dz (dE/dt = -dissipation).
        dissipation = 2.0 * np.pi * self.nu * (om**2 * r * area).sum()
        # Dictionnaire des diagnostics de l'instant courant.
        return {
            "t": self.time,
            "circulation": circ,
            "impulse": np.pi * m2,
            "R": Rc,
            "Z": Zc,
            "sigma": sigma,
            "omega_max": om[ipk],
            "r_peak": g.rf[ipk[0]],
            "z_peak": g.zf[ipk[1]],
            "energy": energy,
            "dissipation": dissipation,
            # Divergence maximale (contrôle de l'incompressibilité discrète).
            "max_div": np.abs(self.divergence(self.u, self.w)).max(),
        }

    def fields(self):
        """Champs aux centres des cellules (pour la visualisation) + ω et ψ aux coins.

        Contient aussi les vitesses sur la grille décalée (``u_face``, ``w_face``), la grille et ν,
        nécessaires aux diagnostics avancés (critères Q et λ2, dissipation Φ = 2μ S:S).
        """
        g = self.g
        # Vitesses interpolées aux centres des cellules.
        uc = 0.5 * (self.u[:-1] + self.u[1:])
        wc = 0.5 * (self.w + _zp(self.w))
        # Copies des coordonnées et des champs (instantané indépendant de la suite du calcul) ;
        # la pression est recalculée pour être exactement cohérente avec la vitesse de l'instantané.
        return {
            "t": self.time,
            "r_c": g.rc.copy(), "z_c": g.zc.copy(), "r_f": g.rf.copy(), "z_f": g.zf.copy(),
            "u_r": uc, "u_z": wc, "p": self.current_pressure(),
            "omega": self.vorticity(), "psi": self.stream_function(),
            "u_face": self.u.copy(), "w_face": self.w.copy(), "grid": g, "nu": self.nu,
        }

    # ------------------------------------------------------------------ boucle principale
    def run(self, t_end, diag_interval=0.1, snapshot_times=(), verbose=False, record_profiles=False, n_rings=0):
        """Intègre jusqu'à ``t_end``. Renvoie {diagnostics (dict de tableaux), snapshots (liste)}.

        Les instants de ``snapshot_times`` sont atteints exactement (pas de temps ajusté).
        ``record_profiles`` : enregistre à chaque diagnostic max_r ω(z) et min_r p(z) (diagrammes de
        Hovmöller z–t) dans la clé « profiles ». ``n_rings`` ≥ 2 : suivi individuel des anneaux
        (clé « rings » : Γ_i(t), R_i(t), Z_i(t)).
        """
        # Instants d'instantanés restant à atteindre, triés (sorted) et filtrés.
        events = sorted(t for t in snapshot_times if t >= self.time - 1e-12)
        # Listes des diagnostics et des instantanés enregistrés.
        diags = []
        snaps = []
        # Profils axiaux (Hovmöller) et suivi des anneaux.
        prof_t, prof_om, prof_p = [], [], []
        ring_t, ring_data, ring_peaks = [], [], []
        # Prochain instant de diagnostic.
        next_diag = self.time
        # Chronomètre du temps de calcul réel (time.perf_counter : horloge haute résolution, en secondes).
        t_wall = time.perf_counter()

        # Fonction interne qui enregistre les diagnostics et instantanés dus à l'instant courant.
        def maybe_record():
            # « nonlocal » : autorise la modification de la variable next_diag de la fonction englobante.
            nonlocal next_diag
            if self.time >= next_diag - 1e-12:
                # Diagnostics globaux de l'instant courant (Γ, impulsion, Z, R, σ, énergie, dissipation...).
                d = self.diagnostics()
                # Mise à jour de la référence de suivi (np.isfinite : ni NaN ni infini).
                self._z_ref = d["Z"] if np.isfinite(d["Z"]) else self._z_ref
                diags.append(d)
                if record_profiles:
                    # Pression courante, référencée à sa moyenne sur la paroi extérieure (champ lointain).
                    p = self.current_pressure()
                    p = p - p[-1].mean()
                    # Instant de la ligne de Hovmöller enregistrée.
                    prof_t.append(self.time)
                    # om.max(axis=0) : maximum sur r de chaque colonne z ; min(axis=0) : minimum sur r.
                    prof_om.append(self.vorticity().max(axis=0))
                    prof_p.append(p.min(axis=0))
                if n_rings >= 2:
                    # Instant et tableau (n_rings, 3) des (Γ, R, Z) de chaque anneau identifié.
                    ring_t.append(self.time)
                    ring_data.append(self.track_rings(n_rings))
                    # Nombre de pics distincts détectés (diagnostic de fusion des cœurs).
                    ring_peaks.append(self._last_n_peaks)
                # Instant du diagnostic suivant.
                next_diag += diag_interval
            # Instantané(s) si l'instant courant coïncide avec un instant demandé.
            while events and abs(self.time - events[0]) < 1e-9:
                # Copie des champs de l'instant (vitesses, pression, ω, ψ, grille, ν).
                snaps.append(self.fields())
                # list.pop(0) retire le premier élément (instant traité).
                events.pop(0)

        # Enregistrement initial.
        maybe_record()
        # Boucle en temps jusqu'à t_end.
        while self.time < t_end - 1e-12:
            # Pas stable, sans dépasser t_end.
            dt = min(self.stable_dt(), t_end - self.time)
            # Raccourcit le pas pour tomber exactement sur le prochain instantané.
            if events and self.time + dt > events[0] - 1e-12:
                dt = events[0] - self.time
            # Un pas de RK4 projeté, puis enregistrements éventuels.
            self.step(dt)
            maybe_record()
            # Affichage périodique de la progression.
            if verbose and self.n_steps % 200 == 0:
                # Derniers diagnostics enregistrés (diags[-1] : dernier élément de la liste).
                d = diags[-1]
                print(f"  t = {self.time:7.3f}  dt = {dt:.4f}  Z = {d['Z']:.4f}  R = {d['R']:.4f}  "
                      f"div = {d['max_div']:.1e}  ({time.perf_counter() - t_wall:5.1f} s)")
        # Conversion de la liste de dictionnaires en dictionnaire de tableaux (une série par clé).
        out = {k: np.array([d[k] for d in diags]) for k in diags[0]}
        # Résultat : diagnostics, instantanés et durée réelle du calcul.
        result = {"diagnostics": out, "snapshots": snaps, "wall_time": time.perf_counter() - t_wall}
        if record_profiles:
            # Tableaux (nt, Nz) : lignes = instants, colonnes = positions axiales.
            result["profiles"] = {"t": np.array(prof_t), "z_f": self.g.zf.copy(), "z_c": self.g.zc.copy(),
                                  "omega_max": np.array(prof_om), "p_min": np.array(prof_p)}
        if n_rings >= 2:
            # np.array(ring_data) : forme (nt, n_rings, 3) ; on sépare les trois colonnes Γ, R, Z.
            rd = np.array(ring_data)
            result["rings"] = {"t": np.array(ring_t), "Gamma": rd[:, :, 0], "R": rd[:, :, 1], "Z": rd[:, :, 2],
                               "n_peaks": np.array(ring_peaks)}
        return result


# =============================================================================================
# Diagnostics avancés sur un instantané : gradient de vitesse, critères Q et λ2, dissipation
# =============================================================================================
def _corners_to_centers(q):
    """Moyenne des 4 coins (i, i+1) × (j, j+1) entourant chaque centre de cellule (i, j)."""
    # Moyenne en r des coins i et i+1 : valeurs en (rc[i], zf[j]), forme (Nr, Nz).
    qr = 0.5 * (q[:-1] + q[1:])
    # Moyenne en z des faces j et j+1 (np.roll(..., -1, axis=1) = colonne j+1, périodique).
    return 0.5 * (qr + np.roll(qr, -1, axis=1))


def velocity_gradient_tensor(snap):
    """Composantes du tenseur gradient de vitesse aux centres des cellules (écoulement sans swirl).

    Dans la base locale (e_r, e_θ, e_z), ∇u = [[a, 0, b], [0, e, 0], [c, 0, d]] avec
        a = ∂u_r/∂r,  b = ∂u_r/∂z,  c = ∂u_z/∂r,  d = ∂u_z/∂z,  e = u_r/r .
    a, d, e sont exacts au centre sur la grille MAC ; b et c, naturels aux coins, y sont moyennés.
    On renvoie aussi les moyennes aux centres de (b + c)² (cisaillement, positif) et de (b - c)² = ω².
    Incompressibilité discrète : a + d + e = div u = 0.
    """
    # Grille et vitesses sur la grille décalée, prises dans l'instantané.
    g = snap["grid"]
    u = snap["u_face"]
    w = snap["w_face"]
    # a = ∂u_r/∂r : différence des faces radiales i+1 et i divisée par la largeur de la cellule.
    a = (u[1:] - u[:-1]) / g.drc[:, None]
    # e = u_r/r : u_r interpolé au centre divisé par le rayon du centre.
    e = 0.5 * (u[:-1] + u[1:]) / g.rc[:, None]
    # d = ∂u_z/∂z : différence des faces axiales j+1 et j (np.roll(w, -1, axis=1) = w[:, j+1]).
    d = (np.roll(w, -1, axis=1) - w) / g.dz
    # b = ∂u_r/∂z aux coins : (u[i, j] - u[i, j-1])/Δz (nul sur l'axe et la paroi où u = 0).
    b_c = (u - np.roll(u, 1, axis=1)) / g.dz
    # c = ∂u_z/∂r aux coins internes ; nul sur l'axe (symétrie) et sur la paroi glissante.
    c_c = np.zeros_like(u)
    # Différence des faces axiales des cellules voisines i+1 et i divisée par la distance de leurs centres.
    c_c[1:-1] = (w[1:] - w[:-1]) / g.drf[1:-1, None]
    # b et c ramenés aux centres ; les carrés (b ± c)² sont moyennés depuis les coins (valeurs positives).
    return {
        "a": a, "b": _corners_to_centers(b_c), "c": _corners_to_centers(c_c), "d": d, "e": e,
        "shear2": _corners_to_centers((b_c + c_c) ** 2),
        "omega2": _corners_to_centers((b_c - c_c) ** 2),
    }


def q_criterion(snap):
    """Critère Q = ½(‖Ω‖² - ‖S‖²) = -½ ∂_i u_j ∂_j u_i  (Hunt, Wray & Moin 1988).

    Avec les notations de :func:`velocity_gradient_tensor` : Q = -½(a² + d² + e²) - b c.
    Q > 0 là où la rotation l'emporte sur la déformation (cœurs tourbillonnaires). Pour un fluide
    incompressible, ∇²p/ρ = 2Q : la pression est minimale là où Q est grand.
    """
    # Composantes du gradient de vitesse aux centres.
    t = velocity_gradient_tensor(snap)
    # -½ tr(∇u ∇u) : termes diagonaux au carré et produit croisé b c des termes hors diagonale.
    return -0.5 * (t["a"] ** 2 + t["d"] ** 2 + t["e"] ** 2) - t["b"] * t["c"]


def lambda2_criterion(snap):
    """Critère λ2 (Jeong & Hussain 1995) : 2ᵉ valeur propre de S² + Ω² ; cœur tourbillonnaire si λ2 < 0.

    Le bloc (r, z) de M = S² + Ω² vaut, avec s = (b + c)/2 et h = (b - c)/2,
        M_rr = a² + s² - h²,  M_zz = d² + s² - h²,  M_rz = s (a + d),
    et la composante azimutale découplée vaut M_θθ = e². Les valeurs propres du bloc 2×2 sont
    (M_rr + M_zz)/2 ± √((M_rr - M_zz)²/4 + M_rz²) ; λ2 est la médiane des trois valeurs propres.
    """
    # Composantes du gradient de vitesse aux centres, puis variables locales plus lisibles.
    t = velocity_gradient_tensor(snap)
    a, b, c, d, e = t["a"], t["b"], t["c"], t["d"], t["e"]
    # Parties symétrique (déformation) et antisymétrique (rotation) du bloc de cisaillement.
    s = 0.5 * (b + c)
    h = 0.5 * (b - c)
    # Composantes du bloc (r, z) de S² + Ω².
    m11 = a**2 + s**2 - h**2
    m22 = d**2 + s**2 - h**2
    m12 = s * (a + d)
    # Valeurs propres analytiques d'une matrice symétrique 2×2.
    # Demi-trace (centre des deux valeurs propres) et demi-écart (rayon du cercle de Mohr).
    mean = 0.5 * (m11 + m22)
    rad = np.sqrt(0.25 * (m11 - m22) ** 2 + m12**2)
    # np.stack empile les trois champs de valeurs propres ; np.median(axis=0) prend la valeur du milieu.
    return np.median(np.stack([mean - rad, mean + rad, e**2]), axis=0)


def dissipation_rate(snap, rho=1.0):
    """Taux local de dissipation visqueuse Φ = 2μ S:S (W/m³ si ρ en kg/m³), aux centres des cellules.

    S:S = a² + d² + e² + ½ (b + c)²  (le terme de cisaillement est moyenné au carré depuis les coins,
    ce qui garantit Φ ≥ 0). Pour un domaine périodique à paroi glissante, ∫Φ dV = μ ∫ω² dV.
    """
    # Composantes du gradient de vitesse aux centres.
    t = velocity_gradient_tensor(snap)
    # μ = ρ ν.
    mu = rho * snap["nu"]
    # 2μ S:S avec S_rr = a, S_zz = d, S_θθ = e et S_rz = (b + c)/2 (compté deux fois : 2 × ¼(b + c)² = ½(b + c)²).
    return 2.0 * mu * (t["a"] ** 2 + t["d"] ** 2 + t["e"] ** 2 + 0.5 * t["shear2"])


def enstrophy_density(snap):
    """Densité d'enstrophie ½ ω² aux centres des cellules (moyenne des ω² des 4 coins)."""
    # ω est connue aux coins : on moyenne ω² (et non ω) pour conserver exactement ∫ω² dV.
    return 0.5 * _corners_to_centers(snap["omega"] ** 2)


def volume_integral(snap, field):
    """Intégrale volumique ∫ f dV = 2π ∬ f r dr dz d'un champ défini aux centres des cellules."""
    # Grille de l'instantané (rayons des centres et largeurs des cellules).
    g = snap["grid"]
    # Poids r dr de chaque cellule (colonne) × Δz, puis somme et facteur 2π (révolution).
    return 2.0 * np.pi * float((field * (g.rc * g.drc)[:, None]).sum() * g.dz)


# =============================================================================================
# Cas type : anneau de Lamb–Oseen et comparaison à Saffman
# =============================================================================================
def run_vortex_ring(R=1.0, a=0.2, Gamma=1.0, Re=1000.0, t_end=20.0, dr=0.025, Lr=8.0, r_uniform=2.2,
                    growth=1.05, Lz=10.0, z0=2.5, cfl=0.8, diag_interval=0.1, snapshot_times=(),
                    verbose=False, record_profiles=False):
    """Simule un anneau de Lamb–Oseen (Re_Γ = Γ/ν) et compare sa vitesse à la théorie.

    Renvoie un dict : solver, diagnostics, snapshots, et les vitesses simulée (dZ/dt) et
    théoriques (Saffman, Saffman + Fukumoto–Moffatt) évaluées avec le rayon R(t) mesuré et
    σ(t)² = a² + 4νt. ``record_profiles`` ajoute les profils axiaux pour les diagrammes de Hovmöller.
    """
    # Grille étirée en r, uniforme en z (Nz = Lz/dr pour des mailles carrées près de l'anneau).
    grid = AxisymmetricGrid.stretched(dr, r_uniform, Lr, growth, Lz, int(round(Lz / dr)))
    # Solveur avec ν = Γ/Re.
    solver = AxisymmetricNavierStokesSolver(grid, nu=Gamma / Re, cfl=cfl)
    # Condition initiale : anneau gaussien.
    solver.init_lamb_oseen_ring(R, a, Gamma, z0)
    # Intégration en temps.
    res = solver.run(t_end, diag_interval=diag_interval, snapshot_times=snapshot_times, verbose=verbose,
                     record_profiles=record_profiles)
    # Séries temporelles des diagnostics et instants correspondants.
    d = res["diagnostics"]
    t = d["t"]
    # Vitesse simulée U = dZ/dt (np.gradient : différences finies centrées, décentrées aux bords).
    U_sim = np.gradient(d["Z"], t)
    # Rayon de cœur théorique σ(t) = √(a² + 4νt).
    sigma_th = theory.viscous_core_radius(a, solver.nu, t)
    # Ajout des grandeurs de comparaison au dictionnaire résultat (dict.update).
    res.update({
        "solver": solver,
        "grid": grid,
        "params": {"R": R, "a": a, "Gamma": Gamma, "Re": Re, "nu": solver.nu},
        "U_sim": U_sim,
        "sigma_theory": sigma_th,
        # Saffman avec le rayon d'anneau mesuré et σ théorique.
        "U_saffman": theory.saffman_speed(Gamma, d["R"], sigma_th),
        # Saffman + correction de Fukumoto–Moffatt.
        "U_saffman_fm": theory.saffman_speed(Gamma, d["R"], sigma_th, second_order=True),
        # Saffman avec le σ mesuré dans la simulation.
        "U_saffman_measured_sigma": theory.saffman_speed(Gamma, d["R"], d["sigma"]),
    })
    return res


def run_coaxial_rings(rings, Re=3000.0, t_end=12.0, dr=0.02, Lr=8.0, r_uniform=2.4, growth=1.05, Lz=10.0,
                      cfl=0.8, diag_interval=0.05, snapshot_times=(), record_profiles=True, verbose=False):
    """Simule plusieurs anneaux coaxiaux visqueux (ex. saute-mouton) avec suivi individuel des cœurs.

    ``rings`` : liste de dictionnaires {"R", "a", "Gamma", "z0"} (voir ``init_rings``) ;
    Re est construit sur la plus grande circulation : ν = max|Γ_i| / Re.
    Renvoie le dict de ``run`` complété par solver, grid et params ; « rings » contient Γ_i(t), R_i(t), Z_i(t).
    """
    # Grille étirée en r et uniforme en z (mailles carrées dans la zone des anneaux).
    grid = AxisymmetricGrid.stretched(dr, r_uniform, Lr, growth, Lz, int(round(Lz / dr)))
    # Circulation de référence (expression génératrice dans max).
    gamma_ref = max(abs(r["Gamma"]) for r in rings)
    # Solveur avec ν = Γ_ref / Re et nombre de CFL imposé.
    solver = AxisymmetricNavierStokesSolver(grid, nu=gamma_ref / Re, cfl=cfl)
    # Superposition des anneaux de Lamb–Oseen.
    solver.init_rings(rings)
    # Intégration avec profils axiaux et suivi des len(rings) cœurs.
    res = solver.run(t_end, diag_interval=diag_interval, snapshot_times=snapshot_times, verbose=verbose,
                     record_profiles=record_profiles, n_rings=len(rings))
    # Ajout du solveur, de la grille et des paramètres (copies des dictionnaires d'anneaux) au résultat.
    res.update({"solver": solver, "grid": grid,
                "params": {"rings": [dict(r) for r in rings], "Re": Re, "nu": solver.nu}})
    return res
