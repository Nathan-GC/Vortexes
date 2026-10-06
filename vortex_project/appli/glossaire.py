"""
appli/glossaire.py — Définitions des termes techniques de l'application.

Les définitions servent d'info-bulles (paramètre « help » des widgets et des métriques) et forment la page
« 📖 Glossaire » de l'onglet de documentation. Chaque entrée renvoie à la section du rapport qui la détaille.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# Terme → (définition, renvoi vers le rapport théorique ou le rapport « moteurs »).
GLOSSAIRE = {
    "Biot–Savart (loi de)": ("Loi qui donne la vitesse induite en tout point par une distribution de vorticité, "
                             "comme le champ magnétique créé par un courant électrique.", "théorie § 1.5"),
    "Circulation Γ": ("Intégrale de la vitesse le long d'un contour qui entoure le cœur ; égale au flux de "
                      "vorticité à travers le cœur. Elle mesure l'intensité de l'anneau et son signe fixe le sens "
                      "de propagation.", "théorie § 0 et § 1.7"),
    "CFL (condition de)": ("Contrainte de stabilité du pas de temps : un élément de fluide ne doit pas traverser "
                           "plus d'une maille par pas. Le solveur choisit Δt automatiquement (CFL = 0,8).",
                           "théorie § 4.2"),
    "Critère λ₂": ("Deuxième valeur propre de S² + W² (S : déformation, W : rotation) ; λ₂ < 0 repère un cœur "
                   "tourbillonnaire, là où la pression est minimale.", "théorie § 5.1"),
    "Critère Q": ("Q = ½(‖W‖² − ‖S‖²) : positif là où la rotation l'emporte sur la déformation, donc dans les cœurs "
                  "tourbillonnaires. Pour un fluide incompressible, ∇²p/ρ = 2Q.", "théorie § 5.1"),
    "Diagramme de Hovmöller": ("Carte espace–temps (z en abscisse, t en ordonnée) d'une grandeur : la pente d'une "
                               "trace donne la vitesse de propagation.", "théorie § 5.3"),
    "Dissipation Φ": ("Puissance transformée en chaleur par le frottement visqueux, Φ = 2μ S:S ; son intégrale "
                      "vaut μ∫ω² dV et égale la perte d'énergie cinétique.", "théorie § 5.2"),
    "Divergence (incompressibilité)": ("∇·u, nulle pour un fluide incompressible ; le solveur la maintient à ~10⁻¹² "
                                       "(contrôle de qualité).", "théorie § 4.2"),
    "Énergie adimensionnelle α": ("α = E/(ρ^½ I^½ Γ^{3/2}) : combinaison sans dimension de l'énergie, de l'impulsion "
                                  "et de la circulation ; un anneau stable ne peut descendre sous α* ≈ 0,33.",
                                  "théorie § 3.2"),
    "Enstrophie Ω": ("Intégrale de ω² sur le volume ; la dissipation visqueuse vaut νΩ (dE/dt = −νΩ).",
                     "théorie § 5.2"),
    "Fukumoto–Moffatt (correction de)": ("Terme correctif de la formule de Saffman pour un cœur d'épaisseur finie : "
                                         "U = Γ/(4πR)[ln(8R/σ) − 0,558 − 0,918 σ²/R²].", "théorie § 2.3"),
    "Fusion des cœurs": ("Deux tourbillons de même signe se fondent en un seul quand leur écart b tombe sous "
                         "≈ 4 rayons de cœur ((σ/b)_c ≈ 0,22–0,24, Meunier et al. 2002).", "théorie § 5.4"),
    "Impulsion I": ("I = πρ∬ω r² dr dz (πρR²Γ pour un anneau mince) : quantité de mouvement communiquée au fluide, "
                    "conservée en l'absence de force extérieure.", "théorie § 1.7"),
    "Lamb–Oseen (cœur de)": ("Cœur gaussien, ω ∝ exp(−s²/a²) : solution des équations visqueuses pour un tourbillon "
                             "rectiligne, dont le rayon croît comme σ² = a² + 4νt.", "théorie § 2.3"),
    "Masse ajoutée": ("Masse de fluide entraînée avec un corps (ou une bulle tourbillonnaire) accéléré ; une sphère "
                      "entraîne la moitié de son volume de fluide (C_A = 1/2).", "moteurs § 2.2"),
    "Modèle coaxial": ("Modèle réduit où chaque anneau est décrit par son rayon R(t) et sa position Z(t) : référence "
                       "en fluide parfait pour le saute-mouton et la collision.", "théorie § 2.4"),
    "Modèle « slug »": ("Modèle d'éjection piston-cylindre : un bouchon de fluide de longueur L sort de la buse à la "
                        "vitesse U_p ; il fournit Γ = U_p L/2.", "théorie § 3.1"),
    "Nombre de formation F*": ("Course L/D au-delà de laquelle l'anneau cesse de grossir et se détache du jet "
                               "(pinch-off) : F* = √(π/2)/α* ≈ 3,8.", "théorie § 3.2"),
    "Nombre de Péclet de maille": ("Pe_Δ = u_max Δr/ν : compare le transport à la diffusion à l'échelle d'une maille. "
                                   "Trop grand avec un cœur mal résolu, il fait apparaître de petites oscillations "
                                   "(vorticité négative parasite).", "tutoriel § 3.6"),
    "Nombre de Reynolds Re_Γ": ("Re_Γ = Γ/ν : rapport entre transport et viscosité. Plus il est grand, plus l'anneau "
                                "garde longtemps son énergie.", "théorie § 0"),
    "Paramètre d'injection κ": ("κ = Q/(2πUaL) : rapport entre débit axial et rotation dans une chambre à vortex ; "
                                "le modèle exact suppose κ ≪ 1.", "moteurs § 1.2"),
    "Portrait de phase": ("Trajectoire du système dans le plan (R₁ − R₂, z₁ − z₂) : cycle fermé pour une dynamique "
                          "périodique (fluide parfait), spirale quand la viscosité dissipe.", "théorie § 5.4"),
    "Rayon de cœur a (σ)": ("Taille de la zone où se concentre la vorticité de l'anneau ; un cœur fin donne un anneau "
                            "rapide.", "théorie § 2.2"),
    "Rendement propulsif η": ("Rapport entre la puissance utile (poussée × vitesse) et la puissance fournie au jet.",
                              "moteurs § 2.3"),
    "Rosenhead–Moore (régularisation de)": ("Lissage du noyau de Biot–Savart, 1/|x|³ → 1/(|x|² + δ²)^{3/2}, qui évite "
                                            "la vitesse infinie sur un filament ; δ ≈ 0,64 × le rayon d'un cœur "
                                            "gaussien équivalent.", "théorie § 1.6"),
    "Saffman (formule de)": ("Vitesse d'un anneau mince à cœur gaussien : U = Γ/(4πR)[ln(8R/σ) − 0,558].",
                             "théorie § 2.2–2.3"),
    "Saute-mouton": ("Deux anneaux coaxiaux de même sens : l'anneau arrière, contracté et accéléré, traverse l'anneau "
                     "avant, puis les rôles s'échangent (leapfrogging).", "théorie § 2.4"),
    "Temps de séjour": ("Durée passée dans la chambre par une particule injectée ; sa moyenne vaut V/Q.",
                        "moteurs § 1.3"),
    "Vortex de Hill": ("Tourbillon sphérique exact des équations d'Euler : la vorticité, proportionnelle à r, remplit "
                       "une sphère qui avance sans se déformer ; Γ = 5Ua.", "théorie § 2.1"),
    "Vorticité ω": ("Rotationnel de la vitesse, ω = ∇ × u : mesure la rotation locale du fluide (deux fois la vitesse "
                    "angulaire d'un petit élément).", "théorie § 1.2"),
}


def aide(*termes):
    """Texte d'info-bulle réunissant les définitions des termes demandés (les termes inconnus sont ignorés)."""
    # Chaque définition est précédée du terme en gras ; « \n\n » sépare les paragraphes dans l'info-bulle.
    return "\n\n".join(f"**{t}** — {GLOSSAIRE[t][0]}" for t in termes if t in GLOSSAIRE) or None


def texte_glossaire():
    """Page Markdown du glossaire : termes par ordre alphabétique, définitions et renvois."""
    # sorted(..., key=str.lower) : ordre alphabétique sans tenir compte des majuscules.
    lignes = ["Définitions des termes employés dans l'application et dans les rapports. Les mêmes définitions "
              "apparaissent en info-bulle (icône ?) à côté des réglages et des résultats.", ""]
    for terme in sorted(GLOSSAIRE, key=str.lower):
        definition, renvoi = GLOSSAIRE[terme]
        lignes.append(f"**{terme}** — {definition} *({renvoi})*")
        lignes.append("")
    return "\n".join(lignes)
