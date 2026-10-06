"""
appli/volets — Un module par volet simulable.

Chaque module définit : CLE, LIBELLE, RESUME, PRESENTATION, FORMULE (LaTeX), THEORIE (sections du rapport),
LIMITES, INSTANTANE, PREFIXE (préfixe des clés de ses widgets), SCENARIOS (scénarios guidés), et les fonctions
controles() → paramètres, verifier(p) → (alertes, estimation), calculer(p, prog, complet) → sortie,
etiquette(p) → texte, raffinement(p, facteur) → (paramètres raffinés, description) ou None. Le volet
analytique (INSTANTANE = True) fournit afficher(p) à la place du calcul.
"""

# Modules des volets, dans l'ordre du menu déroulant.
from appli.volets import anneau, filaments, moteurs, saute_mouton, theorie

# Registre ordonné : clé du volet → module.
VOLETS = {module.CLE: module for module in (anneau, saute_mouton, filaments, moteurs, theorie)}
