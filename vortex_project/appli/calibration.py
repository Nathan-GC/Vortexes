"""
appli/calibration.py — Coefficients des durées estimées, recalibrés après chaque calcul sur ce PC.

Les estimations s'écrivent coût_unitaire × taille × nombre de pas (+ durée des figures). Les valeurs par défaut
ont été mesurées sur le PC de développement ; après chaque calcul, l'application compare la durée réelle à
l'estimation et met à jour les coefficients par moyenne glissante. Ils sont enregistrés dans
.cache/calibration_appli.json (dossier ignoré par git) et survivent donc au redémarrage de l'application.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# copy.deepcopy : copie indépendante des valeurs par défaut ; json : lecture/écriture du fichier ;
# threading.Lock : écritures protégées (plusieurs fils peuvent terminer un calcul en même temps).
import copy
import json
import threading

# Dossier du projet (le fichier de calibration est rangé dans PROJET/.cache).
from appli.commun import PROJET

# Fichier de calibration propre à ce PC.
FICHIER = PROJET / ".cache" / "calibration_appli.json"
# Valeurs par défaut (PC de développement) :
# * cout_pas : secondes par maille et par pas (solveur axisymétrique), par paire de nœuds et par pas
#   (filaments, coût en O(M²)), par point de la carte de rendement (moteurs) ;
# * correction_pas : rapport nombre de pas réel / estimé (deux anneaux : les vitesses induites mutuelles
#   raccourcissent le pas de temps) ;
# * correction_figures : rapport durée réelle / estimée des figures et animations.
DEFAUTS = {
    "cout_pas": {"anneau": 1.5e-6, "hill": 1.5e-6, "saute_mouton": 2.6e-6, "filaments": 0.6e-6, "moteurs": 1.5e-3},
    "correction_pas": {"anneau": 1.0, "hill": 1.0, "saute_mouton": 1.32},
    "correction_figures": {"anneau": 0.8, "hill": 0.8, "saute_mouton": 0.8, "filaments": 0.8, "moteurs": 0.9},
    "mesures": 0,
}
# Poids de la dernière mesure dans la moyenne glissante, et bornes des corrections (une mesure aberrante,
# PC très chargé par exemple, ne peut pas fausser durablement les estimations).
POIDS = 0.5
BORNES = (0.2, 5.0)
# Verrou des lectures/écritures et copie en mémoire du fichier.
_VERROU = threading.Lock()
_DONNEES = None


def lire():
    """Coefficients courants (fichier lu une seule fois, valeurs par défaut pour les clés absentes)."""
    global _DONNEES
    with _VERROU:
        if _DONNEES is None:
            # Copie des valeurs par défaut, complétée par le contenu du fichier s'il existe et se lit correctement.
            _DONNEES = copy.deepcopy(DEFAUTS)
            try:
                lu = json.loads(FICHIER.read_text(encoding="utf-8"))
                for cle, valeur in lu.items():
                    # Dictionnaires fusionnés clé par clé (dict.update) ; valeurs simples remplacées.
                    if isinstance(valeur, dict) and isinstance(_DONNEES.get(cle), dict):
                        _DONNEES[cle].update(valeur)
                    else:
                        _DONNEES[cle] = valeur
            except (OSError, ValueError):
                # Fichier absent ou illisible : on garde les valeurs par défaut.
                pass
        return _DONNEES


def _ecrire():
    """Enregistre les coefficients (silencieusement si le disque refuse : ce n'est qu'une commodité)."""
    try:
        # mkdir(parents=True, exist_ok=True) : crée .cache au besoin ; indent=2 : fichier lisible.
        FICHIER.parent.mkdir(parents=True, exist_ok=True)
        FICHIER.write_text(json.dumps(_DONNEES, indent=2), encoding="utf-8")
    except OSError:
        pass


def _moyenne(groupe, cle, valeur, bornes=None):
    """Moyenne glissante : nouvelle valeur = (1 − POIDS) × ancienne + POIDS × mesure, éventuellement bornée."""
    # Ancienne valeur (la mesure elle-même si la clé n'existe pas encore).
    ancien = _DONNEES[groupe].get(cle, valeur)
    nouveau = (1.0 - POIDS) * ancien + POIDS * valeur
    if bornes is not None:
        # Correction bornée (max/min) pour rester plausible.
        nouveau = min(max(nouveau, bornes[0]), bornes[1])
    _DONNEES[groupe][cle] = nouveau


def apprendre(type_cout, estimation, mesures):
    """Met à jour les coefficients à partir d'un calcul terminé.

    ``type_cout`` : clé du type de calcul (anneau, hill, saute_mouton, filaments, moteurs) ; ``estimation`` :
    dictionnaire produit avant le lancement (taille, n_pas_brut, figures_brut) ; ``mesures`` : durées et nombre de
    pas mesurés pendant le calcul (t_calcul, n_pas, taille, t_figures).
    """
    if not estimation or not mesures:
        return
    lire()
    with _VERROU:
        # Coût unitaire mesuré = durée / (nombre de pas × taille), mêlé à l'ancienne valeur.
        if mesures.get("t_calcul") and mesures.get("n_pas") and mesures.get("taille"):
            cout = mesures["t_calcul"] / (mesures["n_pas"] * mesures["taille"])
            ancien = _DONNEES["cout_pas"].get(type_cout, cout)
            # Bornes relatives : au plus un facteur 5 d'écart avec l'ancienne valeur par mesure.
            _moyenne("cout_pas", type_cout, cout, (ancien / 5.0, ancien * 5.0))
        # Correction du nombre de pas (réel / estimé sans correction).
        if type_cout in _DONNEES["correction_pas"] and mesures.get("n_pas") and estimation.get("n_pas_brut"):
            _moyenne("correction_pas", type_cout, mesures["n_pas"] / estimation["n_pas_brut"], BORNES)
        # Correction de la durée des figures (réelle / estimée sans correction).
        if mesures.get("t_figures") and estimation.get("figures_brut"):
            _moyenne("correction_figures", type_cout, mesures["t_figures"] / estimation["figures_brut"], BORNES)
        _DONNEES["mesures"] = int(_DONNEES.get("mesures", 0)) + 1
        _ecrire()


def estimation(type_cout, taille, n_pas_brut, figures_brut, texte):
    """Dictionnaire d'estimation : durée de calcul et de figures corrigées par la calibration.

    ``taille`` : nombre de mailles, de paires de nœuds ou de points ; ``n_pas_brut`` : nombre de pas estimé
    (None si sans objet) ; ``figures_brut`` : durée estimée des figures ; ``texte`` : description de la taille.
    """
    d = lire()
    # Nombre de pas corrigé (facteur 1 si le type n'a pas de correction) et durée du calcul.
    n_pas = n_pas_brut * d["correction_pas"].get(type_cout, 1.0) if n_pas_brut else None
    calcul = d["cout_pas"][type_cout] * taille * (n_pas if n_pas else 1.0)
    return {"type_cout": type_cout, "taille": taille, "n_pas_brut": n_pas_brut, "n_pas": n_pas, "calcul": calcul,
            "figures_brut": figures_brut, "figures": figures_brut * d["correction_figures"].get(type_cout, 1.0),
            "texte": texte, "mesures": int(d.get("mesures", 0))}
