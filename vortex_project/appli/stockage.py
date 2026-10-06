"""
appli/stockage.py — Historique des calculs et mémoire des résultats déjà calculés.

* Historique : les derniers calculs de chaque volet (8 au plus), affichables et comparables entre eux.
* Mémoire : un calcul relancé avec des paramètres identiques est restitué instantanément (12 résultats au plus).

Les deux vivent dans la mémoire du serveur Streamlit, dans un objet de st.cache_resource : ils survivent au
rechargement de la page et au rechargement des modules (modification du code pendant que l'application tourne),
et sont partagés par les onglets du navigateur, mais pas au redémarrage du serveur.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# itertools.count : compteur des numéros de calcul ; threading.Lock : accès protégé ; time.strftime : heure du
# calcul ; OrderedDict : dictionnaire ordonné (éviction du résultat le plus ancien).
import itertools
import threading
import time
from collections import OrderedDict

# streamlit : st.cache_resource (objet unique et durable du serveur).
import streamlit as st

# Nombre maximal de calculs gardés dans l'historique de chaque volet et de résultats mémorisés.
TAILLE_HISTORIQUE = 8
TAILLE_MEMOIRE = 12


# st.cache_resource : l'objet est créé une seule fois pour tout le serveur, puis renvoyé tel quel à chaque appel.
@st.cache_resource(show_spinner=False)
def _magasin():
    """Structures partagées : verrou, historique par volet, mémoire des résultats et compteur de numéros."""
    return {"verrou": threading.Lock(), "historique": {}, "memoire": OrderedDict(), "numeros": itertools.count(1)}


def cle_parametres(cle_volet, params):
    """Clé immuable d'un calcul : volet + paramètres triés (repr distingue 1 et 1.0 sans ambiguïté utile ici)."""
    # sorted(params.items()) : ordre indépendant de la construction du dictionnaire.
    return (cle_volet, tuple(sorted((k, repr(v)) for k, v in params.items())))


def chercher(cle_volet, params):
    """Résultat déjà calculé pour ces paramètres, ou None."""
    m = _magasin()
    cle = cle_parametres(cle_volet, params)
    with m["verrou"]:
        sortie = m["memoire"].get(cle)
        if sortie is not None:
            # move_to_end : le résultat redevient le plus récemment utilisé.
            m["memoire"].move_to_end(cle)
        return sortie


def memoriser(cle_volet, params, sortie):
    """Mémorise un résultat (le plus ancien est oublié au-delà de TAILLE_MEMOIRE)."""
    m = _magasin()
    with m["verrou"]:
        m["memoire"][cle_parametres(cle_volet, params)] = sortie
        # popitem(last=False) : retire l'entrée la plus ancienne.
        while len(m["memoire"]) > TAILLE_MEMOIRE:
            m["memoire"].popitem(last=False)


def ajouter(cle_volet, params, sortie, etiquette, meta=None):
    """Ajoute un calcul en tête de l'historique du volet et renvoie son entrée.

    Entrée : numéro, paramètres, sortie du calcul, heure, étiquette lisible, métadonnées (scénario, prédiction,
    réglages des widgets) et résultat d'une éventuelle vérification de convergence.
    """
    m = _magasin()
    with m["verrou"]:
        entree = {"id": next(m["numeros"]), "volet": cle_volet, "params": dict(params), "sortie": sortie,
                  "heure": time.strftime("%H:%M:%S"), "etiquette": etiquette, "meta": dict(meta or {}),
                  "convergence": None}
        # setdefault : liste créée au premier calcul du volet ; insert(0, …) : le plus récent en tête.
        liste = m["historique"].setdefault(cle_volet, [])
        liste.insert(0, entree)
        # del liste[N:] : on ne garde que les N derniers calculs.
        del liste[TAILLE_HISTORIQUE:]
        return entree


def historique(cle_volet):
    """Calculs du volet, du plus récent au plus ancien (copie de la liste)."""
    m = _magasin()
    with m["verrou"]:
        return list(m["historique"].get(cle_volet, []))


def trouver(cle_volet, numero):
    """Entrée de l'historique portant ce numéro, ou None."""
    # next(générateur, None) : première entrée qui convient.
    return next((e for e in historique(cle_volet) if e["id"] == numero), None)


def effacer(cle_volet):
    """Vide l'historique du volet (la mémoire des résultats est conservée)."""
    m = _magasin()
    with m["verrou"]:
        m["historique"].pop(cle_volet, None)
