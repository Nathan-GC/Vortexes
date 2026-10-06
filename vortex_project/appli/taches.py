"""
appli/taches.py — Calculs en arrière-plan : lancement, indicateur de chargement, annulation et fin du calcul.

Le calcul tourne dans un fil d'exécution séparé (threading.Thread) : l'interface reste utilisable pendant ce
temps (lecture de la documentation, changement d'onglet) et le bouton « Annuler » l'interrompt proprement.
La page de simulation affiche l'avancement et attend la fin dans une boucle que Streamlit interrompt dès que
l'utilisateur agit ; la boucle reprend à l'exécution suivante du script, tant que le calcul n'est pas terminé.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# threading : fil de calcul ; time : chronomètre et attente ; traceback : pile d'appels d'une erreur.
import threading
import time
import traceback

# streamlit : affichage de la progression ; modules de l'application : calibration et historique.
import streamlit as st

from appli import calibration, stockage
from appli.commun import CalculAnnule, Progression, duree_lisible


class Tache:
    """État d'un calcul partagé entre le fil de calcul (qui l'écrit) et l'interface (qui le lit)."""

    def __init__(self, nature, volet, params, estimation=None, meta=None, entree=None):
        # Nature : « simulation » (calcul complet avec figures) ou « convergence » (calcul raffiné sans figures).
        self.nature = nature
        # Module du volet (fonctions calculer, etiquette...), paramètres et estimation de durée au lancement.
        self.volet = volet
        self.params = dict(params)
        self.estimation = estimation
        # Métadonnées (scénario guidé, prédiction) et entrée d'historique vérifiée (convergence).
        self.meta = dict(meta or {})
        self.entree = entree
        # Avancement écrit par le calcul (fraction, texte) et demande d'annulation écrite par l'interface.
        self.fraction = 0.0
        self.texte = "Préparation du calcul…"
        self.annuler = False
        # Fin du calcul : drapeau, erreur éventuelle (message et pile d'appels), résultat et durée.
        self.termine = False
        self.erreur = None
        self.trace = None
        self.sortie = None
        self.debut = time.perf_counter()
        self.duree = 0.0


def _executer(tache):
    """Corps du fil de calcul : appelle la fonction de calcul du volet et consigne le résultat ou l'erreur."""
    try:
        # Progression : transmet l'avancement à la tâche et lève CalculAnnule si l'utilisateur annule ;
        # complet=False pour la convergence (pas de figures).
        tache.sortie = tache.volet.calculer(tache.params, Progression(tache), complet=tache.nature == "simulation")
    except CalculAnnule:
        tache.erreur = "Calcul annulé."
    except Exception as erreur:  # noqa: BLE001 — toute erreur de calcul est rapportée à l'utilisateur
        tache.erreur = f"Le calcul a échoué : {erreur}"
        # format_exc : texte complet de la pile d'appels (aide au diagnostic).
        tache.trace = traceback.format_exc()
    finally:
        # « finally » : exécuté dans tous les cas ; durée totale et drapeau de fin (lu par l'interface).
        tache.duree = time.perf_counter() - tache.debut
        tache.termine = True


def en_cours():
    """Vrai si un calcul de cette session est en cours."""
    tache = st.session_state.get("tache")
    return tache is not None and not tache.termine


def _demarrer(tache):
    """Mémorise la tâche dans l'état de session et lance son fil de calcul."""
    st.session_state["tache"] = tache
    # daemon=True : le fil n'empêche pas l'arrêt du serveur ; name : nom lisible dans les outils de diagnostic.
    threading.Thread(target=_executer, args=(tache,), daemon=True, name=f"calcul-{tache.volet.CLE}").start()


def lancer_simulation(volet, params, estimation, meta):
    """Lance un calcul complet ; s'il a déjà été fait avec les mêmes paramètres, le restitue aussitôt."""
    # Mémoire des résultats : paramètres identiques → nouvelle entrée d'historique sans recalcul.
    sortie = stockage.chercher(volet.CLE, params)
    if sortie is not None:
        entree = stockage.ajouter(volet.CLE, params, sortie, volet.etiquette(params), meta)
        # Le nouveau calcul devient le calcul affiché (clé du menu « Calcul affiché » de la zone de résultats).
        st.session_state[f"calcul_affiche_{volet.CLE}"] = entree["id"]
        st.session_state["message_calcul"] = ("info", "Ces paramètres avaient déjà été calculés : résultat restitué "
                                                      "instantanément.")
        return
    _demarrer(Tache("simulation", volet, params, estimation, meta))


def lancer_convergence(volet, entree, params_fins, description, estimation):
    """Lance le calcul raffiné qui vérifie la convergence d'un calcul de l'historique."""
    _demarrer(Tache("convergence", volet, params_fins, estimation, {"description": description}, entree))


def _annuler(tache):
    """Fonction du bouton « Annuler » : le calcul s'arrête à sa prochaine mise à jour de progression."""
    tache.annuler = True


def suivre():
    """Affiche le calcul en cours (indicateur de chargement st.spinner, barre de progression, bouton d'annulation),
    attend sa fin, enregistre le résultat puis relance le script pour l'afficher."""
    tache = st.session_state.get("tache")
    if tache is None:
        return
    if not tache.termine:
        # Titre : nature du calcul et volet concerné (il peut différer du volet affiché).
        nature = "Vérification de convergence" if tache.nature == "convergence" else "Simulation"
        # st.container(border=True) : encadré qui regroupe les éléments de suivi.
        with st.container(border=True):
            st.markdown(f"**⏳ {nature} — {tache.volet.LIBELLE}**")
            barre = st.progress(tache.fraction, text=tache.texte)
            st.button("⏹️ Annuler le calcul", on_click=_annuler, args=(tache,), key="annuler_calcul")
            st.caption("Le calcul continue en arrière-plan : vous pouvez lire la documentation en attendant.")
            # st.spinner : indicateur de chargement animé, avec le temps écoulé (show_time=True).
            with st.spinner("Calcul en cours…", show_time=True):
                # Boucle d'attente : rafraîchit la barre toutes les 0,25 s jusqu'à la fin du calcul.
                while not tache.termine:
                    barre.progress(tache.fraction, text=tache.texte)
                    time.sleep(0.25)
    # Calcul terminé : enregistrement, puis nouvelle exécution du script qui affiche les résultats.
    _terminer(tache)
    st.rerun()


def _terminer(tache):
    """Enregistre le résultat (historique, mémoire, calibration) ou l'erreur, sous forme de message à afficher."""
    # La tâche quitte l'état de session (« pop » : pas d'erreur si elle a déjà été retirée).
    st.session_state.pop("tache", None)
    if tache.erreur:
        st.session_state["message_calcul"] = ("erreur", tache.erreur, tache.trace)
        return
    sortie = tache.sortie
    sortie["duree"] = tache.duree
    if tache.nature == "simulation":
        # Mémoire (recalcul instantané) et historique (affichage, comparaison) ; le nouveau calcul est affiché.
        stockage.memoriser(tache.volet.CLE, tache.params, sortie)
        entree = stockage.ajouter(tache.volet.CLE, tache.params, sortie, tache.volet.etiquette(tache.params), tache.meta)
        st.session_state[f"calcul_affiche_{tache.volet.CLE}"] = entree["id"]
        message = f"Calcul terminé en {duree_lisible(tache.duree)}."
    else:
        # Convergence : grandeurs du calcul raffiné rangées dans l'entrée vérifiée.
        tache.entree["convergence"] = {"params": tache.params, "description": tache.meta["description"],
                                       "grandeurs": sortie.get("grandeurs", {}), "duree": tache.duree}
        message = f"Vérification de convergence terminée en {duree_lisible(tache.duree)}."
    # Calibration des estimations à partir des durées mesurées (type de coût fourni par le calcul).
    type_cout = sortie.get("type_cout") or (tache.estimation or {}).get("type_cout") or tache.volet.CLE
    calibration.apprendre(type_cout, tache.estimation, sortie.get("mesures"))
    st.session_state["message_calcul"] = ("succes", message)


def afficher_message():
    """Affiche une seule fois le message laissé par le dernier calcul (succès, information ou erreur)."""
    message = st.session_state.pop("message_calcul", None)
    if message is None:
        return
    if message[0] == "erreur":
        st.error(message[1], icon="❌")
        # Pile d'appels repliée (utile pour signaler un problème).
        if message[2]:
            with st.expander("Détails techniques de l'erreur"):
                st.code(message[2], language="text")
    elif message[0] == "info":
        st.info(message[1], icon="⚡")
    else:
        # st.toast : notification éphémère.
        st.toast(message[1], icon="✅")
