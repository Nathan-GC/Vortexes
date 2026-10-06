"""
appli/simulation.py — Onglet « 🚀 Simulation Interactive ».

Panneau latéral : choix du volet, scénario guidé, paramètres et bouton de lancement. Zone principale :
présentation et formule clé, théorie associée et limites du modèle, « ce qu'il faut observer », garde-fous et
durée estimée, prédiction, suivi du calcul en cours puis résultats.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

import streamlit as st

from appli import commun as c
from appli import pedagogie, resultats, taches
from appli.volets import VOLETS


def _afficher_estimation(est):
    """Légende de la durée estimée (calcul + figures), de la taille du calcul et de l'état de la calibration."""
    pas = f", ≈ {est['n_pas']:.0f} pas de temps" if est.get("n_pas") else ""
    peclet = f", Péclet de maille ≈ {est['peclet']:.0f}" if est.get("peclet") else ""
    calibre = (f" Estimation recalibrée sur {est['mesures']} calcul(s) faits sur ce PC." if est["mesures"]
               else " Estimation de référence, recalibrée après chaque calcul.")
    st.caption(f"⏱️ Durée estimée : ≈ {c.duree_lisible(est['calcul'])} de calcul + ≈ {c.duree_lisible(est['figures'])} "
               f"de figures ({est['texte']}{pas}{peclet}).{calibre}")
    if est["calcul"] + est["figures"] > 180:
        st.warning("Calcul long : réduisez la durée simulée ou la résolution pour un premier essai (vous pourrez "
                   "lire la documentation pendant le calcul).", icon="⏳")


def page_simulation():
    """Contenu de l'onglet de simulation."""
    with st.sidebar:
        st.header("⚙️ Paramètres")
        # Menu des volets (libellés des modules du registre).
        cle = c.menu("Volet à simuler", list(VOLETS), "anneau", "simulation", format_func=lambda k: VOLETS[k].LIBELLE)
        volet = VOLETS[cle]
        # Scénario guidé (applique ses réglages), puis widgets du volet.
        scenario = pedagogie.selecteur_scenario(volet)
        params = volet.controles()
        lancer = False
        if not volet.INSTANTANE:
            # Bouton désactivé pendant un calcul ; st.button renvoie True à l'exécution qui suit le clic.
            lancer = st.button("▶️ Lancer la simulation", type="primary", key="lancer", width="stretch",
                               disabled=taches.en_cours())
            st.caption("Le calcul tourne en arrière-plan : vous pouvez lire la documentation pendant ce temps.")
    # Zone principale : titre et présentation du volet.
    st.subheader(volet.LIBELLE)
    st.markdown(volet.PRESENTATION)
    # Formule clé du volet (st.latex : rendu KaTeX centré).
    st.latex(volet.FORMULE)
    # Théorie associée et limites du modèle, côte à côte (encadrés repliables).
    gauche, droite = st.columns(2)
    with gauche:
        pedagogie.encadre_theorie(volet)
    with droite:
        pedagogie.encadre_limites(volet)
    # Volet analytique : résultats recalculés à chaque exécution, sans bouton.
    if volet.INSTANTANE:
        volet.afficher(params)
        return
    # Scénario guidé : « ce qu'il faut observer ».
    pedagogie.encadre_scenario(volet, scenario)
    # Garde-fous numériques et durée estimée, réévalués à chaque modification d'un réglage.
    alertes, est = volet.verifier(params)
    for alerte in alertes:
        st.warning(alerte, icon="⚠️")
    _afficher_estimation(est)
    # Question « prédire puis vérifier » ; message du dernier calcul.
    prediction = pedagogie.prediction(volet, scenario)
    taches.afficher_message()
    if lancer:
        # Réglages des widgets du volet (clés à son préfixe), enregistrés avec le calcul pour pouvoir les recharger.
        widgets = {k: v for k, v in st.session_state.items() if isinstance(k, str) and k.startswith(volet.PREFIXE)}
        taches.lancer_simulation(volet, params, est, {"scenario": scenario, "prediction": prediction, "widgets": widgets})
        st.rerun()
    # Calcul en cours : indicateur de chargement, barre et bouton d'annulation (attend la fin du calcul).
    taches.suivre()
    resultats.afficher_resultats(volet, params)
