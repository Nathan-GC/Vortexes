"""
appli/pedagogie.py — Aides pédagogiques de l'onglet de simulation.

* Scénarios guidés : réglages prêts à l'emploi et « ce qu'il faut observer » ;
* « Prédire puis vérifier » : question posée avant le lancement, réponse commentée après le calcul ;
* théorie associée : extrait de la section du rapport qui explique le volet (texte et formules) ;
* limites du modèle ;
* transposition des résultats adimensionnels en unités physiques (air, eau…).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# streamlit : interface ; modules de l'application : widgets et extraits de documentation.
import streamlit as st

from appli import documentation
from appli.commun import initialiser


# =============================================================================================
# Scénarios guidés
# =============================================================================================
def _appliquer_scenario(volet):
    """Fonction de rappel du menu des scénarios : écrit les réglages du scénario choisi dans l'état de session."""
    choix = st.session_state[f"scenario_{volet.CLE}"]
    if choix in volet.SCENARIOS:
        # Chaque clé est celle d'un widget du volet : il prendra cette valeur à l'exécution suivante.
        for cle, valeur in volet.SCENARIOS[choix]["reglages"].items():
            st.session_state[cle] = valeur


def selecteur_scenario(volet):
    """Menu des scénarios guidés du volet (panneau latéral) ; renvoie la clé du scénario choisi (ou « libre »)."""
    if not volet.SCENARIOS:
        return "libre"
    # Clé du menu des scénarios du volet, valeur initiale « libre ».
    cle = f"scenario_{volet.CLE}"
    initialiser(cle, "libre")
    # on_change : les réglages du scénario sont appliqués dès qu'on le choisit.
    st.selectbox("🎯 Scénario guidé", ["libre"] + list(volet.SCENARIOS), key=cle, persist_state="page",
                 format_func=lambda k: "Réglages libres" if k == "libre" else volet.SCENARIOS[k]["titre"],
                 on_change=_appliquer_scenario, args=(volet,),
                 help="Un scénario règle tous les paramètres pour illustrer un phénomène ; vous pouvez ensuite les "
                      "modifier librement.")
    return st.session_state[cle]


def encadre_scenario(volet, choix):
    """Encadré « ce qu'il faut observer » du scénario choisi, avec mention des réglages modifiés depuis."""
    # Scénario choisi (None pour les réglages libres).
    scenario = volet.SCENARIOS.get(choix)
    if scenario is None:
        return
    st.info(f"**🎯 {scenario['titre']}** — {scenario['observer']}", icon="🔭")
    # Réglages modifiés depuis l'application du scénario (comparaison avec l'état de session).
    if any(st.session_state.get(k) != v for k, v in scenario["reglages"].items()):
        st.caption("Certains réglages ont été modifiés depuis le choix du scénario.")


def prediction(volet, choix):
    """Question « prédire puis vérifier » du scénario (avant le lancement) ; renvoie la réponse choisie ou None."""
    # Question du scénario (None s'il n'en pose pas).
    question = volet.SCENARIOS.get(choix, {}).get("question")
    if question is None:
        return None
    cle = f"prediction_{volet.CLE}_{choix}"
    # index=None : aucune réponse cochée au départ.
    st.radio(f"🤔 **Avant de lancer, prédisez :** {question['texte']}", question["choix"], index=None, key=cle,
             horizontal=True)
    return st.session_state.get(cle)


def retour_prediction(volet, entree):
    """Après le calcul : compare la prédiction enregistrée avec la bonne réponse et rappelle l'explication."""
    # Scénario et prédiction enregistrés avec le calcul.
    meta = entree.get("meta", {})
    question = volet.SCENARIOS.get(meta.get("scenario"), {}).get("question")
    if question is None or meta.get("prediction") is None:
        return
    # Bonne réponse (encadré vert) ou correction (encadré orange).
    if meta["prediction"] == question["reponse"]:
        st.success(f"Votre prédiction « {meta['prediction']} » était la bonne. {question['explication']}", icon="✅")
    else:
        st.warning(f"Vous aviez prédit « {meta['prediction']} » ; la bonne réponse était « {question['reponse']} ». "
                   f"{question['explication']}", icon="💡")


# =============================================================================================
# Théorie associée et limites du modèle
# =============================================================================================
def encadre_theorie(volet):
    """Encadré repliable avec l'extrait du rapport qui explique le volet (texte et formules, sans les figures)."""
    if not volet.THEORIE:
        return
    # Numéros des sections citées dans le titre de l'encadré.
    numeros = ", ".join(f"§ {numero}" for _, numero in volet.THEORIE)
    with st.expander(f"📖 Théorie associée ({numeros})"):
        for document, numero in volet.THEORIE:
            # Extrait de la section : titre et blocs de texte.
            titre, blocs = documentation.extrait_section(document, numero)
            if titre is None:
                continue
            st.markdown(f"#### {titre} *({document})*")
            # images=False : figures omises (elles restent dans l'onglet de documentation).
            documentation.afficher_blocs(blocs, images=False)
        st.caption("Figures et suite du rapport : onglet « 📚 Théorie & Documentation ».")


def encadre_limites(volet):
    """Encadré repliable listant les hypothèses et limites du modèle simulé."""
    if not volet.LIMITES:
        return
    with st.expander("⚠️ Limites du modèle"):
        st.markdown("\n".join(f"* {ligne}" for ligne in volet.LIMITES))


# =============================================================================================
# Unités physiques
# =============================================================================================
# Viscosités cinématiques ν (m²/s) de quelques fluides à 20 °C, et tournure « dans … » pour le texte.
FLUIDES = {"Air (20 °C)": (1.5e-5, "dans l'air"), "Eau (20 °C)": (1.0e-6, "dans l'eau"),
           "Huile de silicone (50 cSt)": (5.0e-5, "dans une huile de silicone"),
           "Glycérine (20 °C)": (1.2e-3, "dans la glycérine")}


def convertisseur(unites, cle):
    """Transpose un calcul adimensionnel en unités physiques pour un fluide et un rayon réels choisis.

    ``unites`` : {"Re", "R" (rayon simulé), "Gamma" (circulation simulée), "U" (vitesse finale simulée),
    "t_end", "distance" (déplacement simulé)}. Avec Re = Γ/ν imposé, le choix du fluide (ν) et du rayon réel fixe
    les échelles : Γ_réelle = Re ν, longueur L = R_réel/R, temps L²Γ/Γ_réelle, vitesse (Γ_réelle/Γ)/L.
    """
    with st.expander("🌍 Transposer en unités physiques"):
        # Choix du fluide et du rayon réel de l'anneau.
        col1, col2 = st.columns(2)
        fluide = col1.selectbox("Fluide", list(FLUIDES), key=f"fluide_{cle}")
        R_cm = col2.slider("Rayon réel de l'anneau (cm)", 0.5, 50.0, 5.0, 0.5, key=f"rayon_reel_{cle}")
        nu, dans = FLUIDES[fluide]
        # Échelles : longueur (m par unité simulée), circulation réelle, temps et vitesse.
        echelle_L = (R_cm / 100.0) / unites["R"]
        gamma_reelle = unites["Re"] * nu
        echelle_G = gamma_reelle / unites["Gamma"]
        echelle_t = echelle_L**2 / echelle_G
        echelle_U = echelle_G / echelle_L
        # str.capitalize : majuscule initiale (« Dans l'air »).
        st.markdown(f"{dans.capitalize()}, un anneau de {R_cm:g} cm de rayon à Re_Γ = {unites['Re']:g} a une "
                    f"circulation Γ = {gamma_reelle * 1e4:.3g} cm²/s. L'unité de temps de la simulation vaut "
                    f"{echelle_t:.3g} s et l'unité de vitesse {echelle_U * 100:.3g} cm/s.")
        # Vitesse, durée et distance en unités physiques.
        a, b, c = st.columns(3)
        a.metric("Vitesse finale", f"{unites['U'] * echelle_U * 100:.3g} cm/s", border=True)
        b.metric("Durée simulée", f"{unites['t_end'] * echelle_t:.3g} s", border=True)
        c.metric("Distance parcourue", f"{unites['distance'] * echelle_L * 100:.3g} cm", border=True)
        st.caption("Un même calcul adimensionnel décrit tous les anneaux de même Re_Γ et de mêmes proportions a/R : "
                   "seules les échelles de temps et de vitesse changent avec le fluide et la taille.")
