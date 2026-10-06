"""
appli/accueil.py — Page d'accueil : présentation, schéma d'un anneau, animation, parcours conseillé, accès direct
aux volets de simulation et résultats marquants du projet.
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# matplotlib : schéma de l'anneau (cercles, ellipse, flèches) ; numpy : surface du tore.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402

from appli.commun import ONGLET_SIM, PROJET, VERROU, png  # noqa: E402
from src import visualization as viz  # noqa: E402


# st.cache_data : le schéma est tracé une seule fois puis gardé en mémoire (image PNG).
@st.cache_data(show_spinner=False)
def schema_anneau():
    """Schéma d'un anneau tourbillonnaire : (a) tore de vorticité en 3D, (b) coupe méridienne annotée."""
    # Rayon de l'anneau et rayon du cœur du schéma (unités arbitraires).
    R, a = 1.0, 0.27
    with VERROU:
        fig = plt.figure(figsize=(11.0, 4.3))
        # (a) Tore : paramétrage (u autour de l'axe, v autour du cœur) ; plot_surface : surface ombrée.
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 90), np.linspace(0, 2 * np.pi, 40))
        X, Y, Z = (R + a * np.cos(v)) * np.cos(u), (R + a * np.cos(v)) * np.sin(u), a * np.sin(v)
        ax1.plot_surface(X, Y, Z, color=viz.SERIES[0], alpha=0.9, linewidth=0, antialiased=True, shade=True)
        # Flèche de la vitesse de propagation U le long de l'axe (ax.quiver en 3D).
        ax1.quiver(0, 0, 0.05, 0, 0, 1.15, color=viz.SERIES[1], linewidth=2.2, arrow_length_ratio=0.18)
        ax1.text(0.12, 0.0, 1.15, "U", color=viz.SERIES[1], fontsize=13, fontweight="bold")
        # Proportions de la boîte, axes masqués, point de vue oblique.
        ax1.set_box_aspect((1, 1, 0.6))
        ax1.set_axis_off()
        ax1.view_init(elev=24, azim=-60)
        ax1.set_title("(a) Un tore de vorticité qui avance seul", loc="left")
        # (b) Coupe méridienne : axe de symétrie, cœurs (rouge : ω > 0, bleu : ω < 0), bulle transportée.
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axvline(0.0, color=viz.INK["axis"], lw=1.0, ls="--")
        ax2.text(0.05, -1.05, "axe de symétrie z", color=viz.INK["muted"], fontsize=9)
        ax2.add_patch(Ellipse((0.0, 0.0), 2 * (R + 0.55), 1.75, fill=False, ls=":", lw=1.2, color=viz.INK["secondary"]))
        ax2.text(R + 0.25, 0.72, "bulle de fluide\ntransportée", color=viz.INK["secondary"], fontsize=9)
        for x0, couleur in ((R, viz.SERIES[7]), (-R, viz.SERIES[0])):
            ax2.add_patch(Circle((x0, 0.0), a, color=couleur, alpha=0.9))
        # Sens de rotation des cœurs : le fluide monte entre eux (près de l'axe) et redescend à l'extérieur
        # (annotate avec une flèche courbe « arc3 »).
        for x0, signe in ((R, 1.0), (-R, -1.0)):
            ax2.annotate("", xy=(x0 + signe * 0.05, -1.75 * a), xytext=(x0 + signe * 0.05, 1.75 * a),
                         arrowprops=dict(arrowstyle="->", lw=1.6, color=viz.INK["primary"],
                                         connectionstyle=f"arc3,rad={-0.9 * signe}"))
        ax2.text(R + 0.55, -0.35, "Γ", fontsize=15, fontweight="bold", color=viz.INK["primary"])
        # Rayon R de l'anneau (double flèche de l'axe au centre du cœur) et rayon a du cœur (trait blanc).
        ax2.annotate("", xy=(R, -0.62), xytext=(0.0, -0.62), arrowprops=dict(arrowstyle="<->", lw=1.2, color=viz.INK["primary"]))
        ax2.text(0.5 * R, -0.58, "R", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax2.plot([R, R + a], [0.0, 0.0], color="white", lw=1.6)
        ax2.text(R + 0.5 * a, 0.03, "a", color="white", ha="center", va="bottom", fontsize=11, fontweight="bold")
        # Écoulement induit et vitesse de propagation U le long de l'axe.
        ax2.annotate("", xy=(0.0, 0.95), xytext=(0.0, -0.35), arrowprops=dict(arrowstyle="-|>", lw=2.0, color=viz.SERIES[1]))
        ax2.text(0.08, 0.85, "U", color=viz.SERIES[1], fontsize=13, fontweight="bold")
        # Cadre : même échelle sur les deux axes, axes masqués.
        ax2.set_xlim(-1.9, 2.15)
        ax2.set_ylim(-1.15, 1.25)
        ax2.set_aspect("equal")
        ax2.set_axis_off()
        ax2.set_title("(b) Coupe méridienne : rayon R, cœur a, circulation Γ", loc="left")
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.9, wspace=0.05)
        return png(fig)


def ouvrir_volet(cle, scenario=None):
    """Fonction de bouton : ouvre l'onglet de simulation sur un volet (et éventuellement un scénario guidé)."""
    # Changement d'onglet et de volet par l'état de session (clés des widgets st.tabs et st.selectbox).
    st.session_state["onglet"] = ONGLET_SIM
    st.session_state["simulation"] = cle
    if scenario is not None:
        # Import local (évite un import circulaire au chargement) ; application des réglages du scénario.
        from appli.volets import VOLETS
        st.session_state[f"scenario_{cle}"] = scenario
        for widget, valeur in VOLETS[cle].SCENARIOS[scenario]["reglages"].items():
            st.session_state[widget] = valeur


def ouvrir_document(nom):
    """Fonction de bouton : affiche un document de l'onglet de documentation."""
    st.session_state["document"] = nom


def page_accueil():
    """Contenu de la page d'accueil."""
    # Import local : registre des volets (titres et résumés des cartes).
    from appli.volets import VOLETS
    # Présentation : texte à gauche, animation à droite.
    st.header("Bienvenue")
    gauche, droite = st.columns([3, 2], gap="large")
    with gauche:
        st.markdown(
            "Un **anneau tourbillonnaire** est un tore de fluide en rotation qui avance seul : rond de fumée, anneau "
            "de bulles des dauphins, jet pulsé des méduses ou du cœur. Sa vitesse, sa durée de vie et ses "
            "interactions (deux anneaux qui se dépassent, se percutent ou fusionnent) se calculent à partir de "
            "quelques grandeurs : le **rayon R**, le **rayon du cœur a**, la **circulation Γ** et la **viscosité ν** "
            "du fluide, réunies dans le nombre de Reynolds Re_Γ = Γ/ν.")
        st.latex(r"U \simeq \frac{\Gamma}{4\pi R}\left[\ln\frac{8R}{a} - C\right]")
        st.markdown("Ce projet réunit la **théorie** (rapports et formules), deux **simulateurs** — filaments 3D en "
                    "fluide parfait, Navier–Stokes axisymétrique visqueux — et une étude des **moteurs à vortex**.")
    with droite:
        # Animation d'introduction : vidéo du saute-mouton produite par main.py (sinon GIF des exemples).
        video = PROJET / "outputs" / "animations" / "leapfrog.mp4"
        gif = PROJET / "exemples" / "resultats" / "03_filaments" / "animation.gif"
        if video.exists():
            # Lecture à la demande (pas de lecture automatique, qui occuperait le processeur en permanence).
            st.video(str(video), loop=True, muted=True)
            st.caption("▶ Deux anneaux qui se dépassent (saute-mouton), méthode des filaments 3D.")
        elif gif.exists():
            st.image(str(gif), caption="Deux anneaux qui se dépassent (saute-mouton).", width="stretch")
    # Schéma annoté (tracé une fois puis gardé en mémoire).
    st.image(schema_anneau(), width="stretch", alt="Schéma d'un anneau tourbillonnaire : tore, rayon R, cœur a, "
                                                    "circulation Γ et vitesse de propagation U.")
    # Parcours conseillé : quatre étapes, chacune avec un bouton qui y mène directement.
    st.subheader("Par où commencer")
    etapes = st.columns(4)
    with etapes[0]:
        st.markdown("**1. Comprendre** — lire les sections 1 et 2 des fondements théoriques.")
        st.button("📖 Ouvrir la théorie", key="acc_theorie", on_click=ouvrir_document, args=("Fondements théoriques",),
                  width="stretch")
    with etapes[1]:
        st.markdown("**2. Simuler** — lancer l'anneau visqueux de référence et le comparer à Saffman.")
        st.button("🌀 Anneau de référence", key="acc_anneau", on_click=ouvrir_volet, args=("anneau", "reference"),
                  width="stretch")
    with etapes[2]:
        st.markdown("**3. Comparer** — relancer avec un autre Reynolds, puis ouvrir l'onglet « Comparer ».")
        st.button("🔁 Effet du Reynolds", key="acc_reynolds", on_click=ouvrir_volet, args=("anneau", "re_faible"),
                  width="stretch")
    with etapes[3]:
        st.markdown("**4. Explorer en 3D** — faire tourner les filaments du saute-mouton à la souris.")
        st.button("🧵 Saute-mouton 3D", key="acc_filaments", on_click=ouvrir_volet, args=("filaments", "saute_mouton"),
                  width="stretch")
    # Cartes des volets : titre, résumé et bouton d'accès (deux cartes par rangée).
    st.subheader("Les volets de simulation")
    # Clés des volets, dans l'ordre du registre.
    cles = list(VOLETS)
    for debut in range(0, len(cles), 2):
        for colonne, cle in zip(st.columns(2), cles[debut:debut + 2]):
            with colonne.container(border=True):
                st.markdown(f"**{VOLETS[cle].LIBELLE}**")
                st.caption(VOLETS[cle].RESUME)
                st.button("Ouvrir ce volet", key=f"carte_{cle}", on_click=ouvrir_volet, args=(cle,))
    # Résultats marquants (repris du README et des rapports).
    st.subheader("Résultats marquants")
    st.markdown(
        "* **Anneau visqueux** : la vitesse simulée suit la formule de Saffman + Fukumoto–Moffatt à quelques pour cent "
        "près, et l'énergie suit E₀ − ν∫Ω dt à 10⁻⁵ E₀ près (aucune dissipation numérique).\n"
        "* **Saute-mouton visqueux** (Re_Γ = 3000) : deux traversées (t = 1,45 et 3,92), puis fusion des cœurs à "
        "t ≈ 5,4, quand l'écart entre cœurs passe sous ≈ 4 rayons de cœur.\n"
        "* **Filaments 3D** : la vitesse propre converge vers la forme close exacte de Rosenhead–Moore ; le "
        "saute-mouton suit le modèle coaxial à 0,2 % près.\n"
        "* **Moteurs** : le propulseur pulsé gagne jusqu'à ~8 % de rendement sur le jet continu, avec un optimum "
        "vers L/D ≈ F* ≈ 4.")
