"""
app.py — Application web Streamlit du projet « Vortex toroïdaux & moteurs à vortex ».

Lancement, depuis le dossier vortex_project (Windows / PowerShell) :

    .venv\\Scripts\\python -m streamlit run app.py

Le navigateur s'ouvre sur http://localhost:8501 ; Ctrl+C dans le terminal arrête le serveur.

* Onglet « 📚 Théorie & Documentation » : page d'accueil, README, rapports, tutoriel et glossaire, avec formules
  (KaTeX), figures, sommaire cliquable et versions PDF.
* Onglet « 🚀 Simulation Interactive » : cinq volets (anneau visqueux ou vortex de Hill, deux anneaux coaxiaux,
  filaments 3D, moteurs à vortex, formules analytiques), scénarios guidés, calcul en arrière-plan par les fonctions
  du paquet src/, puis métriques, figures, animations et vues 3D interactives, comparaison de calculs, vérification
  de convergence et fiche de résultats.

Le code de l'interface est rangé dans le paquet appli/ (un module par rôle, un module par volet).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# sys.path : dossiers où Python cherche les modules ; Path : chemins de fichiers.
import sys
from pathlib import Path

# streamlit : interface web.
import streamlit as st

# Dossier du projet (celui de app.py), ajouté en tête du chemin de recherche pour que « from src import … » et
# « from appli import … » fonctionnent quel que soit le dossier depuis lequel streamlit est lancé.
PROJET = Path(__file__).resolve().parent
if str(PROJET) not in sys.path:
    sys.path.insert(0, str(PROJET))

# Réglages de la page (premier appel Streamlit) : titre de l'onglet du navigateur, icône, pleine largeur ; le panneau
# latéral s'ouvre sur ordinateur et reste replié sur téléphone (« auto »).
st.set_page_config(page_title="Vortex toroïdaux", page_icon="🌀", layout="wide", initial_sidebar_state="auto")

# Modules de l'interface (importés après set_page_config) : onglets et constantes.
from appli import documentation, simulation  # noqa: E402
from appli.commun import ONGLET_DOC, ONGLET_SIM, SOUS_TITRE, TITRE_APPLI  # noqa: E402

st.title(TITRE_APPLI)
st.caption(SOUS_TITRE)
# Onglets principaux ; on_change="rerun" + key : l'onglet actif est connu (attribut .open), ce qui permet de ne
# construire que son contenu et d'adapter le panneau latéral ; l'état « onglet » peut aussi être changé par
# programme (boutons de la page d'accueil).
onglet_doc, onglet_sim = st.tabs([ONGLET_DOC, ONGLET_SIM], key="onglet", on_change="rerun")
if onglet_doc.open:
    with onglet_doc:
        documentation.page_documentation()
if onglet_sim.open:
    with onglet_sim:
        simulation.page_simulation()
