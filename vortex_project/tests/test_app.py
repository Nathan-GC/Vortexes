"""
Tests de l'application web (app.py et paquet appli/), exécutée sans navigateur par streamlit.testing.v1.AppTest.

Onglet de documentation (pages, ancres des sommaires, glossaire, boutons de la page d'accueil), volet analytique,
scénarios guidés, puis exécution complète de chaque volet de simulation avec des réglages légers (calcul en
arrière-plan, figures, métriques, mémoire des résultats), comparaison et vérification de convergence.
Les tests de calcul sont marqués « lent » : `pytest -m "not lent"` les écarte pour une vérification rapide.
Ils sont tous ignorés si streamlit n'est pas installé.
"""

# Path : chemin du fichier app.py, situé dans le dossier parent de tests/.
from pathlib import Path

# pytest : cadre de test (importorskip, fixture, parametrize, mark).
import pytest

# pytest.importorskip : ignore tout le module si streamlit (ou plotly, utilisé par les vues 3D) est absent.
pytest.importorskip("streamlit")
pytest.importorskip("plotly")
# AppTest : exécute un script Streamlit en mémoire et donne accès à ses éléments et widgets.
from streamlit.testing.v1 import AppTest  # noqa: E402

# Chemin absolu de l'application et libellé de l'onglet de simulation (ONGLET_SIM dans appli/commun.py).
APP = str(Path(__file__).resolve().parents[1] / "app.py")
ONGLET_SIM = "🚀 Simulation Interactive"


# autouse=True : la fixture s'applique à tous les tests du module.
@pytest.fixture(autouse=True)
def calibration_isolee(tmp_path, monkeypatch):
    """Les calculs des tests n'altèrent pas la calibration des durées propre au PC : fichier redirigé vers un
    dossier temporaire et coefficients rechargés depuis les valeurs par défaut."""
    from appli import calibration
    # monkeypatch.setattr : remplace un attribut le temps du test (restauré automatiquement ensuite).
    monkeypatch.setattr(calibration, "FICHIER", tmp_path / "calibration.json")
    monkeypatch.setattr(calibration, "_DONNEES", None)


def lancer_app(onglet_simulation=False):
    """Première exécution de l'application, sur l'onglet de documentation ou de simulation."""
    # default_timeout : durée maximale (s) d'une exécution du script avant échec du test.
    at = AppTest.from_file(APP, default_timeout=180)
    if onglet_simulation:
        # L'onglet actif est un état de session (clé « onglet » de st.tabs) : on le fixe avant l'exécution.
        at.session_state["onglet"] = ONGLET_SIM
    return at.run()


def test_onglet_documentation_affiche_toutes_les_pages():
    at = lancer_app()
    assert not at.exception
    # Page d'accueil par défaut (st.radio de clé « document »), puis chaque page tour à tour ; le widget est
    # relu après chaque exécution (at.radio(key=…)), l'élément d'une exécution précédente étant périmé.
    assert at.radio(key="document").value == "🏠 Accueil"
    for nom in at.radio(key="document").options:
        at.radio(key="document").set_value(nom).run()
        assert not at.exception, nom
        # Aucune figure référencée par un document ne manque sur le disque.
        assert not any("Figure absente" in str(m.value) for m in at.markdown), nom
    # Rapport théorique : titres ancrés « à la GitHub » comme les liens du sommaire.
    at.radio(key="document").set_value("Fondements théoriques").run()
    ancres = {h.proto.anchor for h in at.header}
    assert {"1-équations-fondamentales", "3-formation-des-anneaux--le-nombre-de-formation-f--4",
            "5-diagnostics-et-visualisations-avancés"} <= ancres
    assert any("(#1-équations-fondamentales)" in str(m.value) for m in at.sidebar.markdown)
    # Glossaire : une définition au moins par terme clé.
    at.radio(key="document").set_value("📖 Glossaire").run()
    texte = " ".join(str(m.value) for m in at.markdown)
    assert "Critère Q" in texte and "Nombre de formation F*" in texte


def test_bouton_de_l_accueil_ouvre_un_scenario():
    at = lancer_app()
    # Bouton « Anneau de référence » du parcours conseillé : onglet de simulation, volet et scénario appliqués.
    at.button(key="acc_anneau").click().run()
    assert not at.exception
    assert at.session_state["onglet"] == ONGLET_SIM
    assert at.session_state["simulation"] == "anneau"
    assert at.session_state["scenario_anneau"] == "reference"
    assert at.select_slider(key="an_Re").value == 1000


def test_volet_formules_analytiques():
    at = lancer_app(onglet_simulation=True)
    at.selectbox(key="simulation").set_value("theorie").run()
    assert not at.exception
    valeurs = {m.label: m.value for m in at.metric}
    # Vitesse de Kelvin (noyau uniforme, Γ = R = 1, a/R = 0,1) : [ln(80) − 1/4]/(4π) = 0,32882.
    assert valeurs["U (noyau choisi)"] == "0.3288"
    # Nombre de formation : √(π/2)/0,33 = 3,80.
    assert valeurs["Nombre de formation F*"] == "3.80"


def test_scenario_guide_applique_ses_reglages():
    at = lancer_app(onglet_simulation=True)
    # Scénario « fluide très visqueux » : Re passe à 200 ; l'encadré « à observer » et la question s'affichent.
    at.selectbox(key="scenario_anneau").set_value("re_faible").run()
    assert not at.exception
    assert at.select_slider(key="an_Re").value == 200
    assert any("Fluide très visqueux" in str(i.value) for i in at.info)
    assert any(r.key == "prediction_anneau_re_faible" for r in at.radio)


# Réglages légers de chaque volet : (type de widget, clé, valeur).
REGLAGES = {
    "anneau": [("slider", "an_T", 1.0), ("select_slider", "an_dr", 0.06), ("slider", "an_a", 0.3),
               ("toggle", "an_iso", False), ("toggle", "an_phi", False)],
    "hill": [("selectbox", "an_ci", "hill"), ("slider", "an_T", 1.0), ("select_slider", "an_dr", 0.06),
             ("toggle", "an_iso", False), ("toggle", "an_phi", False)],
    "saute_mouton": [("slider", "sm_T", 1.5), ("select_slider", "sm_dr", 0.06), ("slider", "sm_aA", 0.3),
                     ("slider", "sm_aB", 0.3), ("toggle", "sm_iso", False), ("toggle", "sm_phi", False)],
    "collision": [("slider", "sm_GB", -1.0), ("slider", "sm_sep", 2.0), ("slider", "sm_T", 1.5),
                  ("select_slider", "sm_dr", 0.06), ("slider", "sm_aA", 0.3), ("slider", "sm_aB", 0.3),
                  ("toggle", "sm_iso", False), ("toggle", "sm_phi", False)],
    "filaments": [("slider", "fi_N_std", 32), ("slider", "fi_T_saute_mouton", 2.0)],
    "moteurs": [("slider", "mo_n", 10)],
}
# Volet de l'application correspondant à chaque cas.
VOLET = {"anneau": "anneau", "hill": "anneau", "saute_mouton": "saute_mouton", "collision": "saute_mouton",
         "filaments": "filaments", "moteurs": "moteurs"}


def preparer(cas):
    """Application ouverte sur le volet du cas, réglages légers appliqués."""
    at = lancer_app(onglet_simulation=True)
    at.selectbox(key="simulation").set_value(VOLET[cas]).run()
    for type_widget, cle, valeur in REGLAGES[cas]:
        # getattr(at, "slider")(key=...) : widget du type indiqué portant cette clé ; set_value + run : application.
        getattr(at, type_widget)(key=cle).set_value(valeur)
        at.run()
    assert not at.exception
    return at


@pytest.mark.lent
@pytest.mark.parametrize("cas", list(REGLAGES))
def test_volet_de_simulation_complet(cas):
    from appli import stockage
    at = preparer(cas)
    volet = VOLET[cas]
    avant = len(stockage.historique(volet))
    # Clic sur le bouton : calcul en arrière-plan, attente de la fin, enregistrement dans l'historique.
    at.button(key="lancer").click().run(timeout=300)
    assert not at.exception
    entrees = stockage.historique(volet)
    assert len(entrees) == min(avant + 1, stockage.TAILLE_HISTORIQUE)
    sortie = entrees[0]["sortie"]
    # Figures PNG non vides, métriques affichées, données exportables.
    assert len(sortie["figures"]) >= 2
    assert all(f["png"].startswith(b"\x89PNG") for f in sortie["figures"])
    assert len(at.metric) >= 4
    assert len(sortie["donnees"]) > 0
    if cas in ("anneau", "hill", "saute_mouton", "collision"):
        # Images de l'animation interactive : t = 0 et 24 instants.
        assert len(sortie["animation"]["images"]) == 25
    # Relance avec les mêmes réglages : résultat restitué depuis la mémoire, sans recalcul.
    at.button(key="lancer").click().run(timeout=60)
    assert any("déjà été calculés" in str(i.value) for i in at.info)
    assert stockage.historique(volet)[0]["sortie"] is sortie


@pytest.mark.lent
def test_sous_onglets_comparaison_et_convergence():
    from appli import stockage
    at = preparer("filaments")
    # Deux calculs différents (durées 2 et 3) pour la comparaison.
    at.button(key="lancer").click().run(timeout=300)
    at.slider(key="fi_T_saute_mouton").set_value(3.0).run()
    at.button(key="lancer").click().run(timeout=300)
    assert not at.exception
    # Chaque sous-onglet est construit sans erreur (onglet actif fixé par son état de session).
    for onglet in ["🎞️ Animation", "🧊 Vue 3D", "📈 Courbes", "📊 Comparer", "🔬 Convergence", "💾 Données et export"]:
        at.session_state["sous_onglets_filaments"] = onglet
        at.run()
        assert not at.exception, onglet
    # Vérification de convergence (raffinement × √2) : grandeurs du calcul raffiné rangées dans l'entrée.
    at.session_state["sous_onglets_filaments"] = "🔬 Convergence"
    at.run()
    bouton = next(b for b in at.button if b.key and b.key.startswith("convergence_"))
    at.session_state["sous_onglets_filaments"] = "🔬 Convergence"
    bouton.click().run(timeout=300)
    assert not at.exception
    conv = stockage.historique("filaments")[0]["convergence"]
    assert conv is not None and "1ʳᵉ traversée (t)" in conv["grandeurs"]
