"""
appli/documentation.py — Onglet « 📚 Théorie & Documentation ».

Le panneau latéral propose la page d'accueil, les documents Markdown du projet (README, rapports, tutoriel) et le
glossaire, avec un sommaire cliquable et la version PDF. Les documents sont découpés en blocs (texte, titres,
figures) : les formules $…$ et $$…$$ sont rendues par KaTeX, les titres reçoivent les mêmes ancres que les liens
des sommaires (convention GitHub) et les figures s'affichent depuis docs/figures. Le module fournit aussi les
extraits de sections que l'onglet de simulation montre à côté de chaque volet (« théorie associée »).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# re : expressions régulières ; Path : chemins de fichiers.
import re
from pathlib import Path

# streamlit : interface ; modules de l'application : constantes, page d'accueil et glossaire.
import streamlit as st

from appli import accueil, glossaire
from appli.commun import PROJET, TITRE_APPLI

# Pages du panneau latéral : nom affiché → fichier Markdown (None : page construite par l'application).
DOCUMENTS = {
    "🏠 Accueil": None,
    "README": PROJET / "README.md",
    "Fondements théoriques": PROJET / "docs" / "theoretical_background.md",
    "Moteurs à vortex": PROJET / "docs" / "vortex_engines_analysis.md",
    "Tutoriel": PROJET / "docs" / "tutoriel.md",
    "📖 Glossaire": None,
}
# Titre Markdown (« ## Texte ») ; image seule sur sa ligne (« ![légende](chemin) ») ; lien vers un fichier local
# (ni http(s), ni ancre « # », ni mailto), qui ne peut pas s'ouvrir depuis l'application web.
TITRE_MD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
IMAGE_MD = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)\)\s*$")
LIEN_LOCAL_MD = re.compile(r"\[([^\]]+)\]\((?!https?://|#|mailto:)[^)\s]+\)")


def ancre_github(titre):
    """Ancre « à la GitHub » d'un titre : celle qu'utilisent les sommaires des documents (et docs/build_pdf.py)."""
    # Retrait des marques de code et d'emphase, minuscules, suppression de la ponctuation (\w reconnaît les
    # lettres accentuées), puis espaces → tirets.
    texte = re.sub(r"[`*]", "", titre).strip().lower()
    texte = re.sub(r"[^\w\- ]", "", texte)
    return texte.replace(" ", "-")


# st.cache_data : résultat mémorisé pour des arguments identiques ; la date de modification du fichier fait partie
# des arguments, si bien qu'un document modifié est relu automatiquement.
@st.cache_data(show_spinner=False)
def blocs_document(chemin, date_modification):
    """Découpe un document Markdown en blocs affichables : ("md", texte), ("titre", niveau, texte, ancre) ou
    ("image", chemin, légende). Les liens vers des fichiers locaux deviennent du texte simple."""
    source = Path(chemin)
    # blocs : résultat ; tampon : lignes de texte en attente ; dans_code : vrai à l'intérieur d'un bloc ```.
    blocs, tampon, dans_code = [], [], False

    def vider():
        # Ajoute le texte accumulé comme bloc Markdown (s'il n'est pas vide), puis vide le tampon.
        if "".join(tampon).strip():
            blocs.append(("md", "\n".join(tampon)))
        tampon.clear()

    # splitlines : lignes du fichier (lu en UTF-8).
    for ligne in source.read_text(encoding="utf-8").splitlines():
        # Délimiteur de bloc de code : bascule de l'état ; les lignes du bloc sont recopiées telles quelles
        # (un « # commentaire » de script n'est pas un titre).
        if ligne.lstrip().startswith("```"):
            dans_code = not dans_code
            tampon.append(ligne)
            continue
        if dans_code:
            tampon.append(ligne)
            continue
        # Titre : bloc à part, affiché avec une ancre identique à celle des liens du sommaire.
        m = TITRE_MD.match(ligne)
        if m:
            vider()
            blocs.append(("titre", len(m.group(1)), LIEN_LOCAL_MD.sub(r"\1", m.group(2)), ancre_github(m.group(2))))
            continue
        # Image seule sur sa ligne : chemin résolu par rapport au dossier du document.
        m = IMAGE_MD.match(ligne.strip())
        if m:
            vider()
            fichier = (source.parent / m.group(2)).resolve()
            if fichier.exists():
                blocs.append(("image", str(fichier), m.group(1)))
            else:
                blocs.append(("md", f"*Figure absente : `{m.group(2)}` (à produire avec `python main.py all --docs`).*"))
            continue
        # Ligne ordinaire : liens locaux remplacés par leur texte.
        tampon.append(LIEN_LOCAL_MD.sub(r"\1", ligne))
    vider()
    return blocs


def blocs_de(nom):
    """Blocs du document ``nom`` (clé de DOCUMENTS), avec la date du fichier comme clé de cache."""
    chemin = DOCUMENTS[nom]
    # stat().st_mtime : date de dernière modification du fichier.
    return blocs_document(str(chemin), chemin.stat().st_mtime)


def afficher_blocs(blocs, images=True):
    """Affiche une suite de blocs : Markdown (formules KaTeX), titres ancrés, figures (si ``images``)."""
    for bloc in blocs:
        if bloc[0] == "md":
            # st.markdown : Markdown GitHub (tableaux, listes, code) et formules LaTeX.
            st.markdown(bloc[1])
        elif bloc[0] == "titre":
            # Déballage du tuple ("titre", niveau, texte, ancre) ; « _ » ignore le premier élément.
            _, niveau, texte, ancre = bloc
            # Titre de niveau 1 identique au titre de l'application : déjà affiché en tête de page.
            if niveau == 1 and texte.strip() in TITRE_APPLI:
                continue
            # Niveaux 1 à 3 : composants Streamlit dont l'ancre est imposée (liens du sommaire) ; 4 à 6 : Markdown.
            if niveau == 1:
                st.title(texte, anchor=ancre)
            elif niveau == 2:
                st.header(texte, anchor=ancre)
            elif niveau == 3:
                st.subheader(texte, anchor=ancre)
            else:
                st.markdown(f"{'#' * niveau} {texte}")
        elif images:
            # st.image : figure à la largeur de la colonne, légende et texte alternatif (lecteurs d'écran).
            st.image(bloc[1], caption=bloc[2], width="stretch", alt=bloc[2])


def extrait_section(nom, numero):
    """Blocs d'une section numérotée (« 2.3 », « 5.4 »…) d'un document : du titre jusqu'au titre suivant de
    niveau égal ou supérieur. Renvoie (titre, blocs) ou (None, [])."""
    blocs = blocs_de(nom)
    # Position du titre dont le texte commence par le numéro suivi d'un espace.
    debut = next((i for i, b in enumerate(blocs) if b[0] == "titre" and b[2].startswith(numero + " ")), None)
    if debut is None:
        return None, []
    niveau = blocs[debut][1]
    # Fin : prochain titre de niveau inférieur ou égal (un niveau numérique plus petit est plus haut).
    fin = next((i for i in range(debut + 1, len(blocs)) if blocs[i][0] == "titre" and blocs[i][1] <= niveau), len(blocs))
    return blocs[debut][2], blocs[debut + 1:fin]


def page_documentation():
    """Onglet de documentation : choix de la page et sommaire dans le panneau latéral, contenu au centre."""
    with st.sidebar:
        st.header("📚 Documentation")
        # st.radio : choix exclusif de la page ; persist_state="page" garde le choix quand l'onglet est masqué.
        nom = st.radio("Page", list(DOCUMENTS), key="document", persist_state="page")
        chemin = DOCUMENTS[nom]
        if chemin is not None:
            # Version PDF (produite par docs/build_pdf.py) ; on_click="ignore" : pas de nouvelle exécution du script.
            pdf = chemin.with_suffix(".pdf")
            if pdf.exists():
                st.download_button("⬇️ Version PDF", data=pdf.read_bytes(), file_name=pdf.name, mime="application/pdf",
                                   on_click="ignore", width="stretch", key="pdf")
            # Sommaire : liens vers les titres de niveau 2 (ancres posées par afficher_blocs), hors « Sommaire ».
            titres = [b for b in blocs_de(nom) if b[0] == "titre" and b[1] == 2 and b[2] != "Sommaire"]
            if titres:
                st.markdown("**Sommaire**")
                st.markdown("\n".join(f"- [{b[2]}](#{b[3]})" for b in titres))
    # Contenu de la page choisie.
    if nom == "🏠 Accueil":
        accueil.page_accueil()
    elif nom == "📖 Glossaire":
        st.header("📖 Glossaire")
        st.markdown(glossaire.texte_glossaire())
    else:
        afficher_blocs(blocs_de(nom))
