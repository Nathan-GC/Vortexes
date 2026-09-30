"""
Tests du convertisseur Markdown / LaTeX → HTML + MathML (``docs/build_pdf.py``).

L'impression PDF elle-même (navigateur) n'est pas testée ici : seule la conversion est vérifiée.
"""

# importlib.util : chargement d'un module à partir de son chemin (docs/ n'est pas un paquet Python).
import importlib.util
# re : extraction des éléments <math> du HTML produit.
import re
# xml.etree.ElementTree : vérification que chaque formule MathML est un XML bien formé.
import xml.etree.ElementTree as ET
# pathlib.Path : chemins des documents.
from pathlib import Path

# pytest : paramétrage des tests sur les deux documents.
import pytest

# Dossier docs/ du projet (ce fichier est dans tests/).
DOCS = Path(__file__).resolve().parents[1] / "docs"
# Chargement du module docs/build_pdf.py sous le nom « build_pdf ».
_spec = importlib.util.spec_from_file_location("build_pdf", DOCS / "build_pdf.py")
build_pdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_pdf)


@pytest.mark.parametrize("name", ["theoretical_background.md", "vortex_engines_analysis.md"])
def test_documents_convert_to_wellformed_mathml(name):
    """Chaque rapport se convertit sans commande LaTeX inconnue et chaque formule est un XML valide."""
    page, unknown = build_pdf.build_html(DOCS / name)
    # Aucune commande non reconnue (elles apparaîtraient en rouge dans le PDF).
    assert not unknown
    # Toutes les formules <math>…</math> (re.S : le point reconnaît aussi les retours à la ligne).
    formulas = re.findall(r"<math.*?</math>", page, re.S)
    assert len(formulas) > 50
    for f in formulas:
        # ET.fromstring lève une exception si les balises ne sont pas correctement imbriquées.
        ET.fromstring(f)


def test_basic_latex_constructs():
    unknown = set()

    def conv(tex, display=False):
        # Conversion d'une formule isolée en élément <math>.
        return build_pdf.tex_to_mathml(tex, display, unknown)
    # Fraction, racine sans accolades, décimale française, gras, prime, indice/exposant.
    assert "<mfrac><mrow><mi>a</mi></mrow><mrow><mi>b</mi></mrow></mfrac>" in conv(r"\frac{a}{b}")
    assert "<msqrt><mn>2</mn></msqrt>" in conv(r"\sqrt 2")
    assert "<mn>0,558</mn>" in conv(r"0{,}558")
    assert "𝐮" in conv(r"\mathbf{u}") and "𝝎" in conv(r"\boldsymbol{\omega}")
    assert "′" in conv(r"x'") and "<msubsup>" in conv(r"\sigma_\theta^2")
    # « \frac12 » : un chiffre par argument, comme TeX.
    assert "<mfrac><mn>1</mn><mn>2</mn></mfrac>" in conv(r"\frac12")
    # Délimiteurs : étirables avec \left … \right, fixes sinon.
    assert 'stretchy="true"' in conv(r"\left( x \right)")
    assert 'stretchy="false"' in conv(r"(x)")
    # Moins unaire après « = » (form="prefix"), environnement cases (tableau), formule centrée.
    assert 'form="prefix"' in conv(r"a = -b")
    assert "<mtable>" in conv(r"\begin{cases} a & b \\ c & d \end{cases}")
    assert 'display="block"' in conv("x", display=True)
    assert not unknown


def test_fenced_code_blocks():
    """Bloc ``` : texte brut échappé (ni Markdown, ni LaTeX interprétés), y compris dans une liste."""
    conv = build_pdf.MarkdownToHTML()
    page = conv.convert("```bash\n.venv\\Scripts\\python main.py all  # x < y, $a$ **b**\n```")
    # Contenu échappé tel quel dans <pre><code>.
    assert '<pre class="code short"><code>.venv\\Scripts\\python main.py all  # x &lt; y, $a$ **b**</code></pre>' in page
    # Aucune formule ni gras produits à partir du contenu du bloc.
    assert "<math" not in page and "<strong>" not in page
    # Bloc indenté à l'intérieur d'un élément de liste.
    page = build_pdf.MarkdownToHTML().convert("1. Lancer :\n\n   ```bash\n   pytest\n   ```\n2. Suite")
    assert "<ol><li>" in page and "<pre" in page and "Suite" in page


def test_github_like_anchors():
    """Les ancres reproduisent celles de GitHub (liens du sommaire des rapports)."""
    assert (build_pdf.slugify("3. Formation des anneaux : le nombre de formation F ≈ 4")
            == "3-formation-des-anneaux--le-nombre-de-formation-f--4")
    assert build_pdf.slugify("5. Diagnostics et visualisations avancés") == "5-diagnostics-et-visualisations-avancés"
