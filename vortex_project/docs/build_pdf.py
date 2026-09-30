"""
build_pdf.py — Conversion des documents Markdown du dossier docs/ en PDF.

Chaîne entièrement locale : aucune bibliothèque à installer, aucun accès réseau.
1. Markdown → HTML : sous-ensemble utilisé par les rapports (titres, paragraphes, listes imbriquées,
   tableaux, citations, images, liens, code, gras / italique, typographie française) ;
2. formules LaTeX ($…$ en ligne, $$…$$ centrées) → MathML, que Chrome et Edge affichent nativement ;
3. impression en PDF par le navigateur en mode sans interface (--headless --print-to-pdf).

Usage (depuis vortex_project) :
    .venv\\Scripts\\python docs\\build_pdf.py              # tous les .md de docs/
    .venv\\Scripts\\python docs\\build_pdf.py docs\\x.md    # fichiers choisis
Le navigateur est cherché aux emplacements usuels ; la variable d'environnement CHROME_PATH permet d'en
imposer un autre.
"""

# Annotations non évaluées à l'exécution.
from __future__ import annotations

# html.escape : échappement des caractères spéciaux (&, <, >) dans le HTML produit.
import html
# os : variables d'environnement (CHROME_PATH, LOCALAPPDATA).
import os
# re : expressions régulières (analyse du Markdown et du LaTeX).
import re
# shutil : recherche d'exécutables dans le PATH et suppression du profil temporaire du navigateur.
import shutil
# subprocess : lancement du navigateur en mode sans interface.
import subprocess
# sys : arguments de la ligne de commande et code de sortie.
import sys
# tempfile : dossier temporaire servant de profil au navigateur (indépendant de la session ouverte).
import tempfile
# pathlib.Path : manipulation des chemins de fichiers.
from pathlib import Path

# Dossier contenant ce script (= dossier docs/, où se trouvent les .md et le sous-dossier figures/).
DOCS_DIR = Path(__file__).resolve().parent

# =============================================================================================
# 1. LaTeX → MathML
# =============================================================================================
# Lettres grecques minuscules (italiques en mathématiques, comme en LaTeX).
GREEK_LOWER = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "varsigma": "ς", "tau": "τ",
    "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
}
# Lettres grecques majuscules (droites en LaTeX).
GREEK_UPPER = {
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
# Symboles rendus comme opérateurs (<mo>) : relations, flèches, délimiteurs, opérateurs binaires.
SYMBOLS_MO = {
    "times": "×", "cdot": "⋅", "approx": "≈", "simeq": "≃", "sim": "∼", "propto": "∝", "equiv": "≡",
    "neq": "≠", "ne": "≠", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ll": "≪", "gg": "≫",
    "lesssim": "≲", "gtrsim": "≳", "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "Longrightarrow": "⟹", "Leftrightarrow": "⇔", "mapsto": "↦", "pm": "±", "mp": "∓", "langle": "⟨",
    "rangle": "⟩", "cdots": "⋯", "ldots": "…", "dots": "…", "in": "∈", "notin": "∉", "subset": "⊂",
    "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃", "perp": "⊥", "parallel": "∥", "circ": "∘",
    "star": "⋆", "bullet": "∙", "oplus": "⊕", "otimes": "⊗", "wedge": "∧", "vee": "∨", "colon": ":",
    "mid": "∣", "setminus": "∖", "ast": "∗", "lvert": "|", "rvert": "|", "vert": "|", "lVert": "‖",
    "rVert": "‖", "Vert": "‖", "lbrace": "{", "rbrace": "}", "|": "‖", "{": "{", "}": "}", "%": "%",
    "#": "#", "&": "&", "_": "_",
}
# Symboles rendus comme identifiants (<mi>) : symboles « ordinaires » de LaTeX.
SYMBOLS_MI = {"partial": "∂", "nabla": "∇", "infty": "∞", "ell": "ℓ", "hbar": "ℏ", "emptyset": "∅"}
# Grands opérateurs à limites dessous/dessus en mode centré (somme, produit).
LARGE_OPS = {"sum": "∑", "prod": "∏", "bigcup": "⋃", "bigcap": "⋂"}
# Intégrales (bornes en indice et exposant).
INTEGRALS = {"int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮"}
# Fonctions usuelles (écrites droites, suivies d'une application de fonction).
FUNCTIONS = {"ln", "log", "exp", "sin", "cos", "tan", "cot", "sinh", "cosh", "tanh", "arcsin", "arccos",
             "arctan", "det", "arg", "dim", "ker", "deg", "erf", "sgn"}
# Fonctions à limites (lim, max, min...) : indice placé dessous en mode centré.
LIMIT_FUNCTIONS = {"lim", "max", "min", "sup", "inf", "limsup", "liminf", "argmax", "argmin"}
# Accents : caractère placé au-dessus de la base (<mover accent="true">).
ACCENTS = {"hat": "ˆ", "widehat": "ˆ", "tilde": "˜", "widetilde": "˜", "dot": "˙", "ddot": "¨",
           "bar": "¯", "overline": "‾", "vec": "→", "check": "ˇ", "breve": "˘", "acute": "´"}
# Espacements explicites (largeur en em) : \, \: \; \  \quad \qquad.
SPACES = {",": 0.1667, ":": 0.2222, ">": 0.2222, ";": 0.2778, " ": 0.25, "quad": 1.0, "qquad": 2.0,
          "enspace": 0.5, "thinspace": 0.1667}
# Tailles imposées des délimiteurs \big, \Big, \bigg, \Bigg (hauteur en em).
BIG_SIZES = {"big": 1.2, "Big": 1.6, "bigg": 2.1, "Bigg": 2.6}

# Délimiteurs ordinaires : jamais étirés (en LaTeX, seuls \left, \right et \big changent leur taille).
FENCES = {"(", ")", "[", "]", "{", "}", "|", "‖", "⟨", "⟩"}
# Délimiteurs et barre oblique « ordinaires » en TeX : aucun espace autour (|u|, ‖S‖, a/b).
TIGHT = {"|", "‖", "/"}
# Unités après lesquelles un « − » est unaire (début de formule, relation, ouverture, séparateur).
PREFIX_CONTEXT = {None, "=", "(", "[", "{", ",", ";", ":", "<", ">", "&", "^", "_", "\\approx", "\\simeq",
                  "\\sim", "\\equiv", "\\le", "\\leq", "\\ge", "\\geq", "\\ll", "\\gg", "\\propto", "\\to",
                  "\\left", "\\neq", "\\lesssim", "\\quad", "\\qquad", "\\,", "\\;", "\\big", "\\Big", "\\times",
                  "\\cdot", "\\Longrightarrow", "\\langle", "\\mathrel", "\\op"}


def _mo(ch, attrs=""):
    """Opérateur <mo> : délimiteurs ordinaires non étirables, barres et « / » sans espacement."""
    if ch in FENCES:
        attrs += ' stretchy="false"'
    if ch in TIGHT:
        attrs += ' lspace="0em" rspace="0em"'
    if ch == ":":
        # Deux-points de TeX (relation) : même espace des deux côtés (S : S, double contraction).
        attrs += ' lspace="0.2778em" rspace="0.2778em"'
    return f"<mo{attrs}>{_esc(ch)}</mo>"


# Découpage d'une formule en unités lexicales, dans l'ordre de priorité :
# commande à lettres (\frac), commande à un caractère (\, \{ \\), nombre (décimale française « 0{,}5 »
# ou point décimal), blanc, puis tout autre caractère isolé.
TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|\d+(?:\{,\}\d+|\.\d+)?|\s+|.", re.S)


def _alphabet(ch, latin_upper, latin_lower, digit=None, greek_upper=None, greek_lower=None, exceptions=None):
    """Transpose un caractère dans un alphabet mathématique Unicode (gras, gras italique, ajouré, ronde)."""
    # Exceptions : lettres déjà codées ailleurs dans Unicode (ex. ℝ, ℬ).
    if exceptions and ch in exceptions:
        return exceptions[ch]
    # ord / chr : code Unicode d'un caractère et caractère d'un code ; décalage depuis le début de l'alphabet.
    if "A" <= ch <= "Z":
        return chr(latin_upper + ord(ch) - ord("A"))
    if "a" <= ch <= "z":
        return chr(latin_lower + ord(ch) - ord("a"))
    if digit is not None and "0" <= ch <= "9":
        return chr(digit + ord(ch) - ord("0"))
    # Grec majuscule (U+0391 à U+03A9) et minuscule (U+03B1 à U+03C9), si l'alphabet les contient.
    if greek_upper is not None and "\u0391" <= ch <= "\u03a9":
        return chr(greek_upper + ord(ch) - 0x391)
    if greek_lower is not None and "\u03b1" <= ch <= "\u03c9":
        return chr(greek_lower + ord(ch) - 0x3B1)
    # Variantes grecques en gras italique (ϵ, ϑ, ϰ, ϕ, ϱ, ϖ), placées après ω dans Unicode.
    if greek_lower == 0x1D736:
        variants = {"ϵ": 0x1D750, "ϑ": 0x1D751, "ϰ": 0x1D752, "ϕ": 0x1D753, "ϱ": 0x1D754, "ϖ": 0x1D755}
        if ch in variants:
            return chr(variants[ch])
    # Caractère non transposable : inchangé.
    return ch


def bold(ch):
    # Gras droit (\mathbf) : A → U+1D400, a → U+1D41A, 0 → U+1D7CE, grec → U+1D6A8 / U+1D6C2.
    return _alphabet(ch, 0x1D400, 0x1D41A, 0x1D7CE, 0x1D6A8, 0x1D6C2)


def bold_italic(ch):
    # Gras italique (\boldsymbol) : A → U+1D468, a → U+1D482, grec minuscule → U+1D736 ; les majuscules grecques,
    # droites en LaTeX, restent droites (gras droit), de même que les chiffres.
    if "Α" <= ch <= "Ω":
        return bold(ch)
    return _alphabet(ch, 0x1D468, 0x1D482, 0x1D7CE, 0x1D71C, 0x1D736)


def double_struck(ch):
    # Ajouré (\mathbb) : exceptions ℂ ℍ ℕ ℙ ℚ ℝ ℤ, puis A → U+1D538, a → U+1D552, 0 → U+1D7D8 (𝟙 pour 1).
    return _alphabet(ch, 0x1D538, 0x1D552, 0x1D7D8,
                     exceptions={"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"})


def script(ch):
    # Ronde (\mathcal) : exceptions ℬ ℰ ℱ ℋ ℐ ℒ ℳ ℛ ℯ ℊ ℴ, puis A → U+1D49C, a → U+1D4B6.
    return _alphabet(ch, 0x1D49C, 0x1D4B6,
                     exceptions={"B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ", "L": "ℒ", "M": "ℳ",
                                 "R": "ℛ", "e": "ℯ", "g": "ℊ", "o": "ℴ"})


def _esc(text):
    # Échappement HTML du contenu textuel d'un élément MathML (quote=False : guillemets conservés).
    return html.escape(text, quote=False)


class LatexToMathML:
    """Analyseur récursif d'une formule LaTeX (sous-ensemble des rapports) produisant du MathML.

    Chaque méthode ``_parse_*`` consomme des unités lexicales et renvoie une chaîne MathML. Les commandes
    inconnues sont signalées dans ``unknown`` et rendues en rouge (<merror>) pour être repérées.
    """

    def __init__(self, tex, unknown):
        # Pré-traitement : « \mathrel{+}= » (opérateur composé comme « += ») → opérateur unique \op{+=}.
        tex = re.sub(r"\\mathrel\{([^{}]*)\}=", r"\\op{\1=}", tex)
        # Liste des unités lexicales (re.findall renvoie toutes les correspondances de TOKEN_RE).
        self.toks = TOKEN_RE.findall(tex)
        # Position courante dans la liste.
        self.i = 0
        # Ensemble partagé des commandes inconnues rencontrées (pour le rapport final).
        self.unknown = unknown

    # ------------------------------------------------------------------ accès aux unités lexicales
    def _peek(self, skip_space=True):
        """Unité lexicale courante (None en fin de formule), en sautant les blancs par défaut."""
        # Les blancs sont sans effet en mode mathématique : on les ignore.
        while skip_space and self.i < len(self.toks) and self.toks[self.i].isspace():
            self.i += 1
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self):
        """Consomme et renvoie l'unité lexicale courante (blancs sautés)."""
        tok = self._peek()
        self.i += 1
        return tok

    def _raw_group(self):
        """Texte brut d'un argument : contenu d'une paire d'accolades (imbrications comprises) ou unité isolée."""
        tok = self._peek()
        if tok != "{":
            # Argument sans accolades : une seule unité lexicale.
            return self._next() or ""
        # Consommation de « { » puis accumulation jusqu'à l'accolade fermante de même niveau.
        self.i += 1
        depth, out = 1, []
        while self.i < len(self.toks):
            t = self.toks[self.i]
            self.i += 1
            # Suivi de la profondeur d'imbrication des accolades.
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(t)
        return "".join(out)

    # ------------------------------------------------------------------ analyse
    def parse(self):
        """Formule complète → contenu MathML (suite d'éléments)."""
        return self._parse_seq(stop=())

    def _parse_seq(self, stop):
        """Suite d'atomes jusqu'à une unité d'arrêt (non consommée) ; gère le commutateur \\displaystyle."""
        items = []
        while True:
            tok = self._peek()
            # Fin de formule ou unité d'arrêt (« } », « & », « \\ », « \right », « \end »).
            if tok is None or tok in stop:
                break
            # Accolade fermante orpheline : ignorée pour ne pas boucler indéfiniment.
            if tok == "}":
                self.i += 1
                continue
            if tok == "\\displaystyle":
                # \displaystyle s'applique à toute la suite du groupe courant (style « centré »).
                self.i += 1
                rest = self._parse_seq(stop)
                items.append(f'<mstyle displaystyle="true">{rest}</mstyle>')
                break
            if tok == "\\textstyle":
                # \textstyle : style « en ligne » pour la suite du groupe.
                self.i += 1
                rest = self._parse_seq(stop)
                items.append(f'<mstyle displaystyle="false">{rest}</mstyle>')
                break
            items.append(self._parse_atom())
        return "".join(items)

    def _parse_atom(self):
        """Un atome suivi de ses éventuels indices, exposants et primes."""
        # Unité précédant l'atome (pour l'espacement des fonctions comme ln, cos).
        self._peek()
        prev = self._prev_token_before(self.i)
        base, kind = self._parse_primary()
        sub = sup = None
        primes = 0
        while True:
            tok = self._peek()
            if tok == "_":
                # Indice : argument suivant (groupe entre accolades ou unité isolée).
                self.i += 1
                sub = self._parse_argument()
            elif tok == "^":
                # Exposant.
                self.i += 1
                sup = self._parse_argument()
            elif tok == "'":
                # Prime(s) : x' → x′ (placé en exposant).
                self.i += 1
                primes += 1
            else:
                break
        if primes:
            # Prime(s) accolée(s) à la base : le glyphe ′ est déjà dessiné en position haute (le placer en exposant
            # le surélèverait) ; un indice ou un exposant explicite s'applique ensuite à l'ensemble x′.
            base = f'<mrow>{base}<mo lspace="0em" rspace="0em">{"′" * primes}</mo></mrow>'
        # Opérateurs « à limites » (somme, lim, max...) : limites dessous / dessus (repassées en indices en ligne
        # grâce à movablelimits).
        if kind in ("limits", "func_limits") and (sub or sup):
            if sub and sup:
                out = f"<munderover>{base}{sub}{sup}</munderover>"
            elif sub:
                out = f"<munder>{base}{sub}</munder>"
            else:
                out = f"<mover>{base}{sup}</mover>"
        elif sub and sup:
            # Indice et exposant simultanés.
            out = f"<msubsup>{base}{sub}{sup}</msubsup>"
        elif sub:
            out = f"<msub>{base}{sub}</msub>"
        elif sup:
            out = f"<msup>{base}{sup}</msup>"
        else:
            out = base
        # Fonctions (ln, cos...) : espace fine avant (sauf après une relation, une ouverture ou en début) et après
        # (sauf devant une parenthèse), comme l'espacement « \mathop » de TeX ; U+2061 = application de fonction.
        if kind == "func":
            thin = '<mspace width="0.1667em"/>'
            before = "" if prev in PREFIX_CONTEXT or prev in ("-", "+") else thin
            nxt = self._peek()
            # Pas d'espace après si une ouverture suit, ou une autre fonction (qui ajoute déjà la sienne avant).
            after = "" if (nxt in (None, "(", "[", "\\left", "|", "\\big", "\\Big", "}", "&")
                           or (nxt.startswith("\\") and nxt[1:] in FUNCTIONS)) else thin
            out = f"{before}{out}<mo>&#x2061;</mo>{after}"
        return out

    def _prev_token_before(self, k):
        """Unité lexicale non blanche située avant la position k (None en début de formule)."""
        k -= 1
        while k >= 0 and self.toks[k].isspace():
            k -= 1
        return self.toks[k] if k >= 0 else None

    def _parse_argument(self):
        """Argument d'une commande ou d'un indice : groupe {…} ou unité isolée (un seul chiffre, comme TeX)."""
        tok = self._peek()
        if tok is None:
            return "<mrow></mrow>"
        if tok == "{":
            # Groupe entre accolades → <mrow>.
            self.i += 1
            content = self._parse_seq(stop=("}",))
            # Consommation de l'accolade fermante.
            if self._peek() == "}":
                self.i += 1
            return f"<mrow>{content}</mrow>"
        if tok[0].isdigit() and len(tok) > 1 and tok.isdigit():
            # « \frac12 » : TeX ne prend qu'un chiffre ; le reste est remis dans le flux.
            self.toks[self.i] = tok[1:]
            return f"<mn>{tok[0]}</mn>"
        # Autre unité isolée (lettre, commande) : atome primaire sans indices.
        base, kind = self._parse_primary()
        return base

    def _parse_primary(self):
        """Atome de base. Renvoie (MathML, genre) avec genre ∈ {ord, limits, int, func, func_limits}."""
        tok = self._next()
        if tok is None:
            return "", "ord"
        # Groupe entre accolades.
        if tok == "{":
            content = self._parse_seq(stop=("}",))
            if self._peek() == "}":
                self.i += 1
            return f"<mrow>{content}</mrow>", "ord"
        # Nombre (décimale française « {,} » restituée en virgule simple).
        if tok[0].isdigit():
            return f"<mn>{tok.replace('{,}', ',')}</mn>", "ord"
        # Lettre latine : identifiant (italique automatique pour un caractère seul).
        if tok.isascii() and tok.isalpha():
            return f"<mi>{tok}</mi>", "ord"
        # Commandes.
        if tok.startswith("\\"):
            return self._parse_command(tok[1:])
        # Signe moins typographique : unaire (form="prefix", sans espaces) après une relation, une ouverture
        # ou en début de formule, binaire sinon.
        if tok in "-+":
            sign = "−" if tok == "-" else "+"
            if self._prev_token() in PREFIX_CONTEXT:
                return f'<mo form="prefix">{sign}</mo>', "ord"
            return f"<mo>{sign}</mo>", "ord"
        # Astérisque mathématique (F*, α*).
        if tok == "*":
            return "<mo>∗</mo>", "ord"
        # Tilde LaTeX : espace insécable.
        if tok == "~":
            return '<mspace width="0.25em"/>', "ord"
        # Autre caractère : lettre (ex. accentuée) → identifiant, sinon opérateur (délimiteurs non étirables).
        if tok.isalpha():
            return f"<mi>{_esc(tok)}</mi>", "ord"
        return _mo(tok), "ord"

    def _prev_token(self):
        """Unité lexicale non blanche qui précède l'unité venant d'être consommée (None en début de formule)."""
        # self.i pointe après l'unité courante : on remonte à partir de l'unité précédente.
        k = self.i - 2
        while k >= 0 and self.toks[k].isspace():
            k -= 1
        return self.toks[k] if k >= 0 else None

    def _parse_command(self, name):
        """Commande LaTeX (nom sans la barre oblique inverse)."""
        # --- lettres grecques
        if name in GREEK_LOWER:
            return f"<mi>{GREEK_LOWER[name]}</mi>", "ord"
        if name in GREEK_UPPER:
            # Majuscules grecques droites (mathvariant="normal" évite l'italique automatique).
            return f'<mi mathvariant="normal">{GREEK_UPPER[name]}</mi>', "ord"
        # --- symboles
        if name in SYMBOLS_MI:
            return f'<mi mathvariant="normal">{SYMBOLS_MI[name]}</mi>', "ord"
        if name in SYMBOLS_MO:
            return _mo(SYMBOLS_MO[name]), "ord"
        # --- espacements
        if name in SPACES:
            return f'<mspace width="{SPACES[name]}em"/>', "ord"
        if name == "!":
            # Espace négative de TeX : ignorée (effet visuel négligeable).
            return "", "ord"
        # --- grands opérateurs, intégrales, fonctions
        if name in LARGE_OPS:
            # movablelimits : limites dessous/dessus en mode centré, en indices dans le texte.
            return f'<mo movablelimits="true" largeop="true">{LARGE_OPS[name]}</mo>', "limits"
        if name in INTEGRALS:
            return f'<mo largeop="true">{INTEGRALS[name]}</mo>', "int"
        if name in FUNCTIONS:
            return f"<mi>{name}</mi>", "func"
        if name in LIMIT_FUNCTIONS:
            return f'<mo movablelimits="true" form="prefix">{name}</mo>', "func_limits"
        # --- fractions et racines
        if name in ("frac", "dfrac", "tfrac"):
            num = self._parse_argument()
            den = self._parse_argument()
            frac = f"<mfrac>{num}{den}</mfrac>"
            # \dfrac : fraction de taille « centrée » ; \tfrac : taille « en ligne ».
            if name == "dfrac":
                frac = f'<mstyle displaystyle="true">{frac}</mstyle>'
            elif name == "tfrac":
                frac = f'<mstyle displaystyle="false">{frac}</mstyle>'
            return frac, "ord"
        if name == "sqrt":
            # Indice facultatif entre crochets : \sqrt[n]{x} → racine n-ième.
            if self._peek() == "[":
                self.i += 1
                index = self._parse_seq(stop=("]",))
                self.i += 1
                radicand = self._parse_argument()
                return f"<mroot>{radicand}<mrow>{index}</mrow></mroot>", "ord"
            return f"<msqrt>{self._parse_argument()}</msqrt>", "ord"
        # --- délimiteurs extensibles
        if name == "left":
            return self._parse_left_right(), "ord"
        if name in BIG_SIZES or name.rstrip("lr") in BIG_SIZES:
            # \big( ou \bigl( : délimiteur de hauteur imposée.
            size = BIG_SIZES[name.rstrip("lr")]
            delim = self._delimiter(self._next())
            return (f'<mo stretchy="true" symmetric="true" minsize="{size}em" maxsize="{size}em">'
                    f"{delim}</mo>"), "ord"
        # --- styles de caractères
        if name in ("mathbf", "boldsymbol", "mathbb", "mathcal", "mathrm", "operatorname", "mathit"):
            return self._parse_font(name)
        if name == "text" or name == "mbox" or name == "textrm":
            # Texte : blancs de bord remplacés par des espaces insécables (sinon supprimés par le rendu).
            raw = self._raw_group()
            raw = re.sub(r"^ | $", "\u00a0", raw)
            return f"<mtext>{_esc(raw)}</mtext>", "ord"
        if name == "op":
            # Opérateur composé produit par le pré-traitement de \mathrel{…}=.
            return f"<mo>{_esc(self._raw_group())}</mo>", "ord"
        if name == "mathrel":
            return f"<mo>{_esc(self._raw_group())}</mo>", "ord"
        # --- accents
        if name in ACCENTS:
            base = self._parse_argument()
            stretch = ' stretchy="true"' if name in ("overline", "widehat", "widetilde") else ' stretchy="false"'
            return f'<mover accent="true">{base}<mo{stretch}>{ACCENTS[name]}</mo></mover>', "ord"
        # --- encadré (cas non traité au niveau du bloc : bordure CSS)
        if name == "boxed":
            content = self._parse_argument()
            return (f'<mrow style="border:0.8pt solid currentColor;padding:0.25em 0.4em">{content}</mrow>',
                    "ord")
        # --- environnements (cases)
        if name == "begin":
            env = self._raw_group()
            return self._parse_environment(env), "ord"
        # --- commandes sans effet visuel ici
        if name in ("limits", "nolimits", "displaystyle", "textstyle", "right", "end"):
            return "", "ord"
        # --- commande inconnue : signalée et rendue en rouge
        self.unknown.add("\\" + name)
        return f"<merror><mtext>\\{_esc(name)}</mtext></merror>", "ord"

    def _delimiter(self, tok):
        """Caractère d'un délimiteur (« ( », « \\{ », « \\| », « \\langle », « . » = vide)."""
        if tok is None or tok == ".":
            return ""
        if tok.startswith("\\"):
            name = tok[1:]
            return SYMBOLS_MO.get(name, name)
        return tok

    def _parse_left_right(self):
        """\\left X … \\right Y → <mrow> encadré de deux délimiteurs extensibles."""
        left = self._delimiter(self._next())
        content = self._parse_seq(stop=("\\right",))
        # Consommation de \right puis de son délimiteur.
        if self._peek() == "\\right":
            self.i += 1
        right = self._delimiter(self._next())
        # Délimiteur vide (« . ») : aucun <mo>.
        lo = f'<mo fence="true" stretchy="true" symmetric="true">{_esc(left)}</mo>' if left else ""
        ro = f'<mo fence="true" stretchy="true" symmetric="true">{_esc(right)}</mo>' if right else ""
        return f"<mrow>{lo}{content}{ro}</mrow>"

    def _parse_font(self, name):
        """Alphabets mathématiques : \\mathbf, \\boldsymbol, \\mathbb, \\mathcal, \\mathrm, \\operatorname."""
        # Position de repli si l'argument n'est pas une simple suite de lettres.
        start = self.i
        raw = self._raw_group().strip()
        # Argument réduit à une lettre grecque (ex. \boldsymbol{\omega}) : caractère grec correspondant.
        greek = GREEK_LOWER.get(raw[1:]) or GREEK_UPPER.get(raw[1:]) if raw.startswith("\\") else None
        simple = greek or (raw if re.fullmatch(r"[A-Za-z0-9]+", raw) else None)
        if simple is None:
            # Contenu complexe : analyse normale de l'argument (style non appliqué).
            self.i = start
            return self._parse_argument(), "ord"
        if name == "mathbf":
            return f"<mi>{''.join(bold(c) for c in simple)}</mi>", "ord"
        if name == "boldsymbol":
            return f"<mi>{''.join(bold_italic(c) for c in simple)}</mi>", "ord"
        if name == "mathbb":
            return f"<mi>{''.join(double_struck(c) for c in simple)}</mi>", "ord"
        if name == "mathcal":
            return f"<mi>{''.join(script(c) for c in simple)}</mi>", "ord"
        if name == "mathit":
            return f"<mi>{simple}</mi>", "ord"
        if name == "operatorname":
            return f"<mi>{simple}</mi>", "func"
        # \mathrm : caractères droits (mathvariant="normal" pour une lettre seule).
        return f'<mi mathvariant="normal">{simple}</mi>', "ord"

    def _parse_environment(self, env):
        """Environnement \\begin{cases} … \\end{cases} → accolade extensible + tableau aligné à gauche."""
        rows, cells, current = [], [], []
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok == "\\end":
                # Fin de l'environnement : consommation de \end{nom}.
                self.i += 1
                self._raw_group()
                break
            if tok == "&":
                # Séparateur de colonnes.
                self.i += 1
                cells.append("".join(current))
                current = []
                continue
            if tok == "\\\\":
                # Séparateur de lignes, suivi d'un espacement facultatif [3mm] (ignoré).
                self.i += 1
                cells.append("".join(current))
                rows.append(cells)
                cells, current = [], []
                if self._peek() == "[":
                    while self._next() not in ("]", None):
                        pass
                continue
            current.append(self._parse_atom())
        # Dernière ligne (non terminée par \\).
        if current or cells:
            cells.append("".join(current))
            rows.append(cells)
        # Tableau MathML : une ligne <mtr> par ligne, cellules <mtd> alignées à gauche (CSS text-align).
        body = "".join("<mtr>" + "".join(f'<mtd style="text-align:left;padding:0.15em 0.8em 0.15em 0">{c}</mtd>'
                                         for c in r) + "</mtr>" for r in rows)
        table = f"<mtable>{body}</mtable>"
        if env.startswith("cases"):
            return f'<mrow><mo fence="true" stretchy="true" symmetric="true">{{</mo>{table}</mrow>'
        # Autres environnements (matrix, aligned...) : tableau simple.
        return table


def tex_to_mathml(tex, display, unknown):
    """Formule LaTeX → élément <math> complet (display=True : formule centrée)."""
    body = LatexToMathML(tex.strip(), unknown).parse()
    attr = ' display="block"' if display else ""
    return f"<math{attr}>{body}</math>"


# =============================================================================================
# 2. Markdown → HTML
# =============================================================================================
# Élément de liste : indentation, puce (* + -) ou numéro (1. ou 1)), puis texte.
LIST_RE = re.compile(r"^(\s*)([*+-]|\d+[.)])\s+(.*)$")
# Ligne de séparation d'un tableau (| --- | :---: |).
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
# Formule en ligne $…$ (ni « $$ », ni dollar échappé).
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)")
# Image seule sur sa ligne.
IMAGE_LINE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)\)$")


def slugify(text):
    """Identifiant d'ancre à la manière de GitHub (liens « #… » du sommaire)."""
    # Minuscules, suppression de toute ponctuation sauf tirets et espaces (\w reconnaît les lettres accentuées),
    # puis espaces → tirets.
    s = text.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def french_spaces(text):
    """Espaces insécables de la typographie française (avant : ; ! ? » %, après «)."""
    # Espace fine insécable (U+202F) avant ; ! ? et %, insécable normale (U+00A0) avant : et autour des guillemets.
    text = re.sub(r" ([;!?])", "\u202f\\1", text)
    text = re.sub(r" (:)(?=\s|$|<)", "\u00a0\\1", text)
    text = text.replace("« ", "«\u00a0").replace(" »", "\u00a0»")
    return re.sub(r"(\d) %", "\\1\u202f%", text)


class MarkdownToHTML:
    """Convertisseur Markdown → HTML (sous-ensemble des rapports) avec formules en MathML."""

    def __init__(self):
        # Identifiants d'ancres déjà attribués (doublons numérotés comme sur GitHub).
        self.slugs = {}
        # Commandes LaTeX inconnues rencontrées.
        self.unknown = set()
        # Titre du document (premier titre de niveau 1).
        self.title = None

    # ------------------------------------------------------------------ éléments en ligne
    def inline(self, text, in_table=False):
        """Mise en forme d'un fragment de texte : code, formules, images, liens, gras, italique.

        ``in_table`` : cellule de tableau ; les identifiants de code y deviennent sécables après « _ », « . »,
        « / » et « \\ » (balise <wbr>), pour qu'une colonne étroite ne fasse pas déborder le tableau.
        """
        stash = []

        def keep(fragment):
            # Mise de côté d'un fragment déjà converti ; marqueur \x00n\x00 à sa place.
            stash.append(fragment)
            return f"\x00{len(stash) - 1}\x00"
        # Lien [`rapport.md`](rapport.md) vers un autre document converti : le texte affiché devient rapport.pdf,
        # comme la cible (voir href).
        text = re.sub(r"\[`([\w\-]+)\.md`\]\(([\w\-]+)\.md(#[^)]*)?\)",
                      lambda m: (f"[`{m.group(1)}.pdf`]({m.group(2)}.md{m.group(3) or ''})"
                                 if (DOCS_DIR / f"{m.group(2)}.md").exists() else m.group(0)), text)
        def code_span(m):
            # Code en ligne échappé ; dans un tableau, point de coupure facultatif (<wbr>) après _ . / \.
            code = html.escape(m.group(1))
            if in_table:
                code = re.sub(r"([_./\\])", r"\1<wbr>", code)
            return keep(f"<code>{code}</code>")
        # Code en ligne `…` puis formules $…$, protégés avant toute autre transformation.
        text = re.sub(r"`([^`]+)`", code_span, text)
        text = INLINE_MATH_RE.sub(lambda m: keep(tex_to_mathml(m.group(1), False, self.unknown)), text)
        # Échappement HTML du texte restant.
        text = html.escape(text, quote=False)
        # Images ![alt](src) et liens [texte](cible).
        text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", lambda m: self.image(m.group(2), m.group(1)), text)
        text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
                      lambda m: f'<a href="{self.href(m.group(2))}">{m.group(1)}</a>', text)
        # Gras **…** puis italique *…* (l'astérisque de « F* » n'ouvre pas d'italique : précédé d'une lettre).
        text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])", r"<em>\1</em>", text)
        # Espaces insécables français ; une parenthèse ou une ponctuation collée à une formule (ou à du code) est
        # regroupée avec elle dans un <span class="nw"> insécable, pour interdire une coupure de ligne entre les deux
        # (« ( » en fin de ligne) ; puis restitution des fragments mis de côté (éventuellement imbriqués).
        text = french_spaces(text)
        # Avant : « ( », « [ » ou « «  » ; après : ponctuation, éventuellement précédée de l'espace insécable ajoutée
        # par french_spaces (« $x$ ; » ne doit pas laisser « ; » seul en début de ligne).
        text = re.sub(r"([(\[]|« )?(\x00\d+\x00)([  ]?[)\],.;:!?»])?",
                      lambda m: (f'<span class="nw">{m.group(0)}</span>' if m.group(1) or m.group(3) else m.group(0)),
                      text)
        while "\x00" in text:
            text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
        return text

    @staticmethod
    def href(target):
        """Cible d'un lien : les autres rapports .md deviennent leur version .pdf."""
        # Lien vers un .md du dossier docs/ (éventuellement suivi d'une ancre) → même nom en .pdf.
        m = re.fullmatch(r"([\w\-]+)\.md(#.*)?", target)
        if m and (DOCS_DIR / f"{m.group(1)}.md").exists():
            return f"{m.group(1)}.pdf{m.group(2) or ''}"
        return target

    @staticmethod
    def image(src, alt):
        """Balise <img> (chemin relatif au dossier docs/, où le HTML temporaire est écrit)."""
        return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">'

    # ------------------------------------------------------------------ blocs
    def blocks(self, lines, tight=False):
        """Suite de lignes → HTML des blocs (récursif pour les listes et les citations)."""
        out, para = [], []
        i = 0

        def flush():
            # Paragraphe en cours : lignes jointes par des espaces.
            if para:
                out.append(("p", self.inline(" ".join(para))))
                para.clear()
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if not s:
                # Ligne vide : fin de paragraphe.
                flush()
                i += 1
                continue
            if s.startswith("```"):
                # Bloc de code délimité par ``` (nom du langage facultatif) : texte brut, indentation conservée,
                # aucune interprétation Markdown ni LaTeX à l'intérieur.
                flush()
                buf = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    buf.append(lines[i])
                    i += 1
                # Ligne fermante ``` consommée.
                i += 1
                # Bloc court (≤ 25 lignes) : jamais coupé entre deux pages ; bloc long : coupure autorisée.
                cls = "code short" if len(buf) <= 25 else "code"
                out.append(("pre", f'<pre class="{cls}"><code>{html.escape(chr(10).join(buf))}</code></pre>'))
                continue
            if s.startswith("$$"):
                # Formule centrée $$ … $$ (sur une ou plusieurs lignes).
                flush()
                if len(s) > 4 and s.endswith("$$"):
                    tex = s[2:-2]
                    i += 1
                else:
                    buf = [s[2:]]
                    i += 1
                    while i < len(lines) and not lines[i].strip().endswith("$$"):
                        buf.append(lines[i])
                        i += 1
                    if i < len(lines):
                        buf.append(lines[i].strip()[:-2])
                        i += 1
                    tex = "\n".join(buf)
                out.append(("eq", self.display_math(tex)))
                continue
            m = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", line)
            if m:
                # Titre : niveau = nombre de « # », ancre calculée comme GitHub sur le texte brut
                # (sans `code`, **gras**, *italique* ni syntaxe de lien, et sans les espaces insécables ajoutées).
                flush()
                level, content = len(m.group(1)), m.group(2)
                rendered = self.inline(content)
                plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", content)
                plain = re.sub(r"[`*]", "", plain)
                slug = slugify(plain)
                # Doublon : suffixe -1, -2...
                n = self.slugs.get(slug, 0)
                self.slugs[slug] = n + 1
                if n:
                    slug = f"{slug}-{n}"
                if level == 1 and self.title is None:
                    self.title = plain
                out.append(("h", f'<h{level} id="{html.escape(slug)}">{rendered}</h{level}>'))
                i += 1
                continue
            if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s):
                # Filet horizontal.
                flush()
                out.append(("hr", "<hr>"))
                i += 1
                continue
            if s.startswith(">"):
                # Citation : lignes consécutives commençant par « > », traitées récursivement.
                flush()
                buf = []
                while i < len(lines) and lines[i].strip().startswith(">"):
                    buf.append(re.sub(r"^\s*> ?", "", lines[i]))
                    i += 1
                out.append(("bq", f"<blockquote>{self.blocks(buf)}</blockquote>"))
                continue
            if s.startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
                # Tableau : en-tête, ligne de séparation (alignements), puis lignes de données.
                flush()
                html_table, i = self.table(lines, i)
                out.append(("table", html_table))
                continue
            lm = LIST_RE.match(line)
            if lm and (not para or lm.group(2) in "*+-"):
                # Liste (à puces ou numérotée) ; une liste à puces peut interrompre un paragraphe, pas une liste
                # numérotée (un « 2. » en début de ligne de texte n'est pas une liste).
                flush()
                html_list, i = self.list_block(lines, i)
                out.append(("list", html_list))
                continue
            m = IMAGE_LINE_RE.match(s)
            if m and not para:
                # Image seule sur sa ligne : figure centrée.
                out.append(("fig", f'<figure>{self.image(m.group(2), m.group(1))}</figure>'))
                i += 1
                continue
            # Ligne de texte ordinaire : ajoutée au paragraphe courant.
            para.append(s)
            i += 1
        flush()
        # Élément de liste « serré » réduit à un paragraphe : texte sans balise <p>.
        if tight and len(out) >= 1 and out[0][0] == "p":
            first = out[0][1]
            rest = "".join(frag if kind != "p" else f"<p>{frag}</p>" for kind, frag in out[1:])
            return first + rest
        return "".join(f"<p>{frag}</p>" if kind == "p" else frag for kind, frag in out)

    def display_math(self, tex):
        """Formule centrée ; un \\boxed{…} englobant toute la formule devient un cadre CSS."""
        tex = tex.strip()
        boxed = False
        if tex.startswith("\\boxed{"):
            # Recherche de l'accolade fermante de \boxed{ : si rien ne suit, tout est encadré.
            depth, end = 0, None
            for k, ch in enumerate(tex[6:], start=6):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
            if end is not None and not tex[end + 1:].strip():
                tex, boxed = tex[7:end], True
        cls = "eq boxed" if boxed else "eq"
        return f'<div class="{cls}">{tex_to_mathml(tex, True, self.unknown)}</div>'

    @staticmethod
    def _split_row(line):
        """Cellules d'une ligne de tableau (les « | » à l'intérieur des formules et du code sont protégés)."""
        segs = []

        def protect(m):
            segs.append(m.group(0))
            return f"\x01{len(segs) - 1}\x01"
        s = re.sub(r"`[^`]*`|(?<![\\$])\$(?!\$).+?(?<![\\$])\$", protect, line.strip())
        # Suppression des « | » de bord puis découpage.
        s = s[1:] if s.startswith("|") else s
        s = s[:-1] if s.endswith("|") else s
        return [re.sub(r"\x01(\d+)\x01", lambda m: segs[int(m.group(1))], c).strip() for c in s.split("|")]

    def table(self, lines, i):
        """Tableau Markdown → <table> (alignements lus sur la ligne de séparation)."""
        head = self._split_row(lines[i])
        # « :--- » gauche, « :---: » centre, « ---: » droite.
        aligns = []
        for c in self._split_row(lines[i + 1]):
            c = c.strip()
            aligns.append("center" if c.startswith(":") and c.endswith(":") else
                          "right" if c.endswith(":") else "left")
        i += 2
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append(self._split_row(lines[i]))
            i += 1

        def cell(tag, text, k):
            align = aligns[k] if k < len(aligns) else "left"
            return f'<{tag} style="text-align:{align}">{self.inline(text, in_table=True)}</{tag}>'
        thead = "<tr>" + "".join(cell("th", c, k) for k, c in enumerate(head)) + "</tr>"
        tbody = "".join("<tr>" + "".join(cell("td", c, k) for k, c in enumerate(r)) + "</tr>" for r in rows)
        return f'<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>', i

    def list_block(self, lines, i):
        """Liste : éléments de même indentation ; lignes indentées ou paragraphes suivants rattachés à l'élément."""
        m = LIST_RE.match(lines[i])
        indent = len(m.group(1))
        ordered = m.group(2)[0].isdigit()
        start = int(m.group(2)[:-1]) if ordered else None
        items = []

        def same_level(line):
            # Élément de liste de même indentation et de même type (puces ou numéros).
            mm = LIST_RE.match(line)
            return bool(mm) and len(mm.group(1)) == indent and mm.group(2)[0].isdigit() == ordered
        while i < len(lines):
            if not lines[i].strip():
                # Lignes vides entre deux éléments (liste « aérée ») : la liste continue si l'élément suivant
                # est du même niveau.
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and same_level(lines[j]):
                    i = j
                    continue
                break
            m = LIST_RE.match(lines[i])
            # Fin de liste : ligne qui n'est pas un élément du même niveau et du même type.
            if not same_level(lines[i]):
                break
            # Indentation du contenu (largeur de la puce + espace) ; au moins 2 espaces.
            content_indent = max(len(m.group(1)) + len(m.group(2)) + 1, indent + 2)
            item = [m.group(3)]
            i += 1
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    # Ligne vide : l'élément continue si la ligne non vide suivante est indentée.
                    j = i
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and len(lines[j]) - len(lines[j].lstrip()) >= content_indent:
                        item.extend([""] * (j - i))
                        i = j
                        continue
                    break
                ind = len(line) - len(line.lstrip())
                if ind >= content_indent:
                    # Ligne indentée : continuation (ou sous-liste), désindentée.
                    item.append(line[content_indent:])
                    i += 1
                    continue
                s = line.strip()
                if LIST_RE.match(line) or s.startswith(("$$", "#", "|", ">", "```")) or IMAGE_LINE_RE.match(s):
                    # Nouvel élément ou nouveau bloc : fin de l'élément courant.
                    break
                # Continuation « paresseuse » (ligne non indentée juste après le texte).
                item.append(s)
                i += 1
            items.append(f"<li>{self.blocks(item, tight=True)}</li>")
        tag = "ol" if ordered else "ul"
        start_attr = f' start="{start}"' if ordered and start != 1 else ""
        return f"<{tag}{start_attr}>{''.join(items)}</{tag}>", i

    def convert(self, markdown):
        """Document complet → corps HTML."""
        return self.blocks(markdown.splitlines())


# =============================================================================================
# 3. Page HTML et impression PDF
# =============================================================================================
# Feuille de style d'impression : A4, texte Cambria (assorti à la police mathématique Cambria Math),
# titres en Segoe UI, numéros de page en pied de page.
CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font: 8pt "Segoe UI", sans-serif; color: #6b7280; } }
html { font-family: Cambria, Georgia, serif; font-size: 10.5pt; line-height: 1.42; color: #1b1f24; }
body { margin: 0 auto; width: 178mm; }
h1 { font-family: "Segoe UI", sans-serif; font-weight: 600; font-size: 19pt; line-height: 1.2; color: #0d2a4a; margin: 0 0 8pt; }
h2 { font-family: "Segoe UI", sans-serif; font-weight: 600; font-size: 14pt; color: #0d2a4a; margin: 16pt 0 6pt;
     padding-bottom: 2pt; border-bottom: 0.8pt solid #c7d2de; break-after: avoid; }
h3 { font-family: "Segoe UI", sans-serif; font-weight: 600; font-size: 11.5pt; color: #173e66; margin: 12pt 0 4pt; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: left; orphans: 2; widows: 2; }
a { color: #1f5fa8; text-decoration: none; }
.nw { white-space: nowrap; }
strong { font-weight: 700; }
code { font-family: Consolas, "Courier New", monospace; font-size: 0.84em; background: #f1f3f5; padding: 0 2pt; border-radius: 2pt; }
p code, li code { white-space: nowrap; }
pre.code { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; line-height: 1.35; background: #f4f6f8;
           border: 0.6pt solid #dde3ea; border-left: 3pt solid #9fb3c8; padding: 5pt 8pt; margin: 5pt 0 8pt;
           white-space: pre-wrap; }
pre.code.short { break-inside: avoid; }
pre.code code { background: none; padding: 0; font-size: inherit; }
math { font-family: "Cambria Math", math; }
mfrac { padding-left: 0.12em; padding-right: 0.12em; }
div.eq { margin: 6pt 0 8pt; text-align: center; break-inside: avoid; }
div.eq.boxed { display: table; margin: 8pt auto 10pt; border: 0.8pt solid #3b4450; padding: 4pt 12pt; }
figure { margin: 6pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 118mm; }
img { max-width: 100%; }
table { border-collapse: collapse; margin: 6pt auto 10pt; font-size: 9.4pt; break-inside: avoid; }
th, td { border: 0.6pt solid #b7c1cc; padding: 2.5pt 6pt; vertical-align: middle; }
th { background: #edf1f5; font-weight: 600; }
blockquote { margin: 6pt 0 8pt; padding: 3pt 10pt; border-left: 3pt solid #9fb3c8; background: #f5f7fa; }
blockquote p { margin: 3pt 0; }
ul, ol { margin: 3pt 0 6pt; padding-left: 18pt; }
li { margin: 2pt 0; text-align: left; }
li > p { margin: 2pt 0 4pt; }
hr { border: none; border-top: 0.6pt solid #cfd6de; margin: 10pt 0; }
"""

# Script exécuté avant l'impression : une formule, un tableau ou une formule en ligne plus large que la
# colonne est réduit (taille de police) pour tenir dans la page au lieu d'être coupé.
FIT_SCRIPT = """
window.addEventListener('load', () => {
  const avail = document.body.clientWidth;
  // Largeur réelle du contenu d'un élément (réunion des boîtes de ses enfants) : une formule centrée
  // occupe toute la largeur de la colonne, seul son contenu renseigne sur un éventuel débordement.
  const contentWidth = el => { const r = document.createRange(); r.selectNodeContents(el);
                               return r.getBoundingClientRect().width; };
  const shrink = (el, w, lim) => { if (w > lim) el.style.fontSize = (97 * lim / w).toFixed(1) + '%'; };
  document.querySelectorAll('div.eq math').forEach(m => {
    const lim = m.parentElement.classList.contains('boxed') ? avail - 40 : avail;
    shrink(m, contentWidth(m), lim);
  });
  document.querySelectorAll('table').forEach(t => shrink(t, t.getBoundingClientRect().width, avail));
  document.querySelectorAll('p math, li math').forEach(m => shrink(m, m.getBoundingClientRect().width, avail));
});
"""


def build_html(md_path):
    """Fichier Markdown → (page HTML complète, commandes LaTeX inconnues)."""
    conv = MarkdownToHTML()
    body = conv.convert(md_path.read_text(encoding="utf-8"))
    title = html.escape(conv.title or md_path.stem)
    page = (f'<!DOCTYPE html>\n<html lang="fr"><head><meta charset="utf-8"><title>{title}</title>'
            f"<style>{CSS}</style></head><body>{body}<script>{FIT_SCRIPT}</script></body></html>")
    return page, conv.unknown


def find_browser():
    """Chemin de Chrome ou d'Edge (variable CHROME_PATH prioritaire), ou None."""
    candidates = [os.environ.get("CHROME_PATH", "")]
    # Emplacements d'installation usuels sous Windows (machine puis utilisateur), Chrome puis Edge.
    for base in (os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                 os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                 os.environ.get("LOCALAPPDATA", "")):
        candidates.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
        candidates.append(os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"))
    # Exécutables du PATH (Linux / macOS / installations personnalisées).
    for name in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        candidates.append(shutil.which(name) or "")
    # Premier candidat existant.
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def print_pdf(browser, html_path, pdf_path):
    """Impression d'une page HTML en PDF par le navigateur sans interface (profil temporaire isolé)."""
    # Profil temporaire : le navigateur ne se raccroche pas à une session déjà ouverte.
    profile = tempfile.mkdtemp(prefix="pdf_profile_")
    try:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               "--no-pdf-header-footer", "--print-to-pdf-no-header", f"--user-data-dir={profile}",
               f"--print-to-pdf={pdf_path}", "--virtual-time-budget=8000", html_path.resolve().as_uri()]
        # capture_output : messages du navigateur récupérés (affichés seulement en cas d'échec).
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if res.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(f"échec de l'impression ({res.returncode}) :\n{res.stderr[-2000:]}")
    finally:
        # Suppression du profil temporaire (ignore_errors : fichiers éventuellement encore verrouillés).
        shutil.rmtree(profile, ignore_errors=True)


def main(argv=None):
    # Fichiers à convertir : arguments, sinon tous les .md du dossier docs/.
    args = sys.argv[1:] if argv is None else argv
    keep_html = "--keep-html" in args
    files = [Path(a) for a in args if not a.startswith("--")] or sorted(DOCS_DIR.glob("*.md"))
    browser = find_browser()
    if browser is None:
        print("Chrome ou Edge introuvable : indiquez son chemin dans la variable CHROME_PATH.")
        return 1
    for md in files:
        md = md.resolve()
        page, unknown = build_html(md)
        # HTML temporaire écrit à côté du .md : les chemins relatifs des images et les ancres restent valides.
        tmp_html = md.with_name(f"_print_{md.stem}.html")
        tmp_html.write_text(page, encoding="utf-8")
        pdf = md.with_suffix(".pdf")
        try:
            print_pdf(browser, tmp_html, pdf)
        finally:
            if not keep_html:
                tmp_html.unlink(missing_ok=True)
        # Taille du PDF produit (en kio) et commandes LaTeX non reconnues éventuelles.
        print(f"{md.name} → {pdf.name} ({pdf.stat().st_size / 1024:.0f} kio)")
        if unknown:
            print("   commandes LaTeX non reconnues :", ", ".join(sorted(unknown)))
    return 0


if __name__ == "__main__":
    # sys.exit transmet le code de retour de main au système.
    sys.exit(main())
