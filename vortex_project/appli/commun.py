"""
appli/commun.py — Outils partagés par tous les modules de l'application web.

Constantes (titres, chemins), verrou de tracé matplotlib, conversion des figures en images, mise en forme des
durées et des instants, barre de progression reliée à la tâche de calcul, métriques, courbes interactives
(Altair) et widgets compatibles avec les scénarios guidés (valeur initiale posée dans l'état de session).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# io.BytesIO : fichier en mémoire (PNG, GIF) ; threading.Lock : verrou partagé entre fils d'exécution.
import io
import threading
# Path : chemins de fichiers indépendants du système d'exploitation.
from pathlib import Path

# altair (installé avec streamlit) : graphiques interactifs Vega-Lite.
import altair as alt
# matplotlib en mode « Agg » (rendu en mémoire, sans fenêtre), choisi avant tout import de pyplot.
import matplotlib

matplotlib.use("Agg")

# pyplot : fermeture des figures ; numpy : tableaux ; pandas : tableaux de données ; streamlit : interface.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
# PIL.Image (Pillow) : assemblage d'images en GIF animé.
from PIL import Image  # noqa: E402

# Module de figures du projet : palette de couleurs (SERIES) et style graphique commun.
from src import visualization as viz  # noqa: E402

# =============================================================================================
# Constantes
# =============================================================================================
# Dossier du projet (parent du dossier appli/).
PROJET = Path(__file__).resolve().parents[1]
# Titre de l'application et sous-titre affichés en tête de page.
TITRE_APPLI = "🌀 Vortex toroïdaux & moteurs à vortex"
SOUS_TITRE = ("Anneaux tourbillonnaires : théorie, méthode des filaments 3D, solveur de Navier–Stokes "
              "axisymétrique et applications aux moteurs à vortex.")
# Libellés des deux onglets principaux (demandés tels quels) ; servent aussi à changer d'onglet par programme.
ONGLET_DOC = "📚 Théorie & Documentation"
ONGLET_SIM = "🚀 Simulation Interactive"
# Verrou unique (module chargé une seule fois) : un seul tracé matplotlib à la fois, quel que soit le fil
# d'exécution (sessions du navigateur, calculs en arrière-plan) ; pyplot n'est pas sûr entre fils.
VERROU = threading.Lock()


class CalculAnnule(Exception):
    """Levée dans le fil de calcul quand l'utilisateur clique sur « Annuler »."""


# =============================================================================================
# Images et mise en forme
# =============================================================================================
def png(fig, dpi=150, serre=True):
    """Image PNG (octets) d'une figure matplotlib, puis libération de la figure.

    dpi = 150 : figures nettes même affichées en grand. ``serre=False`` garde la taille exacte de la figure
    (images d'une animation de dimensions identiques).
    """
    # Tampon mémoire qui reçoit le fichier PNG.
    tampon = io.BytesIO()
    # savefig : rendu ; bbox_inches="tight" rogne les marges ; facecolor : fond de la figure (palette du projet).
    fig.savefig(tampon, format="png", dpi=dpi, bbox_inches="tight" if serre else None, facecolor=fig.get_facecolor())
    # plt.close : libère la mémoire de la figure (indispensable sur un serveur qui trace beaucoup).
    plt.close(fig)
    # getvalue : octets du PNG.
    return tampon.getvalue()


def gif_depuis_png(images_png, ms_par_image=120):
    """Assemble des images PNG de même taille en GIF animé (octets), avec une palette commune."""
    # Lecture des PNG en mémoire ; convert("RGB") retire la couche alpha.
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_png]
    # Palette adaptative de 255 couleurs calculée sur la première image (quantize : couleurs indexées).
    palette = images[0].quantize(colors=255)
    # Même palette pour toutes les images (pas de scintillement), sans tramage.
    cadres = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    # save_all + append_images : images successives ; duration en ms ; loop=0 : boucle infinie.
    tampon = io.BytesIO()
    cadres[0].save(tampon, format="GIF", save_all=True, append_images=cadres[1:], duration=ms_par_image, loop=0)
    return tampon.getvalue()


def duree_lisible(secondes):
    """Durée lisible : « 42 s » ou « 3 min 05 s »."""
    # Moins d'une minute : secondes arrondies.
    if secondes < 60:
        return f"{secondes:.0f} s"
    # divmod : quotient (minutes) et reste (secondes) de la division entière par 60.
    minutes, reste = divmod(int(round(secondes)), 60)
    return f"{minutes} min {reste:02d} s"


def croisements(t, ecart):
    """Instants où ``ecart`` (z₁ − z₂) change de signe — un anneau traverse l'autre —, par interpolation linéaire."""
    # Conversion en tableaux et élimination des valeurs non définies (np.isfinite : ni NaN ni infini).
    t, ecart = np.asarray(t, dtype=float), np.asarray(ecart, dtype=float)
    ok = np.isfinite(ecart)
    t, ecart = t[ok], ecart[ok]
    # np.sign : −1, 0 ou +1 ; np.nonzero(...)[0] : indices i où le signe diffère entre i et i + 1.
    k = np.nonzero(np.sign(ecart[1:]) != np.sign(ecart[:-1]))[0]
    # Zéro de la droite passant par (t_i, écart_i) et (t_i+1, écart_i+1) pour chaque changement de signe.
    return [float(t[i] - ecart[i] * (t[i + 1] - t[i]) / (ecart[i + 1] - ecart[i])) for i in k]


def liste_instants(instants):
    """Texte « 1.46, 4.39 » (ou « aucune ») pour une liste d'instants."""
    # ", ".join : instants à deux décimales séparés par des virgules ; « or » : texte de repli si liste vide.
    return ", ".join(f"{x:.2f}" for x in instants) or "aucune"


def fenetre_axiale(Z, marge=2.0, periode=10.0):
    """Centre et demi-largeur d'une fenêtre fixe en z contenant toute la trajectoire Z(t) (domaine périodique)."""
    # np.nanmin / np.nanmax : extrêmes en ignorant d'éventuels NaN.
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    # Demi-largeur bornée par la demi-période (au-delà, la fenêtre couvre déjà tout le domaine).
    return 0.5 * (zmin + zmax), min(0.5 * (zmax - zmin) + marge, 0.5 * periode)


def sous_echantillon(valeurs, n=60):
    """Au plus n valeurs régulièrement espacées d'une série (mini-courbes des métriques), NaN retirés."""
    # Conversion en tableau ; np.isfinite écarte les valeurs non définies.
    v = np.asarray(valeurs, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    # np.linspace(...).round().astype(int) : indices régulièrement répartis ; tolist : liste de réels Python.
    return v[np.linspace(0, v.size - 1, min(n, v.size)).round().astype(int)].tolist()


# =============================================================================================
# Progression des calculs et tracés
# =============================================================================================
class Progression:
    """Transmet l'avancement d'un calcul à sa tâche (fraction, texte) et l'interrompt si l'utilisateur annule.

    Appelée par les fonctions de calcul, dans le fil d'exécution du calcul : elle ne fait aucun appel Streamlit
    (l'interface lit la tâche et affiche la barre elle-même).
    """

    def __init__(self, tache):
        # Tâche décrite par appli.taches.Tache (attributs fraction, texte, annuler).
        self.tache = tache

    def __call__(self, fraction, texte, forcer=False):
        # Demande d'annulation : on lève une exception qui remonte jusqu'au gestionnaire de la tâche.
        if self.tache.annuler:
            raise CalculAnnule()
        # Fraction bornée à [0, 1] (min/max) et texte affiché sous la barre ; « forcer » est conservé pour
        # compatibilité (la mise à jour est toujours immédiate).
        self.tache.fraction = float(min(max(fraction, 0.0), 1.0))
        self.tache.texte = texte


def tracer(taches, prog, debut, fin):
    """Exécute des tracés (titre, légende, fonction qui crée la figure) en faisant avancer la barre de debut à fin.

    Renvoie la liste des figures sous forme de dictionnaires {titre, legende, png}.
    """
    figures = []
    # Un seul tracé à la fois sur tout le serveur (« with » libère le verrou même en cas d'erreur).
    with VERROU:
        # enumerate : numéro k et contenu de chaque tâche de tracé.
        for k, (titre, legende, fabrique) in enumerate(taches):
            # Avancement proportionnel au nombre de figures déjà tracées.
            prog(debut + (fin - debut) * k / len(taches), f"Figure {k + 1}/{len(taches)} : {titre}")
            # fabrique() crée la figure (fonction du projet) ; png() la convertit en image et la ferme.
            figures.append({"titre": titre, "legende": legende, "png": png(fabrique())})
    return figures


# =============================================================================================
# Métriques et courbes interactives
# =============================================================================================
def metrique(label, valeur, delta=None, aide=None, courbe=None):
    """Description d'une métrique : libellé, valeur, complément gris (delta), info-bulle, mini-courbe éventuelle."""
    return {"label": label, "valeur": valeur, "delta": delta, "aide": aide, "courbe": courbe}


def afficher_metriques(metriques, par_ligne=4):
    """Métriques en rangées de cartes encadrées (st.metric), avec mini-courbe quand une série est fournie."""
    # range(0, n, par_ligne) : début de chaque rangée ; st.columns : colonnes de même largeur.
    for debut in range(0, len(metriques), par_ligne):
        for colonne, m in zip(st.columns(par_ligne), metriques[debut:debut + par_ligne]):
            # delta_color/delta_arrow="off" : complément en gris, sans flèche ; border=True : carte encadrée ;
            # chart_data : mini-courbe de l'évolution de la grandeur (m.get : clé absente des anciens résultats).
            colonne.metric(m["label"], m["valeur"], m["delta"], delta_color="off", delta_arrow="off", help=m["aide"],
                           border=True, chart_data=m.get("courbe"), chart_type="line")


def courbe(titre, df, y, y_label, x="t", x_label="t", couleurs=None, log_x=False):
    """Description d'une courbe interactive : colonnes y de df (format large) en fonction de la colonne x, ou
    df au format long (colonnes x, « valeur », « série ») si y vaut "valeur". ``log_x`` : abscisse logarithmique."""
    return {"titre": titre, "df": df, "x": x, "y": y, "x_label": x_label, "y_label": y_label, "couleurs": couleurs,
            "log_x": log_x}


def graphique(c, hauteur=300):
    """Affiche une courbe interactive Altair : zoom à la molette, déplacement à la souris, valeurs au survol,
    axe vertical ajusté aux données (st.line_chart part de 0 et écraserait des écarts de quelques pour cent).

    Les séries dont le nom finit par « (modèle) » ou « (fluide parfait) » sont tracées en tirets, de la même
    couleur que la série calculée correspondante.
    """
    df = c["df"]
    if "série" in df.columns:
        # Format long déjà prêt ; colonnes renommées en noms simples (x, valeur, serie) pour Vega-Lite.
        long = pd.DataFrame({"x": df[c["x"]].to_numpy(), "valeur": df["valeur"].to_numpy(),
                             "serie": df["série"].astype(str).to_numpy()})
    else:
        # Format large → long : DataFrame.melt crée une ligne par couple (abscisse, série).
        long = df.melt(id_vars=[c["x"]], value_vars=c["y"], var_name="serie", value_name="valeur").rename(columns={c["x"]: "x"})
    # Séries de référence : suffixe reconnu par une expression régulière (str.contains / str.replace).
    suffixe = r"\s*\((?:modèle|fluide parfait)\)$"
    long["trait"] = np.where(long["serie"].str.contains(suffixe, regex=True), "référence", "calcul")
    long["base"] = long["serie"].str.replace(suffixe, "", regex=True)
    # Séries dans leur ordre d'apparition (dict.fromkeys supprime les doublons en gardant l'ordre) et couleurs de
    # la palette du projet dans le même ordre (répétées s'il y a plus de séries que de couleurs).
    ordre = list(dict.fromkeys(long["base"]))
    couleurs = list(c.get("couleurs") or viz.SERIES)
    couleurs = (couleurs * (len(ordre) // len(couleurs) + 1))[:len(ordre)]
    # Échelles : abscisse linéaire ou logarithmique ; ordonnée sans zéro imposé.
    echelle_x = alt.Scale(type="log") if c.get("log_x") else alt.Scale(zero=False)
    graphe = (alt.Chart(long).mark_line(strokeWidth=1.8)
              .encode(x=alt.X("x:Q", title=c["x_label"], scale=echelle_x),
                      y=alt.Y("valeur:Q", title=c["y_label"], scale=alt.Scale(zero=False)),
                      color=alt.Color("base:N", title=None, sort=ordre, scale=alt.Scale(domain=ordre, range=couleurs),
                                      legend=alt.Legend(orient="bottom", labelLimit=320)),
                      # Trait plein pour le calcul, tirets pour la référence (légende seulement si les deux existent).
                      strokeDash=alt.StrokeDash("trait:N", title=None,
                                                scale=alt.Scale(domain=["calcul", "référence"], range=[[1, 0], [6, 4]]),
                                                legend=alt.Legend(orient="bottom") if long["trait"].nunique() > 1 else None),
                      # Info-bulle : série, abscisse et valeur du point survolé.
                      tooltip=[alt.Tooltip("serie:N", title="série"), alt.Tooltip("x:Q", title=c["x_label"], format=".3f"),
                               alt.Tooltip("valeur:Q", title=c["y_label"], format=".4g")])
              .properties(height=hauteur)
              # interactive() : zoom et déplacement sur les deux axes.
              .interactive())
    st.markdown(f"**{c['titre']}**")
    st.altair_chart(graphe, width="stretch")


# =============================================================================================
# Widgets compatibles avec les scénarios guidés
# =============================================================================================
# Un scénario guidé change les réglages en écrivant dans st.session_state. Pour éviter le conflit que Streamlit
# signale quand un widget reçoit à la fois une valeur par défaut et une valeur de l'état de session, la valeur
# initiale est posée une seule fois dans l'état de session et le widget est créé sans argument « value ».
# persist_state="page" conserve la valeur quand le widget n'est pas affiché (autre volet, autre onglet).
def initialiser(cle, defaut):
    """Pose la valeur initiale d'un widget dans l'état de session si elle n'y est pas déjà."""
    if cle not in st.session_state:
        st.session_state[cle] = defaut


def curseur(libelle, mini, maxi, defaut, pas, cle, aide=None, format=None):
    """Curseur numérique (st.slider) à valeur initiale gérée par l'état de session."""
    initialiser(cle, defaut)
    return st.slider(libelle, mini, maxi, step=pas, key=cle, help=aide, format=format, persist_state="page")


def curseur_liste(libelle, options, defaut, cle, aide=None, format_func=str):
    """Curseur sur une liste de valeurs (st.select_slider), par exemple une échelle quasi logarithmique."""
    initialiser(cle, defaut)
    return st.select_slider(libelle, options=options, key=cle, help=aide, format_func=format_func, persist_state="page")


def interrupteur(libelle, defaut, cle, aide=None):
    """Interrupteur marche/arrêt (st.toggle)."""
    initialiser(cle, defaut)
    return st.toggle(libelle, key=cle, help=aide, persist_state="page")


def menu(libelle, options, defaut, cle, format_func=str, aide=None, on_change=None, args=None):
    """Menu déroulant (st.selectbox) ; on_change : fonction appelée quand le choix change (args : ses arguments)."""
    initialiser(cle, defaut)
    # Fonction de rappel transmise seulement si elle est fournie (sinon comportement par défaut de Streamlit).
    rappel = {"on_change": on_change, "args": args} if on_change is not None else {}
    return st.selectbox(libelle, options, key=cle, format_func=format_func, help=aide, persist_state="page", **rappel)
