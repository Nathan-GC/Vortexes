"""
appli/volets/moteurs.py — Volet « moteurs à vortex » (unités SI) : chambre à vortex bidirectionnel et propulseur
pulsé à anneaux tourbillonnaires (src/engine_analysis.py).
"""

# Annotations de type non évaluées à l'exécution.
from __future__ import annotations

# time : durée des tracés ; numpy ; pandas ; streamlit.
import time

import numpy as np
import pandas as pd
import streamlit as st

from appli import calibration
from appli import commun as c
from appli.glossaire import aide
from src import engine_analysis as ea
from src import visualization as viz

CLE = "moteurs"
LIBELLE = "🚀 Moteurs à vortex"
RESUME = ("Chambre de combustion à vortex bidirectionnel (trajectoires, temps de séjour, paroi froide) et "
          "propulseur pulsé (formation des anneaux, carte de rendement).")
PRESENTATION = ("Chambre à vortex bidirectionnel (solution exacte de Vyas & Majdalani, `ea.BidirectionalVortexChamber`) "
                "et propulseur pulsé à anneaux tourbillonnaires (`ea.VortexRingThruster`), en unités SI.")
# Formules clés : paramètre d'injection de la chambre et nombre de formation du propulseur.
FORMULE = (r"\kappa = \frac{Q}{2\pi U a L}\ \ \text{(chambre)},\qquad F^* = \frac{\sqrt{\pi/2}}{\alpha^*} \approx 3{,}8"
           r"\ \ \text{(propulseur)}")
THEORIE = [("Moteurs à vortex", "1.2"), ("Moteurs à vortex", "2.3")]
LIMITES = [
    "Chambre : solution exacte non visqueuse d'écoulement stationnaire, valable pour κ ≪ 1 ; pas de combustion "
    "résolue (modèles de réacteur et de paroi froide).",
    "Propulseur : modèle d'impulsion « slug » avec une fermeture semi-empirique de la surpression, calibrée sur des "
    "mesures publiées ; ordres de grandeur, pas une conception détaillée.",
]
INSTANTANE = False
# Préfixe des clés de ses widgets (réglages enregistrés avec chaque calcul, puis rechargeables).
PREFIXE = "mo_"
_BASE = {"mo_a": 0.05, "mo_L": 0.15, "mo_U": 40.0, "mo_Q": 0.05, "mo_rho": 20.0, "mo_D": 0.02, "mo_Uc": 0.5,
         "mo_Dv": 0.1, "mo_cx": 0.3, "mo_n": 20}
SCENARIOS = {
    # Scénario 1 : cas de référence du rapport.
    "reference": {
        "titre": "Cas de référence du rapport",
        "reglages": dict(_BASE),
        "observer": "κ ≈ 0,027 : une particule injectée près de la paroi fait une dizaine de tours avant de sortir. "
                    "Le rendement du propulseur culmine vers 0,55, à une course L/D proche du nombre de formation F*.",
    },
    # Scénario 2 : débit quadruplé dans la chambre.
    "fort_debit": {
        "titre": "Chambre à fort débit (κ ≈ 0,1)",
        "reglages": dict(_BASE, mo_Q=0.2),
        "observer": "Le débit axial l'emporte davantage sur la rotation : moins de tours, temps de séjour plus court "
                    "(le modèle atteint sa limite de validité).",
        "question": {"texte": "en quadruplant le débit, le temps de séjour moyen…",
                     "choix": ["augmente", "ne change pas", "est divisé par 4"], "reponse": "est divisé par 4",
                     "explication": "Le temps de séjour moyen vaut V/Q : à volume fixé, il est inversement proportionnel "
                                    "au débit."},
    },
    # Scénario 3 : tuyère deux fois plus petite.
    "petite_tuyere": {
        "titre": "Propulseur à petite tuyère (D = 1 cm)",
        "reglages": dict(_BASE, mo_D=0.01),
        "observer": "Pour fournir la même poussée, une tuyère plus petite doit éjecter le fluide plus vite ou plus "
                    "souvent : observez le déplacement de l'optimum sur la carte de rendement.",
    },
    # Scénario 4 : véhicule trois fois plus rapide.
    "rapide": {
        "titre": "Véhicule rapide (U∞ = 1,5 m/s)",
        "reglages": dict(_BASE, mo_Uc=1.5),
        "observer": "La traînée, donc la poussée requise, est multipliée par 9 : les points faisables se raréfient sur "
                    "la carte de rendement.",
    },
}


def controles():
    """Widgets du volet regroupés en deux sections repliables ; renvoie le dictionnaire des paramètres."""
    p = {}
    with st.expander("🔥 Chambre à vortex bidirectionnel", expanded=True):
        # Géométrie, injection et gaz de la chambre (format : nombre de décimales).
        p["a"] = c.curseur("Rayon de la chambre a [m]", 0.01, 0.30, 0.05, 0.005, "mo_a", format="%.3f")
        p["L"] = c.curseur("Longueur L [m]", 0.05, 1.0, 0.15, 0.01, "mo_L")
        p["U"] = c.curseur("Vitesse d'injection tangentielle U [m/s]", 5.0, 150.0, 40.0, 1.0, "mo_U")
        p["Q"] = c.curseur("Débit volumique Q [m³/s]", 0.005, 0.5, 0.05, 0.005, "mo_Q", aide("Paramètre d'injection κ"),
                           "%.3f")
        p["rho"] = c.curseur("Masse volumique du gaz ρ [kg/m³]", 1.0, 60.0, 20.0, 1.0, "mo_rho")
    with st.expander("🐙 Propulseur pulsé (dans l'eau)", expanded=True):
        # Tuyère, véhicule (traînée) et résolution de la carte de rendement.
        p["D"] = c.curseur("Diamètre de la tuyère D [m]", 0.005, 0.10, 0.02, 0.005, "mo_D", format="%.3f")
        p["U_croisiere"] = c.curseur("Vitesse de croisière U∞ [m/s]", 0.1, 3.0, 0.5, 0.1, "mo_Uc")
        p["D_vehicule"] = c.curseur("Diamètre du véhicule [m]", 0.03, 0.5, 0.1, 0.01, "mo_Dv",
                                    "Sert au calcul de la traînée, égale à la poussée requise.")
        p["cx"] = c.curseur("Coefficient de traînée C_x", 0.05, 1.2, 0.3, 0.05, "mo_cx")
        p["n_carte"] = c.curseur("Résolution de la carte de rendement", 10, 30, 20, 2, "mo_n",
                                 "Nombre de valeurs de L/D et de fréquences testées.")
    return p


def verifier(p):
    """Validité du modèle de chambre (κ ≪ 1) et durée estimée."""
    alertes = []
    # κ = Q/(2πUaL) : rapport débit axial / rotation.
    kappa = p["Q"] / (2.0 * np.pi * p["U"] * p["a"] * p["L"])
    if kappa > 0.1:
        alertes.append(f"κ = Q/(2πUaL) = {kappa:.3f} : le modèle de Vyas & Majdalani suppose κ ≪ 1.")
    # Coût proportionnel au nombre de points de la carte.
    return alertes, calibration.estimation("moteurs", p["n_carte"] ** 2, None, 9.0,
                                           f"carte {p['n_carte']} × {p['n_carte']}")


def etiquette(p):
    """Description courte d'un calcul."""
    # Dimensions et injection de la chambre, tuyère et vitesse du véhicule.
    return (f"chambre a {p['a']:g} m, U {p['U']:g} m/s, Q {p['Q']:g} m³/s · tuyère {100 * p['D']:g} cm, "
            f"U∞ {p['U_croisiere']:g} m/s")


def raffinement(p, facteur):
    """Pas de vérification de convergence : modèles analytiques et semi-analytiques."""
    return None


def calculer(p, prog, complet=True):
    """Analyse de la chambre et du propulseur (fonctions de engine_analysis)."""
    prog(0.05, "Chambre à vortex : temps de séjour et trajectoires")
    debut = time.perf_counter()
    # Chambre à vortex bidirectionnel (fonction du projet).
    ch = ea.BidirectionalVortexChamber(a=p["a"], L=p["L"], U=p["U"], Q=p["Q"], rho=p["rho"])
    # Distribution des temps de séjour et trois trajectoires injectées à 0,75 a, 0,85 a et 0,95 a.
    rtd = ch.residence_time_distribution()
    trajectoires = [ch.trajectory(f * ch.a, theta0=k * 2.1) for k, f in enumerate((0.75, 0.85, 0.95))]
    # Chute de pression paroi → axe (r = 0).
    dp = float(ch.pressure_drop(np.array([0.0]))[0])
    # Paroi froide (flux pariétal relatif pour plusieurs Péclet radiaux) et conversion d'un réacteur.
    paroi = {pe: ea.cold_wall_heat_flux(pe / ch.kappa, ch.kappa)["ratio"] for pe in (1.0, 2.0, 5.0, 10.0, 20.0)}
    conversion = ea.reactor_conversion(np.geomspace(0.1, 30.0, 60), ch)
    prog(0.2, "Propulseur pulsé : carte de rendement")
    # Propulseur pulsé de diamètre D.
    thr = ea.VortexRingThruster(D=p["D"])
    # Poussée requise = traînée du véhicule à la vitesse de croisière.
    poussee = float(ea.body_drag(p["U_croisiere"], diameter=p["D_vehicule"], cd=p["cx"]))
    # Carte du rendement η(L/D, f) à poussée imposée.
    n = p["n_carte"]
    carte = thr.efficiency_map(np.linspace(0.5, 8.0, n), np.geomspace(0.5, 40.0, n), p["U_croisiere"], poussee)
    carte["formation_number"] = thr.formation_number()
    # Durée du calcul (calibration).
    t_calcul = time.perf_counter() - debut
    alertes = []
    # Meilleur rendement et meilleur gain (np.nanargmax ignore les points infaisables, notés NaN).
    if np.isfinite(carte["efficiency"]).any():
        i, j = np.unravel_index(np.nanargmax(carte["efficiency"]), carte["efficiency"].shape)
        ig, jg = np.unravel_index(np.nanargmax(carte["gain"]), carte["gain"].shape)
        eta = c.metrique("Rendement propulsif maximal η", f"{carte['efficiency'][i, j]:.3f}",
                         f"L/D = {carte['L_over_D'][i]:.2f}, f = {carte['frequency'][j]:.1f} Hz", aide("Rendement propulsif η"))
        gain = c.metrique("Gain sur le jet continu", f"{100 * (carte['gain'][ig, jg] - 1):+.1f} %",
                          f"L/D = {carte['L_over_D'][ig]:.2f}, f = {carte['frequency'][jg]:.1f} Hz")
        eta_max = float(carte["efficiency"][i, j])
    else:
        eta = c.metrique("Rendement propulsif maximal η", "—", "aucun point faisable")
        gain = c.metrique("Gain sur le jet continu", "—")
        eta_max = float("nan")
        alertes.append("Aucun point de fonctionnement faisable : la poussée requise est hors d'atteinte pour cette tuyère.")
    # Grandeurs clés ; calcul léger : on s'arrête là.
    grandeurs = {"κ": float(ch.kappa), "Temps de séjour moyen (ms)": float(rtd["mean"] * 1e3), "η maximal": eta_max}
    sortie = {"grandeurs": grandeurs, "type_cout": CLE, "mesures": {"t_calcul": t_calcul, "n_pas": 1, "taille": n * n}}
    if not complet:
        return sortie
    # Anneau de tête pour quelques courses L/D (vitesse de jet 1 m/s) : tableau de données.
    F = float(thr.formation_number())
    lignes = []
    for ld in (1.0, 2.0, F, 6.0):
        an = thr.leading_ring(ld, 1.0)
        lignes.append({"L/D": round(ld, 2), "β (surpression)": float(thr.overpressure_fraction(ld)),
                       "Γ/(U_j D)": an["circulation"] / thr.D, "U_anneau/U_j": an["U_self"], "a/R": an["core_ratio"],
                       "V_bulle/V_slug": an["bubble_to_slug_volume"], "C_A": an["added_mass"],
                       "bouclage impulsion": an["impulse_closure"]})
    # Chronomètre des tracés.
    debut_figures = time.perf_counter()
    # Figures du rapport « moteurs » (module visualization).
    taches = [
        ("Chambre à vortex", "Vitesses, lignes de courant, trajectoires hélicoïdales et distribution des temps de séjour.",
         lambda: viz.plot_bidirectional_vortex(ch, trajectoires, rtd)),
        ("Paroi froide", "Rapport du flux pariétal à celui d'une chambre sans vortex, en fonction du Péclet.",
         lambda: viz.plot_cold_wall(ch.kappa, ea.cold_wall_heat_flux)),
        ("Conversion", "Conversion d'une réaction d'ordre 1 en fonction du nombre de Damköhler.",
         lambda: viz.plot_reactor_conversion(conversion)),
        ("Formation des anneaux", "Propulseur : formation de l'anneau de tête, surpression et nombre de formation F*.",
         lambda: viz.plot_thruster_formation(thr)),
        ("Carte de rendement", "Rendement propulsif η(L/D, f) à poussée imposée ; zones blanches : points infaisables.",
         lambda: viz.plot_efficiency_map(carte)),
    ]
    # Tracés (barre de 35 % à 100 %).
    figures = c.tracer(taches, prog, 0.35, 1.0)
    sortie["mesures"]["t_figures"] = time.perf_counter() - debut_figures
    # Métriques : chambre (κ, séjour, pression, swirl), propulseur (F*, poussée, rendement).
    metriques = [
        c.metrique("κ = Q/(2πUaL)", f"{ch.kappa:.4f}", None, aide("Paramètre d'injection κ")),
        c.metrique("Temps de séjour moyen", f"{rtd['mean'] * 1e3:.2f} ms", f"≈ {rtd['n_tanks']:.1f} réacteurs en série",
                   aide("Temps de séjour")),
        c.metrique("Chute de pression paroi → axe", f"{dp / 1e5:.3f} bar"),
        c.metrique("Swirl en sortie", f"{ch.exit_swirl_number():.2f}", f"gouttes piégées si d > {ch.droplet_cutoff_diameter() * 1e6:.1f} µm"),
        c.metrique("Nombre de formation F*", f"{F:.2f}", None, aide("Nombre de formation F*")),
        c.metrique("Poussée requise", f"{poussee:.3f} N", "traînée du véhicule"),
        eta, gain,
    ]
    # Trajectoires injectées et rapports de flux pariétal.
    notes = ["Trajectoires : " + " ; ".join(f"injection à {tr['r'][0] / ch.a:.2f} a → séjour {tr['residence_time'] * 1e3:.1f} ms, "
                                            f"{tr['revolutions']:.1f} tours" for tr in trajectoires) + ".",
             "Paroi froide Φ(Pe·κ) : " + ", ".join(f"{k:g} → {v:.3f}" for k, v in paroi.items()) + "."]
    # Rendement en fonction de la fréquence, une courbe par valeur de L/D (une sur quatre), abscisse logarithmique.
    lignes_eta = [pd.DataFrame({"f [Hz]": carte["frequency"], "valeur": carte["efficiency"][k],
                                "série": f"L/D = {carte['L_over_D'][k]:.2f}"}) for k in range(0, n, 4)]
    courbes = [c.courbe("Rendement η en fonction de la fréquence d'éjection (une courbe par course L/D)",
                        pd.concat(lignes_eta), "valeur", "η", x="f [Hz]", x_label="f [Hz]", log_x=True)]
    # Séries comparables : rendement maximal sur L/D pour chaque fréquence (NaN si aucune course n'est faisable).
    eta_f = np.array([np.nanmax(col) if np.isfinite(col).any() else np.nan for col in carte["efficiency"].T])
    comparables = [{"nom": "Rendement maximal η(f)", "x": carte["frequency"], "y": eta_f, "x_label": "f [Hz]",
                    "y_label": "η"}]
    # Éléments de la sortie (données : tableau de l'anneau de tête).
    sortie.update({"metriques": metriques, "notes": notes, "alertes": alertes, "figures": figures, "gif": None,
                   "courbes": courbes, "comparables": comparables, "donnees": pd.DataFrame(lignes),
                   "fichier_csv": "anneau_de_tete.csv"})
    return sortie
