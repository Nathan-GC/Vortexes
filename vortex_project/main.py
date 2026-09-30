"""
main.py — Point d'entrée du projet « Vortex toroïdaux & moteurs à vortex ».

Exemples
--------
    python main.py all                  # tout (≈ 8 min) : théorie, lagrangien 3D, axisymétrique, moteurs, avancé
    python main.py all --quick          # version rapide (≈ 2 min 30, résolutions réduites)
    python main.py theory               # grandeurs analytiques + vortex de Hill
    python main.py particles --scenario leapfrog --animate mp4
    python main.py axisym --re 1000 --t-end 20
    python main.py engines
    python main.py advanced             # saute-mouton visqueux, espace des phases, tubes 3D, rendu volumique
    python main.py advanced --smoke-video

Sorties : ``outputs/figures`` (PNG), ``outputs/animations`` (MP4 ou GIF), ``outputs/data`` (CSV),
``outputs/summary.json`` (chiffres clés). ``--docs`` recopie les figures du rapport dans ``docs/figures``.
"""

# Annotations non évaluées à l'exécution.
from __future__ import annotations

# argparse : lecture des arguments de la ligne de commande.
import argparse
# json : lecture/écriture de la synthèse chiffrée summary.json.
import json
# os : chemins et création de dossiers.
import os
# shutil.copy2 : copie de fichiers (conserve les dates) pour --docs.
import shutil
# sys.stdout : sortie standard (réglée en UTF-8 pour les caractères grecs sous Windows).
import sys
# time.perf_counter : mesure de la durée totale.
import time

# numpy : calculs sur tableaux.
import numpy as np

# Modules du projet (paquet « src ») importés sous des alias courts.
from src import engine_analysis as ea
from src import solver_2d_axisymmetric as ax2d
from src import theory
from src import visualization as viz
from src import visualization_3d as v3
from src import vortex_particles as vp

# Figures recopiées dans docs/figures par l'option --docs (utilisées par les rapports Markdown).
DOC_FIGURES = [
    "theory_hill_vortex.png", "particles_kernel_convergence.png", "particles_leapfrog_diagnostics.png",
    "particles_leapfrog_snapshots.png", "particles_collision_diagnostics.png", "particles_offset_snapshots.png",
    "axisym_fields_final.png", "axisym_vorticity_snapshots.png", "axisym_speed_validation.png",
    "engine_bidirectional_vortex.png", "engine_cold_wall.png", "engine_reactor_conversion.png",
    "engine_thruster_formation.png", "engine_efficiency_map.png",
    # visualisations et diagnostics avancés
    "axisym_energy_budget.png", "axisym_dissipation_map.png", "axisym_hovmoller.png", "axisym_isosurfaces.png",
    "axisym_volume_rendering.png", "leapfrog_ns_isosurfaces.png", "leapfrog_ns_hovmoller.png",
    "leapfrog_ns_dissipation.png", "leapfrog_ns_energy_budget.png", "leapfrog_ns_volume_rendering.png",
    "leapfrog_phase_diagram.png", "leapfrog_filaments_hovmoller.png", "tubes_thick_ring.png", "tubes_collision.png",
    "smoke_offset_collision.png",
]


def _paths(out):
    # Dictionnaire des sous-dossiers de sortie (os.path.join assemble les chemins).
    p = {k: os.path.join(out, k) for k in ("figures", "animations", "data")}
    # Création des dossiers s'ils n'existent pas.
    for d in p.values():
        os.makedirs(d, exist_ok=True)
    return p


def _banner(text):
    # Affiche un bandeau de titre encadré de lignes de « = » (88 caractères).
    print("\n" + "=" * 88 + f"\n{text}\n" + "=" * 88)


# =============================================================================================
# Phase 1 — théorie
# =============================================================================================
def run_theory(out, quick=False):
    _banner("PHASE 1 — Modèles analytiques")
    p = _paths(out)
    # Invariants du vortex de Hill pour a = U = 1.
    hill = theory.hill_invariants(1.0, 1.0)
    # Constantes C des différents types de noyau.
    consts = {
        "uniforme (Kelvin)": theory.core_constant("uniform"),
        "gaussien (Saffman)": theory.core_constant("gaussian"),
        "creux (Hicks)": theory.core_constant("hollow"),
        "Rosenhead–Moore": theory.core_constant("rosenhead_moore"),
    }
    # Vérification numérique à partir des profils de vorticité (uniforme : indicatrice de s < 1 ;
    # gaussien : exp(-s²)) ; « lambda s: ... » définit la fonction profil à la volée.
    num_u = theory.core_constant_from_profile(lambda s: np.where(s < 1.0, 1.0, 0.0), n=400001)
    num_g = theory.core_constant_from_profile(lambda s: np.exp(-s**2))
    print("Constante de noyau C dans U = Γ/(4πR)[ln(8R/a) − C] :")
    # dict.items() donne les couples (nom, valeur) ; « :22s » aligne le texte sur 22 caractères.
    for k, v in consts.items():
        print(f"   {k:22s} C = {v:.4f}")
    print(f"   vérification numérique : uniforme {num_u:.4f}, gaussien {num_g:.4f}")
    # Formatage « % » à l'ancienne : chaque %.nf est remplacé par la valeur correspondante.
    print("Vortex de Hill (a = U = ρ = 1) : Γ = %.3f, I = %.4f, E = %.4f, E_adim = %.4f, C_A = %.2f"
          % (hill["circulation"], hill["impulse"], hill["energy"], hill["dimensionless_energy"], hill["added_mass_coefficient"]))
    # Nombre de formation pour α* = 0.33.
    fs = theory.formation_number(0.33)
    print(f"Nombre de formation : F* = √(π/2)/0.33 = {fs:.3f}")
    print("Correspondance Rosenhead–Moore : δ = %.4f a (noyau uniforme), δ = %.4f σ (noyau gaussien)"
          % (theory.rosenhead_moore_delta(1.0, "uniform"), theory.rosenhead_moore_delta(1.0, "gaussian")))
    # Vitesse adimensionnelle U·4πR/Γ pour quelques épaisseurs de cœur.
    for aR in (0.05, 0.1, 0.2):
        print(f"   a/R = {aR:4.2f} : U·4πR/Γ = {theory.ring_speed(1, 1, aR, 'uniform') * 4 * np.pi:.4f} (uniforme), "
              f"{theory.saffman_speed(1, 1, aR) * 4 * np.pi:.4f} (gaussien)")
    # Figure du vortex de Hill enregistrée en PNG.
    viz.save_figure(viz.plot_hill_vortex(), os.path.join(p["figures"], "theory_hill_vortex.png"))
    # Résumé pour summary.json (float(...) convertit les réels numpy en réels Python sérialisables).
    return {"hill": {k: float(v) for k, v in hill.items()}, "core_constants": consts,
            "core_constants_numerical": {"uniform": num_u, "gaussian": num_g}, "formation_number": fs}


# =============================================================================================
# Phase 2A — méthode lagrangienne
# =============================================================================================
def run_particles(out, quick=False, scenario="all", animate="mp4"):
    _banner("PHASE 2A — Méthode lagrangienne 3D (filaments / particules vortex)")
    p = _paths(out)
    summary = {}
    # --- convergence du noyau
    # Rayon de régularisation utilisé dans tous les scénarios.
    delta = 0.1
    # Vitesse de référence de l'anneau de Rosenhead–Moore (formule asymptotique).
    U_rm = theory.rosenhead_moore_ring_speed(1.0, 1.0, delta)
    table = {}
    # Deux noyaux, chacun avec un libellé pour la légende.
    for kernel, label in (("segment", "segments (intégration exacte)"), ("particle", "particules (point milieu)")):
        rows = []
        # Nombre de nœuds croissant → pas Δs = 2πR/N décroissant.
        for n in (32, 64, 128, 256, 512):
            sim = vp.VortexFilamentSimulation(vp.make_ring(1.0, 1.0, n, delta=delta), kernel=kernel)
            # velocity()[:, 2].mean() : vitesse axiale moyenne des nœuds = vitesse de l'anneau.
            rows.append((2 * np.pi / n / delta, sim.velocity()[:, 2].mean() / U_rm - 1.0))
        table[label] = rows
        print(f"Vitesse propre ({kernel:8s}) : erreur relative à N=128 : {rows[2][1]:+.4f} ; à N=512 : {rows[4][1]:+.4f}")
    viz.save_figure(viz.plot_kernel_convergence(table), os.path.join(p["figures"], "particles_kernel_convergence.png"))
    # list(map(float, r)) convertit chaque couple en liste de réels Python (JSON).
    summary["kernel_convergence"] = {k: [list(map(float, r)) for r in v] for k, v in table.items()}

    # Images par seconde des animations.
    fps = 24
    if scenario in ("all", "leapfrog"):
        # Résolution et durée selon le mode (rapide ou complet).
        n_nodes, t_end, dt = (64, 20.0, 0.05) if quick else (128, 40.0, 0.05)
        sep = 0.6
        print(f"Saute-mouton : 2 anneaux R=1, δ={delta}, séparation {sep}, N={n_nodes}/anneau, t_end={t_end}")
        sim = vp.VortexFilamentSimulation(vp.leapfrog_scenario(delta=delta, separation=sep, n_nodes=n_nodes))
        # Intégration (sauvegarde tous les 4 pas, énergie toutes les 10 sauvegardes).
        hist = sim.run(t_end, dt, save_every=4, energy_every=10, verbose=True)
        # Modèle coaxial de référence intégré sur la même durée (801 instants).
        model = theory.CoaxialRingsModel([1.0, 1.0], [0.0, sep], 1.0, delta, core="rosenhead_moore").integrate(t_end, 801)
        viz.save_figure(viz.plot_leapfrog_diagnostics(hist, model), os.path.join(p["figures"], "particles_leapfrog_diagnostics.png"))
        viz.save_figure(viz.plot_filament_snapshots(hist, n=5, labels=["anneau A", "anneau B"], follow=True,
                                                    title="Saute-mouton — instantanés 3D"),
                        os.path.join(p["figures"], "particles_leapfrog_snapshots.png"))
        d = hist.as_arrays()
        # Dérive relative maximale de l'impulsion axiale.
        dP = float(np.abs(d["impulse"][:, 2] - d["impulse"][0, 2]).max() / d["impulse"][0, 2])
        # Énergies effectivement calculées (np.isfinite écarte les NaN) et leur dérive.
        E = d["energy"][np.isfinite(d["energy"])]
        dE = float(np.abs(E - E[0]).max() / abs(E[0]))
        summary["leapfrog"] = {"R_min_sim": float(d["radius"].min()), "R_max_sim": float(d["radius"].max()),
                               "R_min_model": float(model["R"].min()), "R_max_model": float(model["R"].max()),
                               "impulse_drift": dP, "energy_drift": dE, "wall_time_s": hist.wall_time}
        print(f"   R ∈ [{d['radius'].min():.3f}, {d['radius'].max():.3f}] (modèle coaxial [{model['R'].min():.3f}, "
              f"{model['R'].max():.3f}]) ; dérive impulsion {dP:.1e}, énergie {dE:.1e} ; {hist.wall_time:.1f} s")
        if animate != "none":
            # Animation 3D qui suit les anneaux (fenêtre de 3.2 de côté).
            path = viz.animate_filaments_3d(hist, os.path.join(p["animations"], f"leapfrog.{animate}"), fps=fps,
                                            follow=True, window=3.2, title="Saute-mouton")
            print(f"   animation : {path}")
            summary["leapfrog"]["animation"] = path

    if scenario in ("all", "collision"):
        n_nodes, t_end, dt = (64, 14.0, 0.05) if quick else (96, 14.0, 0.04)
        sep = 3.0
        print(f"Collision frontale : séparation {sep}, N={n_nodes}/anneau, t_end={t_end}")
        sim = vp.VortexFilamentSimulation(vp.head_on_collision_scenario(delta=delta, separation=sep, n_nodes=n_nodes))
        hist = sim.run(t_end, dt, save_every=5, verbose=True)
        # Modèle coaxial : deux anneaux de circulations opposées.
        model = theory.CoaxialRingsModel([1.0, 1.0], [-0.5 * sep, 0.5 * sep], [1.0, -1.0], delta,
                                         core="rosenhead_moore").integrate(t_end, 801)
        viz.save_figure(viz.plot_collision_diagnostics(hist, model, delta=delta), os.path.join(p["figures"], "particles_collision_diagnostics.png"))
        viz.save_figure(viz.plot_filament_snapshots(hist, n=4, labels=["anneau A (Γ > 0)", "anneau B (Γ < 0)"],
                                                    title="Collision frontale — instantanés 3D"),
                        os.path.join(p["figures"], "particles_collision_snapshots.png"))
        d = hist.as_arrays()
        summary["collision"] = {"R_final_sim": float(d["radius"][-1, 0]), "R_final_model": float(model["R"][0, -1]),
                                "nodes_final": int(d["n_nodes"][-1]), "wall_time_s": hist.wall_time}
        print(f"   R final : {d['radius'][-1, 0]:.3f} (modèle {model['R'][0, -1]:.3f}) ; nœuds : {d['n_nodes'][0]} → {d['n_nodes'][-1]}")
        if animate != "none":
            path = viz.animate_filaments_3d(hist, os.path.join(p["animations"], f"collision.{animate}"), fps=fps,
                                            title="Collision frontale", elev=12)
            print(f"   animation : {path}")
            summary["collision"]["animation"] = path

    if scenario in ("all", "offset"):
        n_nodes, t_end, dt = (64, 10.0, 0.05) if quick else (96, 11.0, 0.04)
        print(f"Collision décalée (3D) : N={n_nodes}/anneau, t_end={t_end}")
        sim = vp.VortexFilamentSimulation(vp.offset_collision_scenario(delta=delta, separation=3.0, offset=0.9,
                                                                       n_nodes=n_nodes))
        hist = sim.run(t_end, dt, save_every=5, verbose=True)
        viz.save_figure(viz.plot_filament_snapshots(hist, n=4, labels=["anneau A (Γ > 0)", "anneau B (Γ < 0)"],
                                                    title="Collision décalée — dynamique 3D", elev=15, azim=-35),
                        os.path.join(p["figures"], "particles_offset_snapshots.png"))
        summary["offset"] = {"wall_time_s": hist.wall_time, "nodes_final": int(hist.n_nodes[-1])}
        if animate != "none":
            # Animation avec rotation de la caméra de 60°.
            path = viz.animate_filaments_3d(hist, os.path.join(p["animations"], f"offset_collision.{animate}"), fps=fps,
                                            title="Collision décalée", elev=15, azim=-35, rotate_deg=60)
            print(f"   animation : {path}")
            summary["offset"]["animation"] = path
    return summary


# =============================================================================================
# Phase 2B — solveur axisymétrique
# =============================================================================================
def run_axisym(out, quick=False, Re=1000.0, t_end=None):
    _banner("PHASE 2B — Solveur Navier–Stokes axisymétrique (projection, grille MAC)")
    p = _paths(out)
    # Durée et résolution par défaut selon le mode.
    t_end = (10.0 if quick else 20.0) if t_end is None else t_end
    dr = 0.04 if quick else 0.025
    # Instants des instantanés de champs : début, milieu, fin.
    snaps = (0.0, 0.5 * t_end, t_end)
    print(f"Anneau de Lamb–Oseen : R = 1, a = 0.2, Γ = 1, Re_Γ = {Re:g}, Δr = {dr}, t_end = {t_end}")
    # Simulation complète + comparaison à la théorie (profils axiaux enregistrés pour le Hovmöller).
    res = ax2d.run_vortex_ring(R=1.0, a=0.2, Gamma=1.0, Re=Re, t_end=t_end, dr=dr, snapshot_times=snaps, verbose=True,
                               record_profiles=True)
    d = res["diagnostics"]
    print(f"   {res['grid']}  —  {res['wall_time']:.1f} s")
    # Position de l'anneau à chaque instantané (np.interp : interpolation linéaire de Z(t)).
    Z_s = [np.interp(s["t"], d["t"], d["Z"]) for s in res["snapshots"]]
    # Vitesse finale moyennée sur les 10 derniers diagnostics.
    U_last = float(np.mean(res["U_sim"][-10:]))
    viz.save_figure(viz.plot_axisymmetric_fields(res["snapshots"][-1], Z_ring=Z_s[-1], U_ring=U_last),
                    os.path.join(p["figures"], "axisym_fields_final.png"))
    viz.save_figure(viz.plot_vorticity_snapshots(res["snapshots"], Z_s), os.path.join(p["figures"], "axisym_vorticity_snapshots.png"))
    viz.save_figure(viz.plot_ring_speed_validation(res), os.path.join(p["figures"], "axisym_speed_validation.png"))
    # Export CSV des séries temporelles (np.column_stack assemble les colonnes).
    header = "t,Z,R,sigma,U_sim,U_saffman,U_saffman_fm,circulation,energy,dissipation,max_div"
    data = np.column_stack([d["t"], d["Z"], d["R"], d["sigma"], res["U_sim"], res["U_saffman"], res["U_saffman_fm"],
                            d["circulation"], d["energy"], d["dissipation"], d["max_div"]])
    # np.savetxt : écriture texte ; comments="" évite le « # » devant l'en-tête.
    np.savetxt(os.path.join(p["data"], "axisym_diagnostics.csv"), data, delimiter=",", header=header, comments="")
    # Erreur relative du bilan d'énergie : (E0 - E_fin) comparé à ∫ dissipation dt.
    from scipy.integrate import trapezoid
    budget_err = abs((d["energy"][0] - d["energy"][-1]) - trapezoid(d["dissipation"], d["t"])) / (d["energy"][0] - d["energy"][-1])
    summary = {"grid": repr(res["grid"]), "wall_time_s": res["wall_time"], "max_divergence": float(d["max_div"].max()),
               "energy_budget_relative_error": float(budget_err), "comparison": []}
    print("    t      U_sim    Saffman   (écart)   Saffman+FM (écart)")
    # Comparaison à 4 instants (25 %, 50 %, 75 %, 100 % de la durée).
    for tt in np.linspace(0.25 * t_end, t_end, 4):
        # Indice du diagnostic le plus proche de tt (np.argmin de l'écart absolu).
        k = int(np.argmin(np.abs(d["t"] - tt)))
        e1 = res["U_sim"][k] / res["U_saffman"][k] - 1
        e2 = res["U_sim"][k] / res["U_saffman_fm"][k] - 1
        # « :+.1% » affiche un pourcentage signé avec une décimale.
        print(f"   {d['t'][k]:5.1f}   {res['U_sim'][k]:.4f}   {res['U_saffman'][k]:.4f}  ({e1:+.1%})   "
              f"{res['U_saffman_fm'][k]:.4f}  ({e2:+.1%})")
        summary["comparison"].append({"t": float(d["t"][k]), "U_sim": float(res["U_sim"][k]),
                                      "U_saffman": float(res["U_saffman"][k]), "U_saffman_fm": float(res["U_saffman_fm"][k])})
    print(f"   div max = {d['max_div'].max():.1e} ; erreur du bilan d'énergie = {budget_err:.1e}")
    # --- visualisations et diagnostics avancés de l'anneau unique
    # Bilan temporel E_k, P_z, Ω (validation de l'absence de dissipation numérique).
    viz.save_figure(viz.plot_energy_enstrophy_budget(res), os.path.join(p["figures"], "axisym_energy_budget.png"))
    # Cartes du taux de dissipation Φ(r, z) aux instants des instantanés (Z_s : position de l'anneau à ces instants).
    viz.save_figure(viz.plot_dissipation_map(res["snapshots"], Z_s), os.path.join(p["figures"], "axisym_dissipation_map.png"))
    # Trajectoire théorique : Z(0) + ∫ U_Saffman+FM dt (cumulative_trapezoid = intégrale cumulée).
    from scipy.integrate import cumulative_trapezoid
    Z_th = d["Z"][0] + cumulative_trapezoid(res["U_saffman_fm"], d["t"], initial=0.0)
    # Diagrammes de Hovmöller avec la trajectoire simulée (trait fin) et la trajectoire théorique (tirets).
    viz.save_figure(viz.plot_hovmoller(res["profiles"], tracks=[(d["t"], d["Z"], "centroïde simulé")],
                                       theory_tracks=[(d["t"], Z_th, "∫ U Saffman + F–M dt")],
                                       title="Anneau visqueux — Hovmöller z–t : ralentissement par épaississement du cœur"),
                    os.path.join(p["figures"], "axisym_hovmoller.png"))
    # Isosurfaces Q (couleur |u|) et λ2 (couleur = dépression) du dernier instantané, reconstruites par révolution.
    viz.save_figure(v3.plot_isosurfaces(res["snapshots"][-1], Z_s[-1]), os.path.join(p["figures"], "axisym_isosurfaces.png"))
    # Rendu volumique de l'enstrophie reconstruite par révolution (début et fin, deux points de vue).
    # Taille des images (pixels) et résolution de la grille 3D (réduites en mode --quick).
    res_px = (300, 240) if quick else (420, 330)
    vol_res = 100 if quick else 150
    images, titles = [], []
    # Trois vues : début et fin en vue oblique (élévation 18°), fin en vue plongeante (72°).
    for s, Zc, (elev, az) in zip((res["snapshots"][0], res["snapshots"][-1], res["snapshots"][-1]), (Z_s[0], Z_s[-1], Z_s[-1]),
                                 ((18, -60), (18, -60), (72, -60))):
        # Volume de ½ω² par révolution autour de l'anneau, puis lancer de rayons à émission-absorption.
        vol, org, sp = v3.axisym_enstrophy_volume(s, Zc, half_width=1.6, r_max=1.8, resolution=vol_res)
        images.append(v3.volume_render(vol, org, sp, elev=elev, azim=az, image_size=res_px))
        titles.append(f"t = {s['t']:.0f}, élévation {elev}°")
    viz.save_figure(v3.plot_volume_renderings(images, titles, title="Anneau visqueux — rendu volumique de l'enstrophie ½ω²"),
                    os.path.join(p["figures"], "axisym_volume_rendering.png"))
    # Résidu maximal du bilan d'énergie (dissipation numérique) pour la synthèse.
    E_th = d["energy"][0] - cumulative_trapezoid(d["dissipation"], d["t"], initial=0.0)
    summary["energy_budget_max_residual"] = float(np.abs(d["energy"] - E_th).max() / d["energy"][0])
    return summary


# =============================================================================================
# Visualisations et diagnostics avancés (saute-mouton visqueux, tubes, rendu volumique)
# =============================================================================================
def _crossings(t, dz):
    """Instants où z1 - z2 change de signe (un anneau traverse l'autre), par interpolation linéaire."""
    # Élimination des instants non suivis (NaN), puis indices où le signe change entre deux échantillons.
    ok = np.isfinite(dz)
    # Conversion en tableaux numpy et filtrage par le masque booléen « ok ».
    t, dz = np.asarray(t)[ok], np.asarray(dz)[ok]
    # np.sign : -1, 0 ou +1 ; np.nonzero(...)[0] : indices i tels que le signe diffère entre i et i+1.
    k = np.nonzero(np.sign(dz[1:]) != np.sign(dz[:-1]))[0]
    # Zéro de la droite passant par (t_i, dz_i) et (t_i+1, dz_i+1) pour chaque changement de signe.
    return [float(t[i] - dz[i] * (t[i + 1] - t[i]) / (dz[i + 1] - dz[i])) for i in k]


def run_advanced(out, quick=False, smoke_video=False):
    _banner("VISUALISATIONS AVANCÉES — saute-mouton visqueux, espace des phases, tubes, rendu volumique")
    # Dossiers de sortie (figures, animations, données) et dictionnaire de synthèse de cette phase.
    p = _paths(out)
    summary = {}
    # --- 1. Saute-mouton visqueux (Navier–Stokes axisymétrique, deux anneaux de Lamb–Oseen)
    # Cœurs fins σ0 = 0.10 : à la traversée, la distance des cœurs (≈ 0.48) vaut ≈ 4.8 σ, au-dessus du
    # seuil de fusion de deux tourbillons co-rotatifs (≈ 4 σ) ; l'épaississement visqueux σ² = σ0² + 4νt
    # finit par le franchir. Les filaments équivalents ont δ = σ0 · e^{C_gauss - 1} ≈ 0.064.
    sigma0 = 0.10
    delta_eq = theory.rosenhead_moore_delta(sigma0, "gaussian")
    # Écart axial initial des deux anneaux, durée simulée et pas radial (réduits en mode --quick).
    sep = 0.6
    t_end = 6.0 if quick else 12.0
    dr = 0.025 if quick else 0.02
    # Deux anneaux identiques (R = 1, Γ = 1) : l'arrière en z = 2, l'avant en z = 2 + sep.
    rings = [{"R": 1.0, "a": sigma0, "Gamma": 1.0, "z0": 2.0}, {"R": 1.0, "a": sigma0, "Gamma": 1.0, "z0": 2.0 + sep}]
    # Instants des vues 3D choisis d'après le modèle coaxial (fluide parfait) : départ, 1ʳᵉ traversée,
    # entre les deux traversées suivantes, fin du calcul.
    model = theory.CoaxialRingsModel([1.0, 1.0], [0.0, sep], 1.0, delta_eq).integrate(t_end, 1601)
    # Instants de traversée du modèle (model["Z"] : positions de forme (2, nt)).
    cr_model = _crossings(model["t"], model["Z"][0] - model["Z"][1])
    # round(..., 2) : instants arrondis au centième (atteints exactement par le solveur) ; valeurs de repli
    # si le modèle ne prédit pas assez de traversées sur la durée simulée.
    t_c1 = round(cr_model[0], 2) if cr_model else round(0.25 * t_end, 2)
    t_mid = round(0.5 * (cr_model[0] + cr_model[1]), 2) if len(cr_model) > 1 else round(0.6 * t_end, 2)
    iso_times = (0.0, t_c1, t_mid, t_end)
    print(f"Saute-mouton visqueux : 2 anneaux R = 1, σ0 = {sigma0}, Re_Γ = 3000, Δr = {dr}, t_end = {t_end}")
    # Simulation Navier–Stokes avec instantanés aux instants choisis, profils de Hovmöller et suivi des cœurs.
    res = ax2d.run_coaxial_rings(rings, Re=3000.0, t_end=t_end, dr=dr, snapshot_times=iso_times, verbose=True)
    # Séries du suivi individuel : t, Γ (nt, 2), R (nt, 2), Z (nt, 2), nombre de pics détectés.
    rg = res["rings"]
    print(f"   {res['grid']}  —  {res['wall_time']:.1f} s")
    # Fusion des cœurs : premier instant à partir duquel un seul pic de vorticité subsiste (3 diagnostics
    # consécutifs) ; les courbes de suivi sont tronquées à cet instant.
    single = rg["n_peaks"] < 2
    # next(générateur, None) : premier indice satisfaisant la condition, ou None s'il n'y en a aucun.
    i_merge = next((i for i in range(len(single) - 2) if single[i] and single[i + 1] and single[i + 2]), None)
    t_merge = float(rg["t"][i_merge]) if i_merge is not None else None
    # Masque des instants conservés (tous si aucune fusion ; np.ones_like(..., dtype=bool) : que des True).
    keep = rg["t"] < t_merge if t_merge is not None else np.ones_like(rg["t"], dtype=bool)
    # Dictionnaire instant → instantané, et centre des vues = centroïde de vorticité global (toujours défini,
    # y compris après la fusion des cœurs).
    snap_at = {round(s["t"], 6): s for s in res["snapshots"]}
    # Fonction anonyme (lambda) : Z global interpolé (np.interp) à l'instant tt.
    Zmid = lambda tt: float(np.interp(tt, res["diagnostics"]["t"], res["diagnostics"]["Z"]))
    # Instantanés des vues isosurfaces (ceux effectivement enregistrés).
    iso_snaps = [snap_at[round(tt, 6)] for tt in iso_times if round(tt, 6) in snap_at]
    viz.save_figure(v3.plot_isosurface_sequence(iso_snaps, [Zmid(s["t"]) for s in iso_snaps],
                                                title="Saute-mouton visqueux — isosurfaces Q reconstruites par révolution"),
                    os.path.join(p["figures"], "leapfrog_ns_isosurfaces.png"))
    # Trajectoires suivies des deux anneaux (jusqu'à la fusion) superposées aux diagrammes de Hovmöller ;
    # 'AB'[i] : lettre A pour i = 0, B pour i = 1.
    tracks = [(rg["t"][keep], rg["Z"][keep, i], f"anneau {'AB'[i]} (suivi)") for i in range(2)]
    merge_txt = f" ; fusion des cœurs à t ≈ {t_merge:.1f}" if t_merge is not None else ""
    viz.save_figure(viz.plot_hovmoller(res["profiles"], tracks=tracks,
                                       title="Saute-mouton visqueux — Hovmöller z–t : accélérations et ralentissements alternés"
                                             + merge_txt),
                    os.path.join(p["figures"], "leapfrog_ns_hovmoller.png"))
    # Cartes de dissipation au départ, à la première traversée et à la fin.
    diss_snaps = [snap_at[round(tt, 6)] for tt in (0.0, t_c1, t_end) if round(tt, 6) in snap_at]
    viz.save_figure(viz.plot_dissipation_map(diss_snaps, [Zmid(s["t"]) for s in diss_snaps], half_width=2.2,
                                             title="Saute-mouton visqueux — taux de dissipation Φ(r, z)"),
                    os.path.join(p["figures"], "leapfrog_ns_dissipation.png"))
    # Bilan temporel E_k, P_z, Ω et comparaison à la décroissance visqueuse théorique.
    viz.save_figure(viz.plot_energy_enstrophy_budget(res, title="Saute-mouton visqueux — bilan d'énergie et d'enstrophie"),
                    os.path.join(p["figures"], "leapfrog_ns_energy_budget.png"))
    # Rendu volumique de l'enstrophie (traversée d'un anneau par l'autre, puis état final).
    # Taille des images (pixels) et résolution de la grille 3D reconstruite.
    res_px = (300, 240) if quick else (420, 330)
    vol_res = 100 if quick else 150
    images, titles = [], []
    for tt in (t_c1, t_end):
        # Instant non enregistré (cas limite) : ignoré.
        if round(tt, 6) not in snap_at:
            continue
        # Volume 3D de ½ω² obtenu par révolution, centré sur les anneaux, puis lancer de rayons.
        vol, org, sp = v3.axisym_enstrophy_volume(snap_at[round(tt, 6)], Zmid(tt), half_width=1.8, r_max=1.8,
                                                  resolution=vol_res)
        images.append(v3.volume_render(vol, org, sp, elev=16, azim=-60, image_size=res_px))
        titles.append(f"t = {tt:.1f}")
    viz.save_figure(v3.plot_volume_renderings(images, titles, title="Saute-mouton visqueux — rendu volumique de l'enstrophie ½ω²"),
                    os.path.join(p["figures"], "leapfrog_ns_volume_rendering.png"))
    # Instants de traversée mesurés (avant la fusion) et synthèse chiffrée.
    cr_ns = _crossings(rg["t"][keep], rg["Z"][keep, 0] - rg["Z"][keep, 1])
    d = res["diagnostics"]
    # repr(grille) : description textuelle ; énergie perdue = 1 - E(t_end)/E(0) en pourcentage.
    summary["leapfrog_ns"] = {"grid": repr(res["grid"]), "wall_time_s": res["wall_time"], "crossings": cr_ns,
                              "crossings_inviscid_model": cr_model, "merger_time": t_merge,
                              "energy_loss_percent": float(100 * (1 - d["energy"][-1] / d["energy"][0])),
                              "max_divergence": float(d["max_div"].max())}
    # ", ".join(...) : liste des instants formatés à 2 décimales, séparés par des virgules.
    print(f"   traversées NS : {', '.join(f'{c:.2f}' for c in cr_ns)} (fluide parfait : "
          f"{', '.join(f'{c:.2f}' for c in cr_model)}){merge_txt} ; énergie perdue : "
          f"{summary['leapfrog_ns']['energy_loss_percent']:.1f} %")

    # --- 2. Même saute-mouton en fluide parfait : filaments 3D (δ équivalent) et modèle coaxial → espace des phases
    n_nodes = 96 if quick else 192
    # Deux anneaux-filaments régularisés par δ équivalent, intégrés en RK4 (pas 0.04, une sauvegarde sur 3).
    sim = vp.VortexFilamentSimulation(vp.leapfrog_scenario(delta=delta_eq, separation=sep, n_nodes=n_nodes))
    hist = sim.run(t_end, 0.04, save_every=3)
    # Historique converti en tableaux (t, rayon (nt, 2), position axiale (nt, 2), ...).
    h = hist.as_arrays()
    # Libellé de la courbe Navier–Stokes (mention de l'instant de fusion s'il existe).
    ns_label = "Navier–Stokes, Re_Γ = 3000" + (f" (jusqu'à la fusion, t ≈ {t_merge:.1f})" if t_merge is not None else "")
    # Trois trajectoires dans le plan (R₁ − R₂, z₁ − z₂) ; model["R"].T : transposée → forme (nt, 2).
    curves = [
        {"label": "filaments 3D (fluide parfait)", "t": h["t"], "R": h["radius"], "Z": h["axial"], "ls": "-"},
        {"label": "modèle coaxial régularisé", "t": model["t"], "R": model["R"].T, "Z": model["Z"].T, "ls": "--"},
        {"label": ns_label, "t": rg["t"][keep], "R": rg["R"][keep], "Z": rg["Z"][keep], "ls": "-."},
    ]
    viz.save_figure(viz.plot_phase_diagram(curves), os.path.join(p["figures"], "leapfrog_phase_diagram.png"))
    # Hovmöller des filaments : densité axiale de vorticité λ(z, t) (une ligne par sauvegarde).
    zg = np.linspace(h["axial"].min() - 1.2, h["axial"].max() + 1.2, 400)
    # Une densité λ(z) par sauvegarde (zip : nœuds et δ de chaque instant), empilées en tableau (nt, nz).
    dens = np.array([vp.vorticity_line_density(fr, hist.gammas, dl, zg) for fr, dl in zip(hist.frames, hist.deltas)])
    viz.save_figure(viz.plot_hovmoller_density(zg, h["t"], dens,
                                               tracks=[(h["t"], h["axial"][:, i], f"anneau {'AB'[i]}") for i in range(2)],
                                               title="Saute-mouton (filaments, fluide parfait) — Hovmöller z–t"),
                    os.path.join(p["figures"], "leapfrog_filaments_hovmoller.png"))
    summary["leapfrog_filaments"] = {"crossings": _crossings(h["t"], h["axial"][:, 0] - h["axial"][:, 1])}
    if smoke_video:
        # Option --smoke-video : animation du rendu volumique de l'enstrophie (une sauvegarde sur deux,
        # 12 images/s : ≈ 4 s de vidéo pour t_end = 12).
        path = v3.animate_smoke_rings(hist, os.path.join(p["animations"], "smoke_rings_leapfrog.mp4"),
                                      fps=12, stride=2)
        print(f"   animation fumée : {path}")
        summary["leapfrog_filaments"]["smoke_animation"] = path

    # --- 3. Tubes de vorticité : anneau à cœur épais (multi-filaments) et collision à volume de cœur conservé
    # Nombre de couronnes de filaments, nœuds par filament et durée (réduits en mode --quick).
    layers, n_nodes, t_thick = (1, 48, 2.0) if quick else (2, 64, 4.0)
    # Anneau à cœur gaussien (σ = 0.15) discrétisé en 1 + 6 (+ 12) filaments.
    sim = vp.VortexFilamentSimulation(vp.make_ring(1.0, 1.0, n_nodes, core_radius=0.15, n_core_layers=layers))
    ht = sim.run(t_thick, 0.04, save_every=5)
    # Trois instants affichés : début, milieu (division entière //) et fin.
    idx = [0, len(ht.frames) // 2, len(ht.frames) - 1]
    # Écorché de 110° face à la caméra : les sections des tubes montrent le cœur gaussien (filaments
    # centraux intenses et foncés, couronnes externes plus faibles et plus claires).
    viz.save_figure(v3.plot_vortex_tubes([ht.frames[i] for i in idx], ht.gammas, [ht.deltas[i] for i in idx],
                                         [ht.times[i] for i in idx], radius_scale=0.5, follow=True, window=2.8,
                                         cutaway=110.0,
                                         title="Anneau à cœur épais : tubes de vorticité en écorché (rayon 0,5 δ, couleur = Γ du filament)"),
                    os.path.join(p["figures"], "tubes_thick_ring.png"))
    n_nodes, t_col = (64, 8.0) if quick else (96, 12.0)
    # Collision frontale ; conserve_core_volume=True : δ²L constant (δ diminue quand le filament s'étire).
    sim = vp.VortexFilamentSimulation(vp.head_on_collision_scenario(delta=0.1, separation=3.0, n_nodes=n_nodes),
                                      conserve_core_volume=True)
    hc = sim.run(t_col, 0.04, save_every=5)
    # Quatre instants régulièrement répartis (np.linspace puis arrondi à l'entier le plus proche).
    idx = np.linspace(0, len(hc.frames) - 1, 4).round().astype(int)
    viz.save_figure(v3.plot_vortex_tubes([hc.frames[i] for i in idx], hc.gammas, [hc.deltas[i] for i in idx],
                                         [hc.times[i] for i in idx], radius_scale=1.0, ncols=4, elev=14,
                                         title="Collision frontale à volume de cœur conservé : les tubes s'amincissent (δ²L = cste)"),
                    os.path.join(p["figures"], "tubes_collision.png"))
    summary["tubes"] = {"thick_ring_filaments": len(ht.gammas), "collision_delta_final": float(min(hc.deltas[-1]))}

    # --- 4. Rendu volumique 3D des filaments : collision décalée (dynamique non axisymétrique)
    n_nodes, t_off = (64, 9.0) if quick else (96, 11.0)
    # Deux anneaux en collision avec un décalage latéral de 0.9 (reconnexion non axisymétrique).
    sim = vp.VortexFilamentSimulation(vp.offset_collision_scenario(delta=0.1, separation=3.0, offset=0.9, n_nodes=n_nodes))
    ho = sim.run(t_off, 0.04, save_every=5)
    # Résolution de la grille 3D de reconstruction de la vorticité (l'anneau étiré occupe une grande boîte :
    # 160 mailles gardent le pas h ≈ 0,06 proche de δ ; en deçà, l'anti-crénelage élargit le noyau à 2h).
    fres = 110 if quick else 160
    images, titles = [], []
    for i in np.linspace(0, len(ho.frames) - 1, 3).round().astype(int):
        # ½|ω|² reconstruit par « splatting » des segments, puis rendu volumique (densité 10 : fumée plus opaque).
        vol, org, sp = v3.filament_enstrophy_volume(ho.frames[i], ho.gammas, ho.deltas[i], resolution=fres)
        images.append(v3.volume_render(vol, org, sp, elev=15, azim=-35, image_size=res_px, density=10.0))
        titles.append(f"t = {ho.times[i]:.1f}")
    viz.save_figure(v3.plot_volume_renderings(images, titles,
                                              title="Collision décalée (filaments 3D) — rendu volumique de l'enstrophie ½|ω|²"),
                    os.path.join(p["figures"], "smoke_offset_collision.png"))
    return summary


# =============================================================================================
# Phase 3 — moteurs à vortex
# =============================================================================================
def run_engines(out, quick=False):
    _banner("PHASE 3 — Moteurs à vortex")
    p = _paths(out)
    summary = {}
    # Chambre d'exemple (paramètres par défaut) et sa distribution des temps de séjour.
    ch = ea.BidirectionalVortexChamber()
    rtd = ch.residence_time_distribution()
    # Trois trajectoires injectées à 0.75a, 0.85a, 0.95a (angles initiaux décalés pour la lisibilité).
    trajs = [ch.trajectory(f * ch.a, theta0=k * 2.1) for k, f in enumerate((0.75, 0.85, 0.95))]
    print(f"Chambre : a = {ch.a} m, L = {ch.L} m, U = {ch.U} m/s, Q = {ch.Q} m³/s  →  κ = {ch.kappa:.4f}")
    print(f"   temps de séjour moyen V/Q = {rtd['mean'] * 1e3:.2f} ms ; σ_θ² = {rtd['sigma_theta2']:.3f} "
          f"(≈ {rtd['n_tanks']:.2f} réacteurs agités en série)")
    for tr in trajs:
        print(f"   r0 = {tr['r'][0] / ch.a:.2f} a : séjour {tr['residence_time'] * 1e3:6.2f} ms, {tr['revolutions']:5.1f} tours, "
              f"parcours {tr['path_length']:.2f} m ({tr['path_ratio']:.1f} L), z_min = {tr['z_min'] / ch.L:.2f} L")
    print(f"   débit à travers le manteau : {ch.cross_mantle_flux_per_length():.4f} m²/s = Q/L (uniforme)")
    print(f"   diamètre de coupure centrifuge des gouttes : {ch.droplet_cutoff_diameter() * 1e6:.1f} µm")
    # Chute de pression entre la paroi et l'axe (r = 0).
    dp = ch.pressure_drop(np.array([0.0]))[0]
    print(f"   chute de pression paroi → axe : {dp / 1e5:.3f} bar ; nombre de swirl en sortie : {ch.exit_swirl_number():.2f}")
    # Rapport de flux « paroi froide » pour plusieurs Péclet radiaux (Pe = Pe_r/κ).
    cw = {pe_r: ea.cold_wall_heat_flux(pe_r / ch.kappa, ch.kappa)["ratio"] for pe_r in (1.0, 2.0, 5.0, 10.0, 20.0)}
    # ", ".join(...) assemble les textes séparés par des virgules.
    print("   paroi froide  Φ(Pe·κ) : " + ", ".join(f"{k:g} → {v:.3f}" for k, v in cw.items()))
    # Conversion d'une réaction d'ordre 1 pour 60 Damköhler répartis logarithmiquement.
    rc = ea.reactor_conversion(np.geomspace(0.1, 30, 60), ch)
    viz.save_figure(viz.plot_bidirectional_vortex(ch, trajs, rtd), os.path.join(p["figures"], "engine_bidirectional_vortex.png"))
    viz.save_figure(viz.plot_cold_wall(ch.kappa, ea.cold_wall_heat_flux), os.path.join(p["figures"], "engine_cold_wall.png"))
    viz.save_figure(viz.plot_reactor_conversion(rc), os.path.join(p["figures"], "engine_reactor_conversion.png"))
    summary["chamber"] = {"kappa": ch.kappa, "mean_residence_time_s": rtd["mean"], "sigma_theta2": rtd["sigma_theta2"],
                          "n_tanks": rtd["n_tanks"], "droplet_cutoff_um": ch.droplet_cutoff_diameter() * 1e6,
                          "pressure_drop_bar": dp / 1e5, "swirl_number": ch.exit_swirl_number(),
                          "cold_wall_ratio": {str(k): v for k, v in cw.items()},
                          "trajectories": [{"r0_over_a": float(tr["r"][0] / ch.a), "revolutions": float(tr["revolutions"]),
                                            "path_ratio": float(tr["path_ratio"]), "residence_ms": tr["residence_time"] * 1e3}
                                           for tr in trajs]}

    # Propulseur pulsé d'exemple (tuyère de 2 cm dans l'eau).
    thr = ea.VortexRingThruster()
    fs = thr.formation_number()
    print(f"\nPropulseur pulsé : D = {thr.D * 100:.0f} cm, eau ; F* = {fs:.2f}, β_max = {thr.beta_max}")
    # Propriétés de l'anneau de tête pour quelques courses L/D (U_j = 1 m/s).
    for ld in (1.0, 2.0, fs, 6.0):
        ring = thr.leading_ring(ld, 1.0)
        print(f"   L/D = {ld:4.2f} : β = {float(thr.overpressure_fraction(ld)):.3f}, Γ/(U_j D) = {ring['circulation'] / thr.D:.2f}, "
              f"U_anneau/U_j = {ring['U_self']:.3f}, a/R = {ring['core_ratio']:.2f}, V_b/V_slug = {ring['bubble_to_slug_volume']:.2f}, "
              f"C_A = {ring['added_mass']:.2f}, bouclage impulsion = {ring['impulse_closure']:.3f}")
    viz.save_figure(viz.plot_thruster_formation(thr), os.path.join(p["figures"], "engine_thruster_formation.png"))
    # Véhicule : vitesse de croisière et traînée (= poussée requise).
    U_inf = 0.5
    T_req = ea.body_drag(U_inf, diameter=0.1, cd=0.3)
    # Taille de la carte (n × n points).
    n = 14 if quick else 30
    m = thr.efficiency_map(np.linspace(0.5, 8.0, n), np.geomspace(0.5, 40.0, n), U_inf, T_req)
    m["formation_number"] = fs
    viz.save_figure(viz.plot_efficiency_map(m), os.path.join(p["figures"], "engine_efficiency_map.png"))
    # Positions du meilleur rendement et du meilleur gain (np.nanargmax ignore les NaN).
    i, j = np.unravel_index(np.nanargmax(m["efficiency"]), m["efficiency"].shape)
    ig, jg = np.unravel_index(np.nanargmax(m["gain"]), m["gain"].shape)
    print(f"   véhicule : U∞ = {U_inf} m/s, traînée = poussée = {T_req:.3f} N")
    print(f"   rendement maximal η = {m['efficiency'][i, j]:.3f} à L/D = {m['L_over_D'][i]:.2f}, f = {m['frequency'][j]:.1f} Hz "
          f"(U_j = {m['U_j'][i, j]:.2f} m/s, rapport cyclique {m['duty_cycle'][i, j]:.2f})")
    print(f"   gain maximal vs jet continu : {100 * (m['gain'][ig, jg] - 1):.1f} % à L/D = {m['L_over_D'][ig]:.2f}, "
          f"f = {m['frequency'][jg]:.1f} Hz (η = {m['efficiency'][ig, jg]:.3f})")
    summary["thruster"] = {"formation_number": fs, "required_thrust_N": T_req,
                           "best_efficiency": {"eta": float(m["efficiency"][i, j]), "L_over_D": float(m["L_over_D"][i]),
                                               "frequency_Hz": float(m["frequency"][j]), "U_j": float(m["U_j"][i, j]),
                                               "duty_cycle": float(m["duty_cycle"][i, j])},
                           "best_gain": {"gain_percent": float(100 * (m["gain"][ig, jg] - 1)), "L_over_D": float(m["L_over_D"][ig]),
                                         "frequency_Hz": float(m["frequency"][jg]), "eta": float(m["efficiency"][ig, jg])}}
    return summary


# =============================================================================================
def main(argv=None):
    # Sortie console en UTF-8 (sinon les caractères Γ, δ, σ... échouent dans la console Windows cp1252).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Définition des arguments de la ligne de commande.
    parser = argparse.ArgumentParser(description="Vortex toroïdaux & moteurs à vortex")
    # Argument positionnel obligatoire : la phase à exécuter.
    parser.add_argument("command", choices=["all", "theory", "particles", "axisym", "engines", "advanced"])
    # Options (action="store_true" : drapeau booléen sans valeur).
    parser.add_argument("--quick", action="store_true", help="résolutions réduites (tests rapides)")
    parser.add_argument("--out", default="outputs", help="dossier de sortie")
    parser.add_argument("--scenario", default="all", choices=["all", "leapfrog", "collision", "offset"])
    parser.add_argument("--animate", default="mp4", choices=["mp4", "gif", "none"])
    parser.add_argument("--re", type=float, default=1000.0, help="Re_Γ = Γ/ν du solveur axisymétrique")
    parser.add_argument("--t-end", type=float, default=None, help="durée de la simulation axisymétrique")
    parser.add_argument("--docs", action="store_true", help="recopie les figures du rapport dans docs/figures")
    parser.add_argument("--smoke-video", action="store_true",
                        help="phase advanced : animation MP4 du rendu volumique des anneaux (plusieurs minutes)")
    # parse_args : analyse sys.argv (ou la liste argv fournie) ; « --t-end » devient args.t_end.
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    summary_path = os.path.join(args.out, "summary.json")
    # Reprise de la synthèse existante (les phases non relancées sont conservées).
    summary = {}
    if os.path.exists(summary_path):
        # « with open(...) » ouvre le fichier et le referme automatiquement ; json.load le décode.
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
    # Exécution des phases demandées.
    if args.command in ("all", "theory"):
        summary["theory"] = run_theory(args.out, args.quick)
    if args.command in ("all", "particles"):
        # Fusion avec les résultats précédents : relancer un seul scénario (--scenario collision)
        # ne doit pas effacer ceux des autres scénarios dans summary.json.
        # dict.setdefault(clé, {}) renvoie l'entrée existante (ou en crée une vide) ; .update(...) la complète.
        summary.setdefault("particles", {}).update(run_particles(args.out, args.quick, args.scenario, args.animate))
    if args.command in ("all", "axisym"):
        summary["axisymmetric"] = run_axisym(args.out, args.quick, args.re, args.t_end)
    if args.command in ("all", "engines"):
        summary["engines"] = run_engines(args.out, args.quick)
    if args.command in ("all", "advanced"):
        summary["advanced"] = run_advanced(args.out, args.quick, args.smoke_video)
    # Écriture de la synthèse (indent=2 : lisible ; ensure_ascii=False : accents conservés ;
    # default=float : convertit les réels numpy restants).
    os.makedirs(args.out, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    # Option --docs : copie des figures du rapport dans docs/figures.
    if args.docs:
        dst = os.path.join("docs", "figures")
        os.makedirs(dst, exist_ok=True)
        for name in DOC_FIGURES:
            src = os.path.join(args.out, "figures", name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst, name))
        print(f"Figures du rapport recopiées dans {dst}")
    print(f"\nTerminé en {time.perf_counter() - t0:.1f} s — résultats dans « {args.out} » (synthèse : {summary_path})")


# Exécuté seulement quand le fichier est lancé directement (« python main.py ... »), pas à l'import.
if __name__ == "__main__":
    main()
