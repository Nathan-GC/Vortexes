"""
appli — Paquet de l'application web Streamlit (point d'entrée : app.py à la racine du projet).

Organisation :
* commun.py        outils partagés (verrou de tracé, images, métriques, courbes, widgets) ;
* calibration.py   durées estimées, recalibrées sur les calculs déjà faits sur ce PC ;
* stockage.py      historique des calculs (comparaison) et mémoire des résultats déjà calculés ;
* taches.py        calcul en arrière-plan, barre de progression et annulation ;
* vues3d.py        vues 3D et animations interactives (Plotly) ;
* glossaire.py     définitions des termes techniques ;
* documentation.py onglet de documentation (Markdown, formules, ancres, extraits de sections) ;
* accueil.py       page d'accueil ;
* pedagogie.py     scénarios guidés, prédictions, théorie associée, limites, unités physiques ;
* resultats.py     affichage des résultats, comparaison, convergence, export ;
* simulation.py    onglet de simulation ;
* volets/          un module par volet simulable (anneau, saute-mouton, filaments, moteurs, théorie).
"""
