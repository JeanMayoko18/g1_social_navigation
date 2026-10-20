# Mise à jour du 15 octobre 2026

Lire `review_20261015/READ_ME_FIRST.md` pour le diagnostic de vos 13 enregistrements et la commande de relance. Les métriques de mouvement de la première passe doivent être recalculées après correction des fenêtres.

# Analyse des bags de navigation sociale — SWEET / Unitree G1

Outil autonome, en lecture seule, pour votre campagne `restaurant_social_awareness`.
Il ne lance aucun nœud et n'envoie aucune commande au robot. Code, commentaires et messages en anglais ; guide en français.

## 1. Installation sur le PC d'analyse

Python **3.10 ou plus récent**, conseillé sur votre PC Humble. Aucun `source` ROS ni installation `rclpy` nécessaire. Les bags Foxy restent lisibles sur ce PC.

Décompresser ce dossier sous `~/g1_ws/scripts/SWEET_bag_analysis`, puis :

```bash
cd "$HOME/g1_ws/scripts/SWEET_bag_analysis"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

L'analyse emploie `rosbags` pour lire SQLite3/MCAP. Pour les anciens bags sans définitions embarquées, `--msg-root` charge les `.msg`/`.idl` du workspace. On peut répéter ce paramètre pour une dépendance externe. Employer les définitions correspondant à la révision qui a enregistré les bags.

## 2. Première analyse : campagne complète, modes A et D

```bash
cd "$HOME/g1_ws/scripts/SWEET_bag_analysis"
source .venv/bin/activate
python analyze_social_bags.py \
  "$HOME/g1_ws/experiments/restaurant_social_awareness" \
  --output "$HOME/g1_ws/analysis/social_pass1" \
  --configs A D \
  --distro foxy \
  --msg-root "$HOME/g1_ws/src" \
  --config analysis_config.yaml
```

Retirer `--configs A D` pour traiter aussi C. Le script conserve `technical` et `participant` dans des groupes distincts. Il n'infère pas que le mode C a effectivement été utilisé à partir du seul script d'enregistrement.

Pour un seul essai, remplacer le premier chemin par son dossier `navigation` contenant `metadata.yaml`. Les morceaux `.db3` d'un bag découpé restent ensemble. Un `.db3` isolé n'est pas accepté ; un fichier MCAP autonome l'est.

Le dossier de sortie doit être nouveau : le script refuse de remplacer une analyse existante. Code de sortie 2 si au moins un bag échoue ; les autres restent analysés.

## 3. Vérifier les messages personnalisés

Ouvrir :

- `trials/<essai>/topics_inventory.csv` : types, messages enregistrés et messages décodés ;
- `trials/<essai>/decode_errors.json` : définitions manquantes et erreurs de mapping ;
- `trials/<essai>/schema_samples.json` : un message réel par topic sélectionné ;
- `analysis_config.json` : paramètres effectifs, arguments, avertissements de types.

Adapter `analysis_config.yaml` aux champs réels. Exemple UNIQUEMENT :

```yaml
human_array: persons
human_position: position
human_id: track_id
human_frame: header.frame_id
human_stamp: header.stamp
```

Si votre position se trouve dans `pose.pose.position`, remplacer uniquement `human_position`. Le nom `persons` n'est pas supposé universel. Les mappings humains erronés sont signalés, pas remplacés par de fausses données.

**Important pour les tracks retenus :** `human_stamp` doit correspondre à la dernière vraie observation individuelle, pas à la republication d'une ancienne position. Si nécessaire, ajouter un filtre explicite, par exemple :

```yaml
human_valid_field: observed_now
human_valid_values: [true]
```

Employer les vrais noms et codes de votre message. Pour les statuts de parole numériques, indiquer leurs valeurs exactes dans `speech_started_values` et `speech_finished_values`. Une complétion de lecture exige un identifiant de message commun et un statut reconnu. Ne pas utiliser un ID de personne comme ID d'énoncé.

Le premier passage exporte/trace les positions interprétables, mais **ne produit pas de statistiques humaines validées**. Après vérification des champs et de leur sens, relancer avec `--trust-human-schema`. Ce drapeau est votre confirmation du mapping, pas une validation scientifique automatique.

```bash
python analyze_social_bags.py \
  "$HOME/g1_ws/experiments/restaurant_social_awareness" \
  --output "$HOME/g1_ws/analysis/social_pass2" \
  --configs A D --distro foxy \
  --msg-root "$HOME/g1_ws/src" \
  --config analysis_config.yaml \
  --trust-human-schema
```

Les distances sont calculées seulement pour une position dans un repère de base explicitement reconnu, ou dans le même repère que l'odométrie avec une synchronisation suffisante. **Aucune transformation TF arbitraire n'est inventée.** Pour des positions en repère caméra/map différent, choisir le topic déjà transformé correspondant ou compléter l'outil avec vos transformations vérifiées. Les timestamps de réception du bag sont utilisés pour les graphiques ; le timestamp individuel sert au contrôle de fraîcheur. Une incompatibilité d'horloges doit être résolue avant de publier les distances.

## 4. Fenêtres de mission : distinguer navigation et durée du bag

Par défaut, une fenêtre peut être inférée uniquement lorsqu'un **seul UUID de goal public** présente un état actif suivi de `succeeded` ou `aborted`. L'origine est le premier état actif enregistré, pas une acceptation mesurée au serveur. Le résultat s'appelle `nav2_succeeded`/`nav2_aborted`, pas « réussite sans intervention ».

En cas de plusieurs goals, de cancellation/reprise, de données insuffisantes ou d'absence de statut, l'outil conserve `recording_only`. Il ne transforme pas une annulation sociale en échec et ne baptise pas la durée du bag « temps pour atteindre le goal ».

Ouvrir `windows_to_review.csv`, copier ce fichier ailleurs et remplir :

- `start_s`, `end_s` : secondes depuis le début du bag, fenêtre complète de mission ;
- `outcome` : votre classification documentée, par exemple `success`, `timeout`, `intervention`, `aborted` ;
- `participant_id` : code pseudonyme, si connu ;
- `manual_intervention` : renseignement vérifié.

Les colonnes `suggested_*` ne sont jamais prises pour une confirmation. Les fichiers d'entrée ne sont pas modifiés.

```bash
python analyze_social_bags.py \
  "$HOME/g1_ws/experiments/restaurant_social_awareness" \
  --output "$HOME/g1_ws/analysis/social_reviewed" \
  --configs A D --distro foxy \
  --msg-root "$HOME/g1_ws/src" \
  --config analysis_config.yaml \
  --windows "$HOME/g1_ws/analysis/windows_reviewed.csv" \
  --trust-human-schema
```

Avant analyse finale, fixer les règles de succès, timeout et intervention ; ne pas choisir les frontières pour améliorer artificiellement une condition.

## 5. Ce qui est produit

### Par campagne

| Fichier | Contenu |
|---|---|
| `REPORT.md` | Index des essais et limites d'interprétation |
| `trial_summary.csv` | Une ligne par essai, métriques, couverture et source de fenêtre |
| `group_summary.csv` | Effectifs, médianes et quartiles par type/scénario/configuration/source de fenêtre |
| `comparison_*.png` et `.pdf` | Comparaisons : points individuels, médiane ; pas de fausse précision statistique |
| `navigation_summary.tex` | Tableau LaTeX descriptif |
| `windows_to_review.csv` | Aide à l'annotation des fenêtres de mission |

### Par essai

| Fichier | Contenu |
|---|---|
| `trajectory.png/pdf` | Trajectoire estimée, interrompue aux trous/sauts détectés |
| `motion_distance.png/pdf` | Vitesse mesurée, commandes disponibles, rotation, observations humaines |
| `event_timeline.png/pdf` | Changements d'état enregistrés et repères des messages M1, M2… |
| `motion.csv` | Odométrie décodée dans la fenêtre |
| `stop_intervals.csv` | Intervalles d'immobilité mesurée selon les seuils |
| `human_distances.csv` | Distances estimées, âge d'observation, base de calcul et éligibilité |
| `hri_messages.csv` | Tous les messages HRI reconnus, avec topic, rôle, texte, ID et statut |
| `hri_examples.json` | Exemples de texte robot et échanges proches dans le temps |
| `hri_exchange_01.png/pdf` | Figure de quelques échanges textuels réels, avec rôles et temps |
| `hri_examples.tex` | Tableau de citations exactes prêt à réviser pour le papier |
| `event_messages.jsonl` | Messages sociaux/HRI complets désérialisés, sans perte des champs personnalisés |
| `schema_samples.json` | Échantillons pour ajuster les mappings |
| `action_statuses.json` | Historique des statuts avec UUID |
| `sampled_plans.json` | Plans échantillonnés pour inspection, pas des replans comptés |
| `report.md` | Synthèse locale, messages et figures |

Les noms des sous-dossiers incluent un hash du chemin pour éviter de confondre deux essais de même nom.

## 6. Définitions et limites des métriques

- `path_length_observed_m` : somme des pas d'odométrie dans les intervalles exploitables. Les changements de repère, trous et vitesses impossibles sont exclus et comptés. Si `pose_time_coverage` est faible, ce n'est pas une longueur totale fiable. Le bruit d'odométrie peut gonfler la somme ; le seuil `minimum_pose_step_m` vaut zéro par défaut et doit être justifié s'il est modifié. Aucune correction de drift n'est inventée.
- `stationary_time_observed_s` : durée des séquences où vitesse linéaire ET rotation restent sous les seuils pendant la durée minimale. **Ce n'est pas nécessairement une attente sociale** : départ/arrivée, arrêt technique et parole peuvent aussi produire une immobilité.
- `motion_time_coverage` / `pose_time_coverage` : durée des intervalles retenus / durée de la fenêtre. Zéro couverture indique absence de preuve, pas absence de mouvement.
- `human_min_distance_estimate_m` : distance à l'origine du repère robot ou de l'odométrie, **pas distance libre entre les corps** et pas vérité terrain indépendante.
- `intrusion_time_observed_s` : durée d'intervalles observés dont les deux extrémités sont sous le seuil. Elle est indisponible si `social_distance_m` est null. Il n'y a ni remplissage des trous ni extrapolation quand une personne disparaît. Aux seuils, cette discrétisation peut sous-estimer la durée ; préciser la fréquence et la couverture.
- `human_time_coverage` concerne les observations humaines éligibles, pas une certification que la scène entière a été observée. Une absence de personne détectée n'est pas comptée comme clearance.
- Les minima ne sont pas comparables sans discuter la couverture : la condition avec plus d'observations peut aussi révéler davantage de petits écarts.
- `manual_success_rate` : fraction parmi les issues manuellement classées `success`, `timeout`, `intervention`, `aborted` ou `failure`. Les issues inconnues restent hors de ce dénominateur et visibles dans les effectifs ; ne pas les supprimer du compte rendu.
- `robot_text_messages` : nombre de publications textuelles reconnues, avec déduplication prudente des répétitions consécutives portant le même ID sur le même topic. Plusieurs topics peuvent publier le même énoncé : ce n'est pas automatiquement un nombre d'énoncés distincts. Champ vide si aucun topic textuel décodé.
- Un `speech_status` de fin indique une complétion **logicielle**. Une vidéo/audio est nécessaire pour confirmer audibilité et personne destinataire.
- Le délai texte → début vocal est calculé seulement pour les exemples dont l'ID et le code de statut concordent. Ce n'est pas la latence de génération LLM.
- Une transcription peut être erronée ; `closed_captions` reste de locuteur non vérifié. `human_intent` est une interprétation, pas une citation humaine.
- Une publication périodique de `/plan` n'est pas comptée comme une replanification.
- Aucun score de confort, trust, compréhension, violation de règle ou succès social n'est automatiquement déduit de ces logs.
- Les graphiques de groupe n'effectuent pas de test de significativité et ne traitent pas plusieurs essais d'une personne comme plusieurs participants. Les groupes techniques et participants sont séparés.

## 7. Exemples HRI pour l'article

Le script choisit chronologiquement le premier message par contexte enregistré, jusqu'à `max_examples`. C'est une présélection reproductible, pas une sélection des « meilleurs succès ». La fenêtre d'échange inclut les transcriptions, intentions et statuts proches. Le rapprochement temporel n'établit pas une causalité.

Avant publication : vérifier le texte exact, l'ID d'énoncé, l'état réel du robot, les répétitions et les éventuelles informations personnelles. Conserver l'espagnol original et ajouter manuellement une traduction anglaise identifiée. Montrer un échec ou une limite si pertinent. La latence et la réussite ne sont pas inventées.

## 8. Illustrations pour le short paper

Quatre éléments suffisent :

1. **Dispositif expérimental** : photo réelle et plan du couloir avec largeur, départ, goal, repères humains. À fournir séparément ; le recorder ne collecte pas systématiquement carte et images. Le script ne dessine pas un faux restaurant.
2. **Comportement spatial** : sélectionner des trajectoires A/D correspondant au même scénario et aux conditions vérifiées. Ne pas superposer deux repères odom ayant des origines différentes sans recalage documenté. Les figures individuelles restent utilisables côte à côte.
3. **Résultats quantitatifs** : figures `comparison_*` et tableau de synthèse après revue des fenêtres/couvertures.
4. **Interaction** : chronologie d'un essai + 2–4 messages exacts tirés de `hri_examples.tex`.

Le fichier `paper_figures.tex` fournit les blocs Overleaf. Les PDF des figures sont vectoriels. Les graphiques sont des résultats des logs ; l'environnement et les conditions doivent être décrits dans Experimental Setup.

## 9. Vérifications réalisées sur l'outil

Tests automatiques : trou d'observation, cancellation/reprise avec plusieurs UUID, inférence d'un goal unique, refus de métriques humaines non confirmées, absence de preuve de parole à partir du texte seul. Un test de bout en bout génère deux bags synthétiques A/D et vérifie lecture/figures/tableaux/messages. **Ces données synthétiques ne sont pas des résultats de votre robot et ne sont pas livrées comme figures du papier.**

```bash
python -m unittest discover -s tests -v
python tests/make_synthetic_bags.py /tmp/sweet_synthetic_bags
python analyze_social_bags.py /tmp/sweet_synthetic_bags \
  --output /tmp/sweet_synthetic_analysis \
  --msg-root /tmp/sweet_synthetic_bags/definitions \
  --trust-human-schema
```

Ne pas réutiliser des chemins de test existants. L'outil a été testé ici avec Python 3.12 et rosbags 0.11.7 ; il reste à vérifier les schémas personnalisés de vos bags réels.

## 10. Retour utile pour adapter exactement à vos messages

Après le premier passage, transmettre `trial_summary.csv`, les `decode_errors.json` et `schema_samples.json` d'un A et d'un D. Ajouter `hri_messages.csv` et `event_messages.jsonl` pour rédiger les exemples à partir des données. Si l'odométrie choisie est absente, identifier le topic effectivement utilisé et ajuster `odom_topic` ; ne pas mélanger automatiquement plusieurs odométries.

Sources techniques : documentation officielle rosbags, https://ternaris.gitlab.io/rosbags/topics/highlevel.html et https://ternaris.gitlab.io/rosbags/examples/register_types.html.
