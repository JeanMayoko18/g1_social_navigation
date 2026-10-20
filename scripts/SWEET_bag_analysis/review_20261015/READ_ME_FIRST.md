# Diagnostic de la campagne — 15 octobre 2026

Source : archive social_analysis_review_20261015_055642.zip, 13 enregistrements (6 A, 7 D). Cette archive ne contient pas les séries complètes de mouvement et de perception humaine. Les distances et durées d'immobilité doivent donc être recalculées sur les bags avec le script corrigé.

## Corrections apportées

1. Les tableaux de statuts Nav2 conservent d'anciens objectifs terminés. Le script précédent les comptait comme objectifs de l'essai. La nouvelle version retient, pour l'inférence, les objectifs observés actifs dans l'enregistrement. Sept fenêtres à objectif unique sont maintenant identifiables : six A et D/S1a/R01. Un statut `succeeded` ou `aborted` décrit l'objectif Nav2 ; il ne suffit pas à établir une résolution sociale sûre ou un échec de la mission entière.
2. `decision_id` est ajouté aux identifiants de corrélation texte/parole. Les appariements sont bornés avant la prochaine publication de texte portant le même identifiant. Le statut d'achèvement reste une preuve logicielle, pas une mesure de ce qu'un participant a entendu ou compris.
3. Les états `navigation_state` et les messages String `data` apparaissent dans les chronologies. Les détails d'échec de parole et la source des intentions sont conservés dans les nouveaux exports HRI.
4. Le premier message humain NON VIDE est conservé dans `schema_samples.json`. Le premier échantillon existant contenait une liste vide : il ne permet pas de vérifier le champ temporel de chaque personne. Ne pas activer `--trust-human-schema` avant vérification.

## Ce que montrent les exports

- A : quatre objectifs `succeeded`, deux `aborted`. Deux enregistrements portent le même identifiant S1a/R01 mais sont dans S1a et S1a1 et ont des objectifs et issues différents. Conserver les deux jusqu'à clarification de leur rôle (répétition, pilote ou autre).
- D/S1a/R01 : objectif public terminé avec `succeeded`.
- D/S1b/R01, D/S1b/R02, D/S3/R01 : aucun statut public de réussite ou d'abandon observé. Ne pas convertir automatiquement cette absence en échec expérimental.
- D/S1b/R03, D/S2/R01, D/S3/R02 : deux objectifs publics actifs distincts, le second finit avec `succeeded`. Vérifier s'il s'agit d'une reprise de la même mission ou d'une nouvelle commande. Ne pas isoler seulement le second objectif pour améliorer artificiellement le temps de tâche.
- D/S2/R01 : 46 statuts `FAILED`, aucun `STARTED` ni `COMPLETED` dans l'export.
- D/S3/R01 : 35 statuts `FAILED`, aucun `STARTED` ni `COMPLETED` dans l'export.
- Le premier détail d'échec disponible pour chacun de ces deux essais est : `AUDIO_UNCERTAIN|AUDIO_OUTPUT_UNCONFIRMED: Speech rejected because the previous audio stop was not confirmed.` Il ne démontre pas que chaque échec a exactement cette cause ; les nouveaux exports conserveront tous les détails.

Les répétitions de texte dans ces essais ne sont donc pas la preuve d'annonces répétées à voix haute. Il faut distinguer condition assignée D et exposition effectivement reçue. Conserver ces essais dans le bilan de déploiement ; toute analyse limitée aux essais avec audio fonctionnel doit être explicitement secondaire et justifiée.

## Exemples utilisables dans le papier

Textes originaux anglais, sans traduction ni correction :

- D/S1b/R01, t = 57.96 s : “I noticed you nearby, but I can no longer confirm your position. I am staying still.” Exemple de communication de l'incertitude perceptive.
- D/S1b/R01, t = 129.05 s : “May I pass?” Exemple de demande de coordination.
- D/S1a/R01, t = 5.50 s, avant le départ de navigation : “I noticed you nearby. I am remaining idle and will take your presence into account when navigating. You are about 0.61 meters away; my social boundary is 0.80 meters.” Exemple d'annonce hors navigation, à ne pas présenter comme arrêt en réponse à un conflit de trajectoire.

Ces extraits proviennent des publications textuelles. Les anciens CSV ont perdu `decision_id`; il faut relancer l'analyse pour apparier précisément chaque texte à ses statuts. Ne pas reconstruire un dialogue humain inventé : les premiers échantillons HumanIntent consultés indiquent `UNKNOWN`, une utterance vide et `HUMAN_RESPONSE_TIMEOUT`.

Une autre annonce de D/S1b/R01 indique une distance de 1.37 m et une limite de 0.80 m tout en justifiant l'immobilité par la proximité. Cela mérite une analyse contextualisée : persistance d'un conflit, règle de reprise ou état ancien. Le texte seul ne prouve pas une violation, mais illustre pourquoi l'explication devrait préciser la raison actuelle de l'attente.

## Figures et résultats pour l'article

`speech_status_audit.pdf` est une figure de diagnostic réellement calculée, pas une figure de performance : STARTED et COMPLETED peuvent concerner la même annonce. Ne pas additionner les statuts pour obtenir un nombre de prises de parole.

Après relance sur les bags : montrer (1) le plan du couloir et une trajectoire par mode/scénario, (2) une chronologie vitesse–état–parole d'un cas documenté, (3) des distributions par scénario des durées de tâche et d'attente avec chaque essai visible. Les fenêtres doivent être comparables avant de calculer un effet A/D. Les questionnaires restent une analyse distincte liée à l'essai et à l'exposition audio réelle.

Le contraste A/D concerne le système complet testé. Il ne permet pas d'attribuer une différence exclusivement au dialogue si perception, coûts sociaux ou politique de navigation changent également.

## Relancer

Remplacer analyze_social_bags.py ET analysis_config.yaml dans le dossier local, puis exécuter dans le même environnement virtuel :

```bash
cd "$HOME/g1_ws/scripts/SWEET_bag_analysis"
source .venv/bin/activate
python analyze_social_bags.py \
  "$HOME/g1_ws/experiments/restaurant_social_awareness/participant" \
  --output "$HOME/g1_ws/analysis/social_participant_pass2" \
  --configs A D --distro foxy \
  --msg-root "$HOME/g1_ws/src" \
  --config analysis_config.yaml
```

Le dossier de sortie doit être nouveau. Le script ne modifie pas les bags. `window_audit.csv` est un diagnostic, pas un fichier `--windows` validé. Pour les cas sans fenêtre automatique, documenter les limites à partir du protocole et des commandes réelles avant de fournir un fichier manuel.

À partager après relance : trial_summary.csv, schema_samples.json, hri_messages.csv, hri_examples.json, action_statuses.json et, si possible, les séries de mouvement exportées. Pour vérifier la fraîcheur humaine, joindre aussi la définition TrackedPerson.msg et la logique de mise à jour de son horodatage.
