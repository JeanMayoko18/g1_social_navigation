# Executable Protocol — "Stationary Person" Scenario (Nav-A vs Nav-C pilot)

Goal: obtain a first quantitative and subjective dataset comparing Nav-A (LiDAR only) and Nav-C (social layer enabled) on the simplest encounter scenario, to seed Section 5 of the paper.

The study has **two tiers**:
- **Tier 1 — Direct participants** (N = 5–7): live, in-person trials on the physical robot. Provides the geometric/navigation metrics and a first-person comfort rating.
- **Tier 2 — Video-based external raters** (N = 15): watch recorded clips from Tier 1 and rate them as observers. Provides a better-powered subjective/perceptual measure without needing more physical sessions.

Present these as **two complementary, non-pooled sources of evidence** in the paper — do not merge Tier 1 and Tier 2 scores into a single sample.

---

## 1. Before the session

### 1.1 Equipment checklist
- [ ] Unitree G1 charged, adapted RL walking policy loaded and sanity-checked (straight-line walk, no obstacle)
- [ ] Map already built, AMCL localized and stable (check pose covariance before every session)
- [ ] Camera(s) for YOLO calibrated, FOV covering the planned encounter zone
- [ ] **External fixed camera** for video capture (constant angle, framing robot + person + path) — feeds Tier 2
- [ ] Locomotion bridge Nav2→RL tested: measure the gap between commanded velocity (`u_nav`) and executed velocity (odometry) on a reference run with no person present
- [ ] ROS clock synchronized between external PC and robot (NTP/`chrony`)
- [ ] Test area cleared, floor markers for the three waypoints (start, person position, service goal)
- [ ] Printed/digital questionnaire ready for Tier 1 (Section 4) and an online form ready for Tier 2 (Section 8)
- [ ] **Consent forms** for the standing participants (image release, since they will be recorded and the footage shown to Tier-2 raters) and for Tier-2 raters (informed consent for the rating task) — see Section 9

### 1.2 Parameters to fix and document
| Parameter | Pilot value | Adjust after pilot? |
|---|---|---|
| `d_social` (social-zone threshold) | 1.2 m | |
| `d_intr` (intrusion-event threshold) | 0.8 m | |
| `r_excl` (lethal exclusion radius) | 0.4 m | |
| Robot walking speed | 0.3–0.4 m/s (cf. Ginés et al. 2019) | |
| Trial timeout | 60 s | |
| Goal position/orientation tolerance | 0.1 m / 10° | |

### 1.3 Pilot phase (before the campaign)
Run 3–5 uncounted trials per configuration (Nav-A, Nav-C) with a non-participant stand-in to:
1. confirm the robot reliably reaches the goal in Nav-A;
2. confirm the social zone in Nav-C actually deflects the trajectory without oscillation or stalling near the goal (verify in practice the near-goal rotation-critic relaxation already described in Section 3.1);
3. retune `d_social` / `d_intr` / `r_excl` if the robot is either overly cautious (excessive detour) or passes too close (Psi > 0);
4. check camera framing/exposure for the Tier-2 recordings.

---

## 2. Tier 1 — Direct-participant trial procedure

### 2.1 Design: repeated trials per participant, aggregated before testing
With only 5–7 participants, a single Nav-A/Nav-C trial per person gives too few paired observations to say much. Instead, **each participant performs multiple trials per condition**, and you average within participant before running the statistical test — this reduces per-trial measurement noise without inflating the sample size artificially (never treat the repeated trials as independent observations; that would be pseudo-replication).

Recommended: **3 trials per condition per participant** → 6 trials per participant → 30–42 physical trials total for 5–7 participants.

### 2.2 Procedure (per trial)
1. **Reset**: robot at the start waypoint, fixed orientation, AMCL re-localized if needed.
2. **Place the standing participant** at the marked point, stationary, beside the direct start→goal path (identical position in both configurations).
3. **Synchronized start**: begin rosbag *and* external-camera recording *before* sending the Nav2 goal.
4. **Send the service goal**.
5. **Let the robot navigate** until: success (goal reached within tolerances), failure (timeout, manual intervention, contact), or repeated recovery (Nav2 recovery behaviours triggered > 2 times → logged as failure).
6. **Stop recording**, label both the bag and the video clip `p<participant_id>_<config>_<trial_number>_<timestamp>`.
7. **Administer the Tier-1 questionnaire immediately** (Section 4) — once per trial, while it is still fresh.
8. **Minimum 30 s pause** before the next trial (let human tracks reset).

### 2.3 Trial order (counterbalanced)
Randomize and balance the order of Nav-A/Nav-C blocks across participants (e.g. participants 1, 3, 5, 7 start with Nav-A; 2, 4, 6 start with Nav-C) to control for order effects (robot battery drift, localization drift within a session, participant habituation).

---

## 3. Measurement grid (Tier 1, fill in per trial)

| Participant | Trial # | Config | Success (Y/N) | `d_t` traveled (m) | `τ` time-to-goal (s) | `d_min` to person (m) | `T_intr` (s) | `Psi` (%) | Recoveries triggered | Manual intervention (Y/N) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 1 | | | | | | | | | | |
| P1 | 2 | | | | | | | | | | |
| P1 | 3 | | | | | | | | | | |
| ... | | | | | | | | | | | |

- `d_min` and `T_intr` are extracted from the rosbag afterwards — no need to measure them live.
- `Psi (%) = T_intr / τ × 100`, computed in post-processing.
- A "recovery" = any Nav2 recovery behaviour trigger (in-place rotation, costmap clearing, etc.), identified from the logs.
- Before analysis, **average the 3 trials per participant per condition** into one value per participant × condition × metric.

---

## 4. Tier 1 questionnaire (first-person, standing participant)

**When**: immediately after *each individual trial* (step 7 above) — three ratings per condition per participant, later averaged per participant × condition like the geometric metrics.

**Format**: 5-point Likert scale (1 = strongly disagree, 5 = strongly agree).

1. The robot kept an appropriate distance from me.
2. I felt safe during the robot's approach.
3. The robot's path change (if any) felt natural.
4. I was surprised by the robot's behaviour. *(reverse-scored)*
5. Overall, I was comfortable with how the robot passed by me.

Keep the item set **identical** across Nav-A and Nav-C trials.

---

## 5. Topics to record (rosbag)

- `/amcl_pose`, `/tf`, `/tf_static`
- `/cmd_vel` (Nav2 command before the bridge) and the command sent to the RL policy after the bridge (to quantify the `u_nav` vs `u_RL` gap)
- `/plan` (global path, recomputed at each replan)
- Human detection/tracking topic(s) — ideally also published under `/humans/persons/tracked` if you adopt the ROS4HRI format alongside your internal pipeline
- Local `/costmap/costmap` (to visualize the Gaussian zone afterwards)
- Social decision manager events (intrusion, replanning requests)
- Generated LLM messages (text + timestamp) — keep them for protocol consistency even though explanation quality is not the focus of this scenario

---

## 6. Tier 1 analysis

1. Average the 3 trials per participant × condition for each metric (geometric + questionnaire).
2. Check normality of the paired differences (Nav-C − Nav-A) per metric — with N=5–7, **do not rely on a normality test to justify a t-test**; default to the **Wilcoxon signed-rank test** unless you have strong reason to assume normality.
3. Report the result honestly as **exploratory/pilot-scale**: show individual per-participant differences (a small paired dot/line plot is more transparent than a bar chart of means at this N), alongside the test statistic and p-value.
4. Be explicit in the paper that at N=5–7 only large, consistent effects will reach p<0.05 (e.g. at N=6, all 6 differences must point the same way to reach p≈0.03 two-tailed); frame this tier as motivating rather than confirmatory, and note scaling to more participants as future work — this pre-empts the most likely reviewer objection.
5. Check that any reduction in `Psi` under Nav-C is not accompanied by a disproportionate drop in success rate.

---

## 7. G1-specific watch points

- Bridge command-to-execution drift can by itself explain part of a reduced `d_min` or noisy trajectory: document it separately (idle calibration, Section 1.1) so it isn't wrongly attributed to the social layer.
- The near-goal rotation critics (Section 3.1) can interact with the Gaussian zone when the person stands close to the goal — check this case in the pilot; it can create an artificial deadlock.

---

## 8. Tier 2 — Video-based external rating study (N = 15)

### 8.1 Clip selection
From the Tier-1 recordings, select **one representative clip per condition** (Nav-A, Nav-C) — e.g. a trial close to the median `d_min`/`Psi` for that condition, not an outlier. That gives **2 clips** for the stationary-person scenario (extend to 6 if you later add the frontal-encounter and crossing scenarios: 3 scenarios × 2 configs).

### 8.2 Design: within-subjects, counterbalanced
Each of the 15 raters watches **both** clips (Nav-A and Nav-C, or all 6 if multi-scenario), in individually randomized order, and rates each one separately with the questionnaire below immediately after watching it — do not let raters compare clips side by side before rating, to avoid demand effects.

### 8.3 Observer questionnaire (third-person, reworded from Tier 1)
5-point Likert scale (1 = strongly disagree, 5 = strongly agree), same structure as Tier 1 but rephrased for an outside observer judging the scene:

1. The robot kept an appropriate distance from the person.
2. If I were that person, I would have felt safe during the robot's approach.
3. The robot's path change (if any) looked natural.
4. The robot's behaviour looked surprising or erratic. *(reverse-scored)*
5. Overall, the robot's passing behaviour looked comfortable and appropriate.

Keep item wording **identical** across the Nav-A and Nav-C clips shown to a given rater.

### 8.4 Distribution
A simple online form (Google Forms / Qualtrics / LimeSurvey) embedding the two (or six) clips in randomized order per respondent, with the 5 items after each clip, is sufficient — no in-person session required for this tier.

### 8.5 Analysis
1. For each rater, one score per item per condition (average across scenarios if you show 6 clips per rater).
2. Paired test across the 15 raters (Nav-C − Nav-A per item): **paired t-test** if the differences look reasonably normal at N=15 (check with a QQ-plot or Shapiro-Wilk), otherwise **Wilcoxon signed-rank**.
3. Report this tier as the better-powered subjective evidence, complementing — not replacing — the small physical-participant sample in Tier 1.

---

## 9. Ethics and consent notes

- **Standing participants (Tier 1)**: need informed consent covering (a) physical participation with the robot, and (b) an **image/video release**, since footage will be shown to third parties (Tier-2 raters).
- **External raters (Tier 2)**: need informed consent for the rating task (standard for online survey-based HRI perception studies).
- Both should go through your institution's ethics review (Universidad de León) before data collection — reviewers commonly check for this in papers involving human subjects, even at small N.
