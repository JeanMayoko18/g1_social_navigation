# Social navigation

Start with a working [mapped-navigation session](navigation.md). The [system launcher](../src/g1_social_bringup/launch/system.launch.py) is the entry point for integrated social bringup. Its [launch directory](../src/g1_social_bringup/launch/) contains separate entry points for image ingress, perception, vision inference, social decisions, speech, navigation and visualization.

| Question during an encounter | Where to look |
| --- | --- |
| Is the same person tracked across frames, and is depth available? | [Person perception](../src/g1_social_perception/) |
| Who decides when to hold or resume the goal? | [Collaboration manager](../src/g1_robot_collaboration/) |
| How are speech and explanations produced? | [HRI](../src/g1_hri/) and [explainability](../src/g1_robot_explainability/) |
| Where are directional costmap constraints implemented? | [Social costmap](../src/g1_social_costmap/) |

## Observe one encounter

For an announcement-only trial, inspect the sequence from detection to stop, announcement and clearance or replanning. For a negotiated trial, also record the person’s response, permission decision and any directional constraint before resumption. Record the selected profile and interaction setting with the trial.

With navigation idle, observe the intrusion announcement separately. Assigning a goal changes the navigation context even if the same person remains visible. During a directional trial, distinguish a confirmed human preference from an operational costmap mask. The intended fallback when the mask is unavailable is autonomous Nav2 route selection after the other resume conditions are satisfied. Verify that transition in the recorded run.

Use the [social trial recording script](../scripts/record_social_trial.sh) and the [stationary-person protocol](../protocol_stationary_person.md) to reproduce an encounter.
