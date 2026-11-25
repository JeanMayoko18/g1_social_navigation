"""
Shared mathematical model: from perception to social interpretation.

Purpose:
    Keep equations in one place so that ROS nodes use the same mathematical
    definitions. Nodes handle messages, synchronization and publication;
    this package handles calculations and their validity constraints.

Module map:
    types.py
        Mathematical inputs and results:
        source detections, enriched detections, retained track features,
        cue scores and association weights.

    validation.py
        Shared mathematical constraints:
        finite values, positive scales, bounded qualities, integer counts
        and normalized distributions.

    temporal.py
        Source-time differences, observation age and feature freshness.
        A retained feature keeps its own last measurement timestamp.

    geometry.py
        Bounding-box geometry, intersection over union, camera projection,
        metric extents and rigid coordinate transformations.

    appearance.py
        Upper/lower body regions, normalized appearance histograms
        and histogram similarity.

    quality.py
        Feature availability and reliability.
        Pair qualities q[r, i, j] describe which evidence can contribute
        to an association and how strongly it should contribute.

    prediction.py
        Track prediction at the candidate observation time:
            predicted_position = position + velocity * delta_time
        Predictions remain separate from actual observations.

    association_scores.py
        Geometric, motion, appearance and size similarities.
        Quality-weighted association:
            S[i, j] = sum(w[r] * q[r, i, j] * S[r, i, j])
                      / sum(w[r] * q[r, i, j])
        Effective weights:
            effective_w[r, i, j] = w[r] * q[r, i, j]
                                  / sum(w[l] * q[l, i, j])

    assignment.py
        Admissible track/detection pairs, one-to-one global assignment
        and ambiguity margins against competing assignments.

    track_update.py
        Accepted observation updates:
        position, velocity, appearance, size and observation counts.
        Unobserved features retain their previous values and timestamps.

    social_geometry.py
        Robot-relative distance, social-zone geometry and the radius
        of an oriented ellipse in the direction of the robot.

    enrichment.py
        Construct enriched candidates from source detections.
        Calls appearance extraction and combines its outputs with
        depth/TF position, apparent size and metric size measurements.
        
Data flow:
    Source detection
        -> enriched measurements and qualities
        -> prediction from retained track state
        -> pair scores and assignment
        -> update from accepted observations
        -> social geometry and downstream interpretation

Boundary:
    This package does not subscribe to topics, perform TF lookups,
    publish messages or issue navigation commands.

    Social episode memory and temporal interpretation remain owned by
    their dedicated components; shared equations do not own ROS state.
"""