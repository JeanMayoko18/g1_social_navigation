
#!/usr/bin/env python3

from pathlib import Path
import shutil
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

PACKAGE_DIR = Path(__file__).resolve().parent
MSG_DIR = PACKAGE_DIR / "msg"

CATEGORIES = {
    "perception": [
        "TrackedPerson.msg",
        "TrackedPersonArray.msg",
        "SocialZone.msg",
        "SocialZoneArray.msg",
        "SpeakerAssociation.msg",
        "PersonDetectionArray.msg",
        "FeatureMetadata.msg",
        "PositionFeature.msg",
        "AppearanceFeature.msg",
        "DepthEvidence.msg",
        "PersonDetection.msg",
    ],

    "navigation": [
        "SocialNavigationDecision.msg",
        "SocialState.msg",
        "NavigationState.msg",
        "NavigationControl.msg",
        "NavigationGoalContext.msg",
        "RobotMotionState.msg",
        "SocialParticipant.msg",
    ],

    "trajectory": [
        "TrajectoryGenerationRequest.msg",
        "CandidateTrajectory.msg",
        "TrajectorySocialValidation.msg",
    ],

    "collaboration": [
        "HumanIntent.msg",
        "DialogueRequest.msg",
        "CollaborativeDecision.msg",
    ],

    "explainability": [
        "ExplanationContext.msg",
        "ExplanationText.msg",
        "SpeechStatus.msg",
        "SpeechCancelRequest.msg",
    ],
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not MSG_DIR.is_dir():
        print(f"ERROR: msg directory not found: {MSG_DIR}")
        return 1

    print("=" * 60)
    print("G1 SOCIAL MSGS - MESSAGE ORGANIZER")
    print("=" * 60)

    # Validate the configuration before changing files.
    names = [
        name
        for messages in CATEGORIES.values()
        for name in messages
    ]

    if len(names) != len(set(names)):
        print("ERROR: Duplicate names in category configuration")
        return 1

    # Check destination conflicts before moving anything.
    conflicts = []

    for category, messages in CATEGORIES.items():
        for name in messages:
            source = MSG_DIR / name
            target = MSG_DIR / category / name

            if (source.exists() or source.is_symlink()) and (
                target.exists() or target.is_symlink()
            ):
                conflicts.append(name)

    if conflicts:
        print("\nERROR: Source and destination both exist:")
        for name in conflicts:
            print(f"  - {name}")
        print("No files have been moved.")
        return 1

    # Create categories.
    for category in CATEGORIES:
        directory = MSG_DIR / category
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] {directory.relative_to(PACKAGE_DIR)}")

    moved = 0
    existing = 0
    missing = 0

    # Move the message files.
    for category, messages in CATEGORIES.items():
        print(f"\n[{category.upper()}]")

        for name in messages:
            source = MSG_DIR / name
            target = MSG_DIR / category / name

            if source.exists() and source.is_file():
                shutil.move(str(source), str(target))
                print(f"  [MOVE] {name}")
                moved += 1

            elif target.is_file():
                print(f"  [OK]   {name} already organized")
                existing += 1

            else:
                print(f"  [WARN] {name} not found")
                missing += 1

    # Detect unexpected messages left in msg/.
    unknown = sorted(
        path.name
        for path in MSG_DIR.glob("*.msg")
    )

    if unknown:
        print("\n[WARNING] Unclassified messages:")
        for name in unknown:
            print(f"  - {name}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Moved:              {moved}")
    print(f"Already organized:  {existing}")
    print(f"Missing:            {missing}")
    print(f"Unclassified:       {len(unknown)}")
    print(f"Total configured:   {len(names)}")
    print("=" * 60)

    return 0 if missing == 0 and not unknown else 1


if __name__ == "__main__":
    sys.exit(main())

"""
# Sauvegarder les modifications dans Git
git add .
git commit -m "Backup before message reorganization"

# Organiser les messages
python3 organize_msgs.py

# Vérifier l'organisation
find msg -type f -name "*.msg" | sort

"""