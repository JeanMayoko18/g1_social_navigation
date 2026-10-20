#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

/**
 * Policy joint index (0–28) -> EDU23 hardware availability
 *
 * Policy order (29 DoF):
 *
 *  0  LeftHipPitch          ✓
 *  1  LeftHipRoll           ✓
 *  2  LeftHipYaw            ✓
 *  3  LeftKnee              ✓
 *  4  LeftAnklePitch        ✓
 *  5  LeftAnkleRoll         ✓
 *
 *  6  RightHipPitch         ✓
 *  7  RightHipRoll          ✓
 *  8  RightHipYaw           ✓
 *  9  RightKnee             ✓
 * 10  RightAnklePitch       ✓
 * 11  RightAnkleRoll        ✓
 *
 * 12  WaistYaw              ✓
 * 13  WaistRoll             ✗ (locked on EDU23)
 * 14  WaistPitch            ✗ (locked on EDU23)
 *
 * 15  LeftShoulderPitch     ✓
 * 16  LeftShoulderRoll      ✓
 * 17  LeftShoulderYaw       ✓
 * 18  LeftElbow             ✓
 * 19  LeftWristRoll         ✓
 * 20  LeftWristPitch        ✗ (not present on EDU23)
 * 21  LeftWristYaw          ✗ (not present on EDU23)
 *
 * 22  RightShoulderPitch    ✓
 * 23  RightShoulderRoll     ✓
 * 24  RightShoulderYaw      ✓
 * 25  RightElbow            ✓
 * 26  RightWristRoll        ✓
 * 27  RightWristPitch       ✗ (not present on EDU23)
 * 28  RightWristYaw         ✗ (not present on EDU23)
 */

static constexpr bool is_edu23_active_joint(int policy_joint_index)
{
    switch (policy_joint_index)
    {
        // ===== LOWER BODY =====
        case 0:   // LeftHipPitch
        case 1:   // LeftHipRoll
        case 2:   // LeftHipYaw
        case 3:   // LeftKnee
        case 4:   // LeftAnklePitch
        case 5:   // LeftAnkleRoll

        case 6:   // RightHipPitch
        case 7:   // RightHipRoll
        case 8:   // RightHipYaw
        case 9:   // RightKnee
        case 10:  // RightAnklePitch
        case 11:  // RightAnkleRoll

        // ===== WAIST =====
        case 12:  // WaistYaw

        // ===== LEFT ARM =====
        case 15:  // LeftShoulderPitch
        case 16:  // LeftShoulderRoll
        case 17:  // LeftShoulderYaw
        case 18:  // LeftElbow
        case 19:  // LeftWristRoll

        // ===== RIGHT ARM =====
        case 22:  // RightShoulderPitch
        case 23:  // RightShoulderRoll
        case 24:  // RightShoulderYaw
        case 25:  // RightElbow
        case 26:  // RightWristRoll
            return true;

        // ===== DISABLED ON EDU23 =====
        case 13:  // WaistRoll
        case 14:  // WaistPitch
        case 20:  // LeftWristPitch
        case 21:  // LeftWristYaw
        case 27:  // RightWristPitch
        case 28:  // RightWristYaw
            return false;

        default:
            return false;
    }
}

static constexpr int kPolicyToMs[29] = {
    // 0..12
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12,
    // 13,14 absent
    -1, -1,
    // 15..19 -> hw23 13..17
    13, 14, 15, 16, 17,
    // 20,21 absent
    -1, -1,
    // 22..26 -> hw23 18..22
    18, 19, 20, 21, 22,
    // 27,28 absent
    -1, -1
};
static_assert(sizeof(kPolicyToMs) / sizeof(int) == 29, "kPolicyToMs must have 29 entries");

// ------------------------------------------------------------------
// Compile-time mapping:
// policy joint index (0..28) -> hw23 slot (0..22 or -1 if absent)
// ------------------------------------------------------------------
static constexpr int kPolicyToHw23[29] = {
    // 0..12 (legs + waist_yaw)
     0,  1,  2,  3,  4,  5,
     6,  7,  8,  9, 10, 11,
    12,

    // 13,14 absent
    -1, -1,

    // 15..19 (left arm)
    13, 14, 15, 16, 17,

    // 20,21 absent
    -1, -1,

    // 22..26 (right arm)
    18, 19, 20, 21, 22,

    // 27,28 absent
    -1, -1
};
static_assert(sizeof(kPolicyToHw23) / sizeof(int) == 29, "kPolicyToHw23 must have 29 entries");

// ------------------------------------------------------------------
// Policy joint names (29 DoF)
// The order MUST match the policy joint index order (0..28).
// These names MUST also match the URDF/Xacro joint names exactly.
// ------------------------------------------------------------------
static constexpr std::array<const char*, 29> kPolicyJointNames = {{
    "left_hip_pitch_joint",    // 0
    "left_hip_roll_joint",     // 1
    "left_hip_yaw_joint",      // 2
    "left_knee_joint",         // 3
    "left_ankle_pitch_joint",  // 4
    "left_ankle_roll_joint",   // 5

    "right_hip_pitch_joint",   // 6
    "right_hip_roll_joint",    // 7
    "right_hip_yaw_joint",     // 8
    "right_knee_joint",        // 9
    "right_ankle_pitch_joint", // 10
    "right_ankle_roll_joint",  // 11

    "waist_yaw_joint",         // 12
    "waist_roll_joint",        // 13
    "waist_pitch_joint",       // 14

    "left_shoulder_pitch_joint", // 15
    "left_shoulder_roll_joint",  // 16
    "left_shoulder_yaw_joint",   // 17
    "left_elbow_joint",          // 18
    "left_wrist_roll_joint",     // 19
    "left_wrist_pitch_joint",    // 20
    "left_wrist_yaw_joint",      // 21

    "right_shoulder_pitch_joint", // 22
    "right_shoulder_roll_joint",  // 23
    "right_shoulder_yaw_joint",   // 24
    "right_elbow_joint",          // 25
    "right_wrist_roll_joint",     // 26
    "right_wrist_pitch_joint",    // 27
    "right_wrist_yaw_joint"       // 28
}};

static inline std::vector<std::string> GetEdu23ActiveJointNames()
{
    std::vector<std::string> names;
    names.reserve(23);

    for (int p = 0; p < 29; ++p)
    {
        if (is_edu23_active_joint(p))
        {
            names.emplace_back(kPolicyJointNames[p]);
        }
    }

    return names;
}