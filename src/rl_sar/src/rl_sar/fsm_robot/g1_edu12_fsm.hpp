/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef G1_EDU12_FSM_HPP
#define G1_EDU12_FSM_HPP

#include <chrono>
#include "fsm.hpp"
#include "rl_sdk.hpp"
#include "g1_edu23_joints.hpp"

namespace g1_edu12_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }
    /*
    void Run() override
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 8;
            fsm_command->motor_command.tau[i] = 0;
        }
    }
    */

    void Run() override
    {
        const int ndof = std::min(
            rl.params.Get<int>("num_of_dofs"),
            static_cast<int>(fsm_command->motor_command.q.size())
        );
        /*
        std::cout
        << "[FSM PASSIVE] num_of_dofs="
        << rl.params.Get<int>("num_of_dofs")
        << " motor_cmd_size="
        << fsm_command->motor_command.q.size()
        << " used_ndof="
        << ndof
        << std::endl;
        */

        for (int i = 0; i < ndof; ++i)
        {
            fsm_command->motor_command.dq[i] = 0.0f;
            fsm_command->motor_command.kp[i] = 0.0f;
            fsm_command->motor_command.kd[i] = 8.0f;
            fsm_command->motor_command.tau[i] = 0.0f;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_getup = 0.0f;
    std::vector<float> getup_start_q;
    std::vector<float> getup_target_q;
    bool interpolation_completed_displayed_ = false;
    
    void Enter() override
    {
        rl.OnEnterGetUp();

        percent_getup = 0.0f;
        interpolation_completed_displayed_ = false;
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;

        const std::string robot_config_path =
            rl.robot_name + "/robomimic/locomotion";

        if (!rl.OnEnterPolicy(robot_config_path))
        {
            std::cerr << "[FATAL] GetUp: OnEnterPolicy failed\n";
            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }

        const int ndof =
            rl.params.Get<int>("num_of_dofs");

        getup_start_q =
            rl.now_state.motor_state.q;

        getup_target_q =
            rl.params.Get<std::vector<float>>(
                "edu12_getup_dof_pos",
                {}
            );

        if ((int)getup_start_q.size() != ndof)
        {
            std::cerr
                << "[FATAL] GetUp Enter: getup_start_q size="
                << getup_start_q.size()
                << " expected ndof="
                << ndof
                << std::endl;

            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }

        if ((int)getup_target_q.size() != ndof)
        {
            std::cerr
                << "[FATAL] GetUp Enter: getup_target_q size="
                << getup_target_q.size()
                << " expected ndof="
                << ndof
                << std::endl;

            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }

        std::cout
            << "[GETUP] Enter ready: ndof="
            << ndof
            << " start="
            << getup_start_q.size()
            << " target="
            << getup_target_q.size()
            << std::endl;
    }

    /*
            void Run() override
        {
            Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 2.0f, "Getting up", true);
        }
    */
    void Run() override
    {
        const int ndof =
            rl.params.Get<int>("num_of_dofs");

        if ((int)getup_target_q.size() != ndof)
        {
            std::cerr
                << "[FATAL] GetUp: getup_target_q size="
                << getup_target_q.size()
                << " expected ndof="
                << ndof
                << "\n";

            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }

        if ((int)getup_start_q.size() != ndof)
        {
            std::cerr
                << "[FATAL] GetUp: getup_start_q size="
                << getup_start_q.size()
                << " expected ndof="
                << ndof
                << "\n";

            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }

        if (percent_getup == 0.0f)
        {
            std::cout
                << "[GETUP] Starting EDU12 GetUp interpolation"
                << " ndof=" << ndof
                << " target=edu12_getup_dof_pos"
                << std::endl;
        }


        if (!Interpolate(
                percent_getup,
                getup_start_q,
                getup_target_q,
                2.0f,
                "Getting up",
                true))
        {
            if (!interpolation_completed_displayed_)
            {
                std::cout
                    << "[GETUP] Interpolation completed"
                    << std::endl;

                interpolation_completed_displayed_ = true;
            }
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStateRLRoboMimicLocomotion";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }
};

class RLFSMStateRLRoboMimicLocomotion : public RLFSMState
{
public:
RLFSMStateRLRoboMimicLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLRoboMimicLocomotion") {}

    float percent_transition = 0.0f;

     void Enter() override
    {
        rl.episode_length_buf = 0;

        rl.config_name = "robomimic/locomotion";
        const std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            // Polymorphic hook:
            //  - Default RL: InitRL(robot_config_path)
            //  - EDU23 adapter: merge base+policy + build mapping table
            if (!rl.OnEnterPolicy(robot_config_path)) {
            std::cerr << "[FATAL] OnEnterPolicy failed\n";
            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
            }
            rl.OnEnterLocomotion();
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "OnEnterPolicy() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        //std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        static auto last_print = std::chrono::steady_clock::now();

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_print).count();
        if (dt >= 5.0)
        {
            last_print = now;

            std::cout << LOGGER::INFO
                    << "RL Controller [" << rl.config_name << "] "
                    << "x:" << rl.control.x
                    << " y:" << rl.control.y
                    << " yaw:" << rl.control.yaw
                    << std::endl;
        }
        RLControl();
    }

    void Exit() override
    {
        rl.OnExitLocomotion();
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLRoboMimicLocomotion";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadDown)
        {
            return "RLFSMStateRLRoboMimicCharleston";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num3 || rl.control.current_gamepad == Input::Gamepad::RB_DPadLeft)
        {
            return "RLFSMStateRLWholeBodyTrackingDance102";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num4 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
        {
            return "RLFSMStateRLWholeBodyTrackingGangnamStyle";
        }
        return state_name_;
    }
};

class RLFSMStateRLRoboMimicCharleston : public RLFSMState
{
public:
    RLFSMStateRLRoboMimicCharleston(RL *rl) : RLFSMState(*rl, "RLFSMStateRLRoboMimicCharleston") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "robomimic/charleston";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }

        rl.motion_length = 18.0;
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        float motion_time = rl.episode_length_buf * rl.params.Get<float>("dt") * rl.params.Get<int>("decimation");
        motion_time = fmin(motion_time, rl.motion_length);
        float percent = motion_time / rl.motion_length;
        LOGGER::PrintProgress(percent, rl.config_name);

        RLControl();

        if (motion_time / rl.motion_length == 1)
        {
            rl.fsm.RequestStateChange("RLFSMStateRLRoboMimicLocomotion");
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLRoboMimicLocomotion";
        }
        return state_name_;
    }
};

class RLFSMStateRLWholeBodyTrackingDance102 : public RLFSMState
{
public:
    RLFSMStateRLWholeBodyTrackingDance102(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWholeBodyTrackingDance102") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "whole_body_tracking/dance_102";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);

            // Initialize motion loader
            std::string motion_file_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + rl.params.Get<std::string>("motion_file");
            float fps = 1.0f / (rl.params.Get<float>("dt") * rl.params.Get<int>("decimation"));
            rl.motion_loader = std::make_unique<MotionLoader>(motion_file_path, fps);
            rl.motion_length = rl.motion_loader->GetDuration();

            auto waist_sdk_indices = rl.params.Get<std::vector<int>>("waist_joint_indices");
            std::vector<float> waist_angles = {
                fsm_state->motor_state.q[rl.InverseJointMapping(waist_sdk_indices[0])],
                fsm_state->motor_state.q[rl.InverseJointMapping(waist_sdk_indices[1])],
                fsm_state->motor_state.q[rl.InverseJointMapping(waist_sdk_indices[2])]
            };
            rl.motion_loader->Reset(fsm_state->imu.quaternion, waist_angles);

            std::cout << LOGGER::INFO << "Motion duration: " << rl.motion_length << "s" << std::endl;

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // Calculate motion time and progress
        float motion_time = rl.episode_length_buf * rl.params.Get<float>("dt") * rl.params.Get<int>("decimation");
        motion_time = std::fmin(motion_time, rl.motion_length);
        float percent = motion_time / rl.motion_length;
        LOGGER::PrintProgress(percent, rl.config_name);

        rl.motion_loader->Update(motion_time);

        RLControl();

        if (motion_time / rl.motion_length == 1)
        {
            rl.fsm.RequestStateChange("RLFSMStateRLRoboMimicLocomotion");
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        return state_name_;
    }
};

class RLFSMStateRLWholeBodyTrackingGangnamStyle : public RLFSMState
{
public:
RLFSMStateRLWholeBodyTrackingGangnamStyle(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWholeBodyTrackingGangnamStyle") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "whole_body_tracking/gangnam_style";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);

            // Initialize motion loader
            std::string motion_file_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + rl.params.Get<std::string>("motion_file");
            float fps = 1.0f / (rl.params.Get<float>("dt") * rl.params.Get<int>("decimation"));
            rl.motion_loader = std::make_unique<MotionLoader>(motion_file_path, fps);
            rl.motion_length = rl.motion_loader->GetDuration();

            auto waist_sdk_indices = rl.params.Get<std::vector<int>>("waist_joint_indices");
            std::vector<float> waist_angles = {
                fsm_state->motor_state.q[rl.InverseJointMapping(waist_sdk_indices[0])],
                fsm_state->motor_state.q[rl.InverseJointMapping(waist_sdk_indices[1])],
                fsm_state->motor_state.q[rl.InverseJointMapping(waist_sdk_indices[2])]
            };
            rl.motion_loader->Reset(fsm_state->imu.quaternion, waist_angles);

            std::cout << LOGGER::INFO << "Motion duration: " << rl.motion_length << "s" << std::endl;

            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // Calculate motion time and progress
        float motion_time = rl.episode_length_buf * rl.params.Get<float>("dt") * rl.params.Get<int>("decimation");
        motion_time = std::fmin(motion_time, rl.motion_length);
        float percent = motion_time / rl.motion_length;
        LOGGER::PrintProgress(percent, rl.config_name);

        rl.motion_loader->Update(motion_time);

        RLControl();

        if (motion_time / rl.motion_length == 1)
        {
            rl.fsm.RequestStateChange("RLFSMStateRLRoboMimicLocomotion");
        }
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLRoboMimicLocomotion";
        }
        return state_name_;
    }
};

} // namespace g1_edu12_fsm

class G1Edu12FSMFactory : public FSMFactory
{
public:
    G1Edu12FSMFactory(const std::string& initial)
        : initial_state_(initial) {}

    std::shared_ptr<FSMState> CreateState(
        void *context,
        const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);

        if (state_name == "RLFSMStatePassive")
            return std::make_shared<g1_edu12_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<g1_edu12_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<g1_edu12_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLRoboMimicLocomotion")
            return std::make_shared<g1_edu12_fsm::RLFSMStateRLRoboMimicLocomotion>(rl);
        else if (state_name == "RLFSMStateRLRoboMimicCharleston")
            return std::make_shared<g1_edu12_fsm::RLFSMStateRLRoboMimicCharleston>(rl);
        else if (state_name == "RLFSMStateRLWholeBodyTrackingDance102")
            return std::make_shared<g1_edu12_fsm::RLFSMStateRLWholeBodyTrackingDance102>(rl);
        else if (state_name == "RLFSMStateRLWholeBodyTrackingGangnamStyle")
            return std::make_shared<g1_edu12_fsm::RLFSMStateRLWholeBodyTrackingGangnamStyle>(rl);

        return nullptr;
    }

    std::string GetType() const override
    {
        return "g1_edu12";
    }

    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLRoboMimicLocomotion",
            "RLFSMStateRLRoboMimicCharleston",
            "RLFSMStateRLWholeBodyTrackingDance102",
            "RLFSMStateRLWholeBodyTrackingGangnamStyle"
        };
    }

    std::string GetInitialState() const override
    {
        return initial_state_;
    }

private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(G1Edu12FSMFactory, "RLFSMStatePassive")

#endif // G1_EDU12_FSM_HPP
