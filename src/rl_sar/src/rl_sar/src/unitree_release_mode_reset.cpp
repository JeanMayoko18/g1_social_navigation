#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::robot;

int main(int argc, char** argv)
{
    const std::string iface = (argc >= 2) ? argv[1] : "eth0";

    std::cout << "[RESET MS] Init ChannelFactory on " << iface << std::endl;
    ChannelFactory::Instance()->Init(0, iface);

    auto msc = std::make_unique<b2::MotionSwitcherClient>();
    msc->SetTimeout(5.0f);
    msc->Init();

    std::string form;
    std::string name;

    for (int i = 0; i < 10; ++i)
    {
        msc->CheckMode(form, name);

        std::cout << "[RESET MS] before form='" << form
                  << "' name='" << name << "'" << std::endl;

        if (!name.empty())
        {
            int ret = msc->ReleaseMode();
            std::cout << "[RESET MS] ReleaseMode ret=" << ret << std::endl;
        }

        sleep(1);

        msc->CheckMode(form, name);

        std::cout << "[RESET MS] after form='" << form
                  << "' name='" << name << "'" << std::endl;
    }

    return 0;
}