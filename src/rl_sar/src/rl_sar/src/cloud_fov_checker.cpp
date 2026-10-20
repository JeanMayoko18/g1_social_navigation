#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cmath>
#include <cstring>
#include <vector>
#include <limits>

class CloudFovChecker : public rclcpp::Node
{
public:
    CloudFovChecker()
    : Node("cloud_fov_checker")
    {
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/cloud_fixed",
            qos,
            std::bind(&CloudFovChecker::callback,this,std::placeholders::_1));

        RCLCPP_INFO(get_logger(),"Waiting for one PointCloud...");
    }

private:

    int fieldOffset(const sensor_msgs::msg::PointCloud2 &msg,
                    const std::string &name)
    {
        for(const auto &f : msg.fields)
            if(f.name==name)
                return f.offset;

        return -1;
    }

    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        int ox=fieldOffset(*msg,"x");
        int oy=fieldOffset(*msg,"y");

        if(ox<0 || oy<0)
        {
            RCLCPP_ERROR(get_logger(),"x/y fields not found");
            return;
        }

        double amin=999;
        double amax=-999;

        std::vector<int> hist(36,0);

        size_t count=0;

        for(size_t i=0;i<msg->width*msg->height;i++)
        {
            size_t base=i*msg->point_step;

            float x,y;

            std::memcpy(&x,&msg->data[base+ox],sizeof(float));
            std::memcpy(&y,&msg->data[base+oy],sizeof(float));

            if(!std::isfinite(x)||!std::isfinite(y))
                continue;

            double a=atan2(y,x)*180.0/M_PI;

            amin=std::min(amin,a);
            amax=std::max(amax,a);

            int bin=int((a+180.0)/10.0);

            if(bin>=0 && bin<36)
                hist[bin]++;

            count++;
        }

        std::cout<<"\n=====================================\n";
        std::cout<<"Cloud diagnostics\n";
        std::cout<<"=====================================\n";
        std::cout<<"points : "<<count<<"\n";
        std::cout<<"angle min : "<<amin<<" deg\n";
        std::cout<<"angle max : "<<amax<<" deg\n";
        std::cout<<"FOV : "<<amax-amin<<" deg\n";

        std::cout<<"\nHistogram (10 deg)\n";

        for(int i=0;i<36;i++)
        {
            double a0=-180+i*10;
            double a1=a0+10;

            std::cout
                <<a0<<" -> "<<a1
                <<" : "<<hist[i]<<"\n";
        }

        rclcpp::shutdown();
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc,char** argv)
{
    rclcpp::init(argc,argv);
    rclcpp::spin(std::make_shared<CloudFovChecker>());
    return 0;
}