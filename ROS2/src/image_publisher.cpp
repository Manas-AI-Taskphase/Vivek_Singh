#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("image", 1);

    std::string image_path = "ROS2/maps/map_test.png"; // Path to your image file
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        ROS_ERROR("Failed to read image from file");
        return -1;
    }

    sensor_msgs::ImagePtr msg;
    msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();

    ros::Rate loop_rate(1); // Publish only once

    while (ros::ok()) {
        // Publish the image message
        image_pub.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}