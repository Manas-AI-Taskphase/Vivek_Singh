// image_reader.cpp
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        
        // Resize the image
        cv::Mat img;
        cv::resize(cv_ptr->image, img, cv::Size(), 20, 20);

        // Display the image
        cv::imshow("Image Viewer", img);
        cv::waitKey(10); // Adjust the wait time as needed
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_reader");
    ros::NodeHandle nh;

    // Subscribe to the image topic
    ros::Subscriber sub = nh.subscribe("image", 1, imageCallback);

    ros::spin();
    return 0;
}