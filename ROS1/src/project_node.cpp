#include <ros/ros.h>
#include <std_msgs/String.h>
#include <iostream>

void writeMsgToLog(const std_msgs::String::ConstPtr& msg){
	ROS_INFO("%s", msg-> data.c_str());

}

int main(int argc, char **argv){
	std::string username;
	std::cout << "Enter Username: ";
	std::getline(std::cin, username);
	ros::init(argc, argv, username);
	ros::NodeHandle nh;
	
	ros::Publisher topic_pub = nh.advertise<std_msgs::String>("Chat", 1000);
	ros::Rate loop_rate(1);

	ros::Subscriber topic_sub= nh.subscribe("Chat", 1000, writeMsgToLog);
	ros::AsyncSpinner spinner(1);
	spinner.start();

	while (ros::ok()) {
		std::string message;
		std::getline(std::cin, message);
		std_msgs::String msg;
		msg.data = message;
		topic_pub.publish(msg);
		ros::spinOnce();
		loop_rate.sleep();
	}
}