cmake_minimum_required(VERSION 2.8.3)
project(nav)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(image_reader src/image_reader.cpp)
target_link_libraries(image_reader ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(image_publisher src/image_publisher.cpp)
target_link_libraries(image_publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(map_image_loader src/map_image_loader.cpp)
target_link_libraries(map_image_loader ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)
