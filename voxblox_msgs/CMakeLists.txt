CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(voxblox_msgs)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

##########
# EXPORT #
##########

cs_install()
cs_export()
