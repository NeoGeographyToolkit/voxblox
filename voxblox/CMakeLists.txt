############
# PROTOBUF #
############
# General idea: first check if we have protobuf catkin, then use that.
# Otherwise use system protobuf.
set(PROTO_DEFNS proto/voxblox/Block.proto
                proto/voxblox/Layer.proto)
set(PROTO_LIBRARIES "")
#find_package(protobuf_catkin QUIET)
#if (protobuf_catkin_FOUND)
#  message(STATUS "Using protobuf_catkin")
#  PROTOBUF_CATKIN_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
#  set(PROTO_LIBRARIES ${protobuf_catkin_LIBRARIES})
#else()
  find_package(Protobuf REQUIRED)
  PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
  set(PROTO_LIBRARIES ${PROTOBUF_LIBRARIES})
  message(STATUS "Found protobuf libraries: ${PROTO_LIBRARIES}")
#endif()

# Include protofub  
include_directories("${Protobuf_INCLUDE_DIR}")

# Include voxblox header files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

####################
# SET SOURCE FILES #
####################

set("${PROJECT_NAME}_SRCS"
  src/alignment/icp.cc
  src/core/block.cc
  src/core/esdf_map.cc
  src/core/tsdf_map.cc
  src/integrator/esdf_integrator.cc
  src/integrator/esdf_occ_integrator.cc
  src/integrator/integrator_utils.cc
  src/integrator/intensity_integrator.cc
  src/integrator/tsdf_integrator.cc
  src/io/mesh_ply.cc
  src/io/sdf_ply.cc
  src/mesh/marching_cubes.cc
  src/simulation/objects.cc
  src/simulation/simulation_world.cc
  src/utils/camera_model.cc
  src/utils/evaluation_utils.cc
  src/utils/layer_utils.cc
  src/utils/neighbor_tools.cc
  src/utils/protobuf_utils.cc
  src/utils/timing.cc
  src/utils/voxel_utils.cc
)

#############
# LIBRARIES #
#############
# NOTE(mereweth@jpl.nasa.gov) - Be careful when compiling Proto cc files. It is
# best to do so only once (make a shared library for each set of Proto files).
# Otherwise, at some point, you will get errors from double-adding protobuf
# formats, of the following form:
# [libprotobuf ERROR google/protobuf/descriptor_database.cc:57]
# File already exists in database: Block.proto

# Avoid having multiple compiled copies of the same .pb.cc
# file sharing a single copy of libprotobuf.so
message(STATUS "Generated proto files: ${PROTO_SRCS}")

# Make this static as no other tool will use it
add_library(${PROJECT_NAME}_proto STATIC
    ${PROTO_SRCS})

message(STATUS "Create library ${PROJECT_NAME}_proto")

# Specify proto headers as public headers so we can install them later.
set_target_properties(${PROJECT_NAME}_proto PROPERTIES PUBLIC_HEADER "${PROTO_HDRS}")
# We need the binary folder in the include directories so we can find the proto
# headers within this library and all dependent packages.
target_include_directories(${PROJECT_NAME}_proto
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME}_proto ${PROTO_LIBRARIES})

# Make this static as no other tool will use it
add_library(${PROJECT_NAME} STATIC
  ${${PROJECT_NAME}_SRCS}
)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_proto)

# ############
# # BINARIES #
# ############

add_executable(voxblox_mesh
   bin/voxblox_mesh.cc
)
target_link_libraries(voxblox_mesh ${PROJECT_NAME} gflags glog pcl_common pcl_io Boost::filesystem)

# TODO(oalexan1): Note sure about the robustness of the logic below but it seems to work
if(APPLE)
  set_target_properties(voxblox_mesh PROPERTIES
    INSTALL_RPATH "@loader_path;${MULTIVIEW_DEPS_DIR}/lib")
elseif(UNIX) # Unix which is not Apple
  set_target_properties(voxblox_mesh PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
endif()

install(TARGETS voxblox_mesh RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# add_executable(test_load_esdf
#   test/test_load_esdf.cc
# )
# target_link_libraries(test_load_esdf ${PROJECT_NAME})

# #########
# # TESTS #
# #########
# add_custom_target(test_data)
# add_custom_command(TARGET test_data
#                   COMMAND rm -rf test_data
#                   COMMAND mkdir -p test_data
#                   COMMAND cp -r ${CMAKE_SOURCE_DIR}/test/test_data/*
#                                 test_data/ || :)

# #add_definitions(-DVISUALIZE_UNIT_TEST_RESULTS)

# catkin_add_gtest(test_approx_hash_array
#   test/test_approx_hash_array.cc
# )
# target_link_libraries(test_approx_hash_array ${PROJECT_NAME})

# catkin_add_gtest(test_tsdf_map
#   test/test_tsdf_map.cc
# )
# target_link_libraries(test_tsdf_map ${PROJECT_NAME})

# catkin_add_gtest(test_protobuf
#   test/test_protobuf.cc
# )
# target_link_libraries(test_protobuf ${PROJECT_NAME})

# catkin_add_gtest(test_tsdf_interpolator
#   test/test_tsdf_interpolator.cc
# )
# target_link_libraries(test_tsdf_interpolator ${PROJECT_NAME})

# catkin_add_gtest(test_layer
#   test/test_layer.cc
# )
# target_link_libraries(test_layer ${PROJECT_NAME})

# catkin_add_gtest(test_merge_integration
#   test/test_merge_integration.cc
# )
# target_link_libraries(test_merge_integration ${PROJECT_NAME})

# catkin_add_gtest(test_layer_utils
#   test/test_layer_utils.cc
# )
# target_link_libraries(test_layer_utils ${PROJECT_NAME})

# catkin_add_gtest(test_sdf_integrators
#   test/test_sdf_integrators.cc
# )
# target_link_libraries(test_sdf_integrators ${PROJECT_NAME})

# catkin_add_gtest(test_bucket_queue
#   test/test_bucket_queue.cc
# )
# target_link_libraries(test_bucket_queue ${PROJECT_NAME})

# catkin_add_gtest(test_clear_spheres
#   test/test_clear_spheres.cc
# )
# target_link_libraries(test_clear_spheres ${PROJECT_NAME})

# ##########
# # EXPORT #
# ##########

# # The proto headers are not installed automatically with cs_install,
# # we need to specify this here.
# install(TARGETS ${PROJECT_NAME}_proto
#         LIBRARY DESTINATION "lib"
#         PUBLIC_HEADER DESTINATION "include"
# )
# cs_install()
# cs_export(INCLUDE_DIRS include ${CMAKE_CURRENT_BINARY_DIR}
#           CFG_EXTRAS voxblox-extras.cmake)
