#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/io/layer_io.h"
#include "voxblox/simulation/simulation_world.h"
#include "voxblox/utils/evaluation_utils.h"
#include "voxblox/utils/layer_utils.h"
#include "voxblox_ros/tsdf_server.h"
#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

DECLARE_bool(alsologtostderr);
DECLARE_bool(logtostderr);
DECLARE_int32(v);

using namespace voxblox;  // NOLINT

// Read an affine matrix with double values
void readAffine(Eigen::Affine3d & T, std::string const& filename) {
  Eigen::MatrixXd M(4, 4);

  std::ifstream ifs(filename.c_str(), std::ifstream::in);
  for (int row = 0; row < 4; row++){
    for (int col = 0; col < 4; col++){
      double val;
      if ( !(ifs >> val) ) {
        ROS_ERROR_STREAM("Could not read a 4x4 matrix from: " << filename);
        return;
      }
      M(row, col) = val;
    }
  }

  //std::cout << "-Read:\n" << M << std::endl;

  T.linear()      = M.block<3, 3>(0, 0);
  T.translation() = M.block<3,1>(0,3);

  //std::cout << "affine is\n" << T.matrix() << std::endl;
}

class BatchSdfIntegrator {
 public:
  BatchSdfIntegrator(){}

  void Run(std::string const& index_file, std::string const& out_cloud,
           double max_ray_length_m, double voxel_size,
           std::string const& integrator, int beg, int end){

    voxel_size_ = voxel_size;
    voxels_per_side_ = 16;
    block_size_ = voxels_per_side_ * voxel_size_;

    // The grid will exist until this far away from the surface in the normal
    // direction to it. 
    truncation_distance_ = 20 * voxel_size_;

    double intensity_max_value = 256.0;

    TsdfIntegratorBase::Config config;
    config.default_truncation_distance = truncation_distance_;
    config.max_ray_length_m = max_ray_length_m;
    config.integrator_threads = 16;
    config.voxel_carving_enabled = false;

    // The default weight of 1.0e+4 seemed too small. But care here,
    // as doing weighted averages with floats can result in loss of precision.
    config.max_weight = 1.0e+5; 

    std::cout << "Index file:          " << index_file << std::endl;
    std::cout << "Output cloud:        " << out_cloud << std::endl;
    std::cout << "Voxel size:          " << voxel_size_ << std::endl;
    std::cout << "Voxels per side:     " << voxels_per_side_ << std::endl;
    std::cout << "Block size           " << block_size_ << std::endl;
    std::cout << "Truncation distance: " << truncation_distance_ << std::endl;
    std::cout << "Intensity max value: " << intensity_max_value << std::endl;
    std::cout << "Integrator:          " << integrator << std::endl;

    std::cout << config.print() << std::endl;
    
    // Simple integrator
    Layer<TsdfVoxel> simple_layer(voxel_size_, voxels_per_side_);
    SimpleTsdfIntegrator simple_integrator(config, &simple_layer);
    
    // Merged integrator
    Layer<TsdfVoxel> merged_layer(voxel_size_, voxels_per_side_);
    MergedTsdfIntegrator merged_integrator(config, &merged_layer);

    // Fast integrator
    Layer<TsdfVoxel> fast_layer(voxel_size_, voxels_per_side_);
    FastTsdfIntegrator fast_integrator(config, &fast_layer);

    mesh_layer_.reset(new MeshLayer(block_size_));
    color_map_.reset(new GrayscaleColorMap());
    color_map_->setMaxValue(intensity_max_value);

    MeshIntegratorConfig mesh_config;
    if (integrator == "simple") 
      mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(mesh_config,
                                                           &simple_layer,
                                                           mesh_layer_.get()));
    else if (integrator == "merged") 
      mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(mesh_config,
                                                           &merged_layer,
                                                           mesh_layer_.get()));
    else if (integrator == "fast") 
      mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(mesh_config,
                                                           &fast_layer,
                                                           mesh_layer_.get()));
    else {
      std::cout << "Unknown integrator: " << integrator << std::endl;
      exit(1);
    }
    
    std::vector<std::string> clouds, poses;
    std::cout << "Reading: " << index_file << std::endl;

    std::ifstream ifs(index_file.c_str());
    std::string pose_file, cloud_file;
    while (ifs >> pose_file >> cloud_file){
      poses.push_back(pose_file);
      clouds.push_back(cloud_file);
    }
    
    for (int i = 0; i < static_cast<int>(clouds.size()); i++) {

      if (i < beg || i >= end) 
        continue;
      
      Pointcloud points_C;
      Colors colors;
      std::vector<float> weights;

      // Load the cloud. x, y, z are the coordinates, normal_x is the color, normal_y is
      // the weight. Invalid coordinates are inf, inf, inf.
      // TODO(oalexan1): This needs to be documented somewhere. 
      pcl::PointCloud<pcl::PointNormal> pointcloud_pcl;
      if (pcl::io::loadPCDFile<pcl::PointNormal> (clouds[i], pointcloud_pcl) == -1) {
        std::cout << "Could not read: " << clouds[i] << std::endl;
        return;
      }

      std::cout << "Processing: " << clouds[i] << std::endl;
      //std::cout << "Loaded " << pointcloud_pcl.width * pointcloud_pcl.height
      //          << " data points\n";

      convertPointCloudWithWeight(pointcloud_pcl, color_map_, &points_C, &colors, &weights);

      // Read the transform from camera that acquired the point clouds to world
      Eigen::Affine3d T;
      readAffine(T, poses[i]);
      
      Point position = T.translation().cast<float>();
      Quaternion rotation(T.linear().cast<float>());
      Transformation Pose(rotation, position);

      if (integrator == "simple") {
        simple_integrator.pointWeights = weights; 
        simple_integrator.integratePointCloud(Pose, points_C, colors);
      }
      else if (integrator == "merged") {
        merged_integrator.pointWeights = weights; 
        merged_integrator.integratePointCloud(Pose, points_C, colors);
      }
      else if (integrator == "fast") 
        fast_integrator.integratePointCloud(Pose, points_C, colors);
    }
    
    const bool clear_mesh = true;
    if (clear_mesh) {
      constexpr bool only_mesh_updated_blocks = false;
      constexpr bool clear_updated_flag = true;
      mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                     clear_updated_flag);

      std::cout << "Writing: " << out_cloud << std::endl;
      const bool success = outputMeshLayerAsPly(out_cloud, *mesh_layer_);
      if (!success) {
        std::cout << "Could not write the mesh." << std::endl;
      }
    }

  }
  
 protected:

  // Map settings
  FloatingPoint voxel_size_;
  int voxels_per_side_;
  double block_size_;
  FloatingPoint truncation_distance_;

  std::shared_ptr<MeshLayer> mesh_layer_;
  std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;

  std::shared_ptr<ColorMap> color_map_;
};

int main(int argc, char** argv) {

  if (argc < 6) {
    std::cout << "Must specify the data index file, output cloud name, max ray length, "
              << "voxel size, and integrator type.\n";
    return 1;
  }

  int beg = 0;
  int end = std::numeric_limits<int>::max();
  std::string index_file  = argv[1];
  std::string out_cloud   = argv[2];
  double max_ray_length_m = atof(argv[3]);
  double voxel_size       = atof(argv[4]);
  std::string integrator   = argv[5];
  if (argc >= 8) {
    beg = atoi(argv[6]);
    end = atoi(argv[7]);
  }
  
  BatchSdfIntegrator B;
  B.Run(index_file, out_cloud, max_ray_length_m, voxel_size, integrator, beg, end);
  
  return 0;
}
