#include <voxblox/alignment/icp.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/voxel.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/mesh/mesh_layer.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox/utils/evaluation_utils.h>
#include <voxblox/utils/layer_utils.h>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

DECLARE_bool(alsologtostderr);
DECLARE_bool(logtostderr);
DECLARE_int32(v);

DEFINE_string(index, "", "A list having input camera transforms and point cloud files.");

DEFINE_string(output_mesh, "", "The output mesh file name, in .ply format.");

DEFINE_double(min_ray_length, -1.0,
              "The minimum length of a ray from camera center to the points. Points closer than that will be ignored.");

DEFINE_double(max_ray_length, -1.0,
              "The maximum length of a ray from camera center to the points. Points beyond that will be ignored.");

DEFINE_double(voxel_size, 0.01,
              "Voxel size, in meters.");

DEFINE_double(min_weight, 1.0e-6,
              "The minimum weighting needed for a point to be included in the mesh.");

DEFINE_string(integrator, "merged", "Specify how the points should be integrated. "
              "Options: 'simple', 'merged', 'fast'. See the VoxBlox documentation for details.");

DEFINE_bool(voxel_carving_enabled, false,
            "If true, the entire length of a ray is integrated. Otherwise "
            "only the region inside the truncation distance is used. This is an advanced option.");

DEFINE_bool(enable_anti_grazing, false,
            "If true, enable anti-grazing. This is an advanced option.");

// A tool that uses voxblox to fuse point clouds in camera coordinates
// to a mesh. The input is index file, having a list of camera to world
// transforms and point clouds. The point cloud is in .pcd format,
// as output by pc_filter.

// TODO(oalexan1): Make the interface more friendly.

// TODO(oalexan1): Figure out how to save and load the
// the voxel grid which creates the mesh and how to merge
// such grids. This will make it possible to do most processing
// in parallel.

// TODO(oalexan1): See if the clouds can be loaded in
// world coordinates rather than camera coordinates.

using namespace voxblox;  // NOLINT

/// Check if all coordinates in the PCL point are finite.
template <typename PCLPoint>
inline bool isPointFinite(const PCLPoint& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

// Given a pcl::PointNormal value, use its normal_x to store a grayscale color.
// There does not seem to be a more appropriate structure.
inline Color convertColor(const pcl::PointNormal& point,
                          const std::shared_ptr<ColorMap>& color_map) {
  return color_map->colorLookup(point.normal_x);
}

/// Parse a point cloud having x, y, z, color, and weight
/// from a pcl::PointNormal structure.
/// normal_x is the color and normal_y is the weight.
/// normal_z and the curvature are ignored.
inline void convertPointCloudWithWeight(
    const pcl::PointCloud<pcl::PointNormal >& pointcloud_pcl,
    const std::shared_ptr<ColorMap>         & color_map,
    Pointcloud                              * points_C,
    Colors                                  * colors,
    std::vector<float>                      * weights) {
  points_C->reserve(pointcloud_pcl.size());
  colors->reserve(pointcloud_pcl.size());
  weights->reserve(pointcloud_pcl.size());
  points_C->clear();
  colors->clear();
  weights->clear();
  
  for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
    if (!isPointFinite(pointcloud_pcl.points[i])) {
      continue;
    }

    // Read xyz
    points_C->push_back(Point(pointcloud_pcl.points[i].x,
                              pointcloud_pcl.points[i].y,
                              pointcloud_pcl.points[i].z));

    // Read the color
    colors->emplace_back(convertColor(pointcloud_pcl.points[i], color_map));

    // read the weight
    weights->push_back(pointcloud_pcl.points[i].normal_y);
  }
}

// Read an affine matrix with double values
void readAffine(Eigen::Affine3d & T, std::string const& filename) {
  Eigen::MatrixXd M(4, 4);

  std::ifstream ifs(filename.c_str(), std::ifstream::in);
  for (int row = 0; row < 4; row++){
    for (int col = 0; col < 4; col++){
      double val;
      if ( !(ifs >> val) ) {
        LOG(FATAL) << "Could not read a 4x4 matrix from: " << filename;
        return;
      }
      M(row, col) = val;
    }
  }

  T.linear()      = M.block<3, 3>(0, 0);
  T.translation() = M.block<3,1>(0,3);
}

class BatchSdfIntegrator {
 public:
  BatchSdfIntegrator(){}

  void Run(std::string const& index_file, std::string const& output_mesh,
           double min_ray_length_m, double max_ray_length_m, double voxel_size,
           std::string const& integrator) {

    voxels_per_side_ = 16;
    block_size_ = voxels_per_side_ * voxel_size;

    // The grid will exist until this far away from the surface in the normal
    // direction to it. 
    truncation_distance_ = 20 * voxel_size;

    double intensity_max_value = 256.0;

    TsdfIntegratorBase::Config config;
    config.default_truncation_distance = truncation_distance_;
    config.min_ray_length_m = min_ray_length_m;
    config.max_ray_length_m = max_ray_length_m;
    config.integrator_threads = 16;
    config.voxel_carving_enabled = FLAGS_voxel_carving_enabled;
    config.enable_anti_grazing = FLAGS_enable_anti_grazing;

    // The default weight of 1.0e+4 seemed too small. But care here,
    // as doing weighted averages with floats can result in loss of precision.
    config.max_weight = 1.0e+5; 

    std::cout << "Index file:          " << index_file << std::endl;
    std::cout << "Output mesh:         " << output_mesh << std::endl;
    std::cout << "Voxel size:          " << voxel_size << std::endl;
    std::cout << "Voxels per side:     " << voxels_per_side_ << std::endl;
    std::cout << "Block size           " << block_size_ << std::endl;
    std::cout << "Truncation distance: " << truncation_distance_ << std::endl;
    std::cout << "Intensity max value: " << intensity_max_value << std::endl;
    std::cout << "Integrator:          " << integrator << std::endl;

    std::cout << config.print() << std::endl;
    
    // Simple integrator
    Layer<TsdfVoxel> simple_layer(voxel_size, voxels_per_side_);
    SimpleTsdfIntegrator simple_integrator(config, &simple_layer);
    
    // Merged integrator
    Layer<TsdfVoxel> merged_layer(voxel_size, voxels_per_side_);
    MergedTsdfIntegrator merged_integrator(config, &merged_layer);

    // Fast integrator
    Layer<TsdfVoxel> fast_layer(voxel_size, voxels_per_side_);
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

    mesh_config.min_weight = FLAGS_min_weight;
    
    std::cout << mesh_config.print() << std::endl;
    
    std::vector<std::string> clouds, poses;
    std::cout << "Reading: " << index_file << std::endl;

    std::ifstream ifs(index_file.c_str());
    std::string pose_file, cloud_file;
    while (ifs >> pose_file >> cloud_file){
      poses.push_back(pose_file);
      clouds.push_back(cloud_file);
    }
    
    for (int i = 0; i < static_cast<int>(clouds.size()); i++) {

      Pointcloud points_C;
      Colors colors;
      std::vector<float> weights;

      // Load the cloud. x, y, z are the coordinates, normal_x is the color, normal_y is
      // the weight. Invalid coordinates are inf, inf, inf.
      // Don't fail because of a single missing cloud.
      pcl::PointCloud<pcl::PointNormal> pointcloud_pcl;
      try { 
        if (pcl::io::loadPCDFile<pcl::PointNormal> (clouds[i], pointcloud_pcl) == -1) {
          // The above will print an error message
          continue;
        }
      } catch(std::exception const& e) {
        std::cout << e.what() << std::endl;
        continue;
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

      std::cout << "Writing: " << output_mesh << std::endl;
      const bool success = outputMeshLayerAsPly(output_mesh, *mesh_layer_);
      if (!success) {
        std::cout << "Could not write the mesh." << std::endl;
      }
    }

  }
  
 protected:

  // Map settings
  int voxels_per_side_;
  double block_size_;
  FloatingPoint truncation_distance_;

  std::shared_ptr<MeshLayer> mesh_layer_;
  std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;

  std::shared_ptr<ColorMap> color_map_;
};

int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_index == "" || FLAGS_output_mesh == "" || FLAGS_integrator == "") 
    LOG(FATAL) << "Not all inputs were specified.\n";  

  if (FLAGS_min_ray_length <= 0.0 || FLAGS_max_ray_length <= 0.0 || FLAGS_voxel_size <= 0.0) 
    LOG(FATAL) << "Min/max ray length and voxel size must be positive.\n";  
  if (FLAGS_min_weight <= 0.0) 
    LOG(FATAL) << "The minimum weight must be positive.\n";
  
  BatchSdfIntegrator B;
  B.Run(FLAGS_index, FLAGS_output_mesh, FLAGS_min_ray_length, FLAGS_max_ray_length,
        FLAGS_voxel_size,
        FLAGS_integrator);
  
  return 0;
}
