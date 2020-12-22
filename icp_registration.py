# examples/Python/Basic/icp_registration.py

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun000_Structured.pcd")
    target = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun045_Structured.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == "__main__":
    
    voxel_size = 0.005  # means 0.5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
    
    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    
    threshold = 0.02
    #evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
    #                                                    threshold, result_ransac.transformation)
    #print(evaluation)
    
    
    #source = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun000_Structured.pcd")
    #target = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun045_Structured.pcd")
    
    #source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    #target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    
    #threshold = 0.02
    #trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                         [-0.139, 0.967, -0.215, 0.7],
    #                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    #draw_registration_result(source, target, trans_init)
    #print("Initial alignment")
    #evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
    #                                                    threshold, trans_init)
    #print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)
    
    
    
    
    
    #print("Apply point-to-plane ICP")
    #result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
    #                             voxel_size)
    #print(result_icp)
    #draw_registration_result(source, target, result_icp.transformation)

    
    
    
    
    
    #print("Apply point-to-plane ICP")
    #reg_p2l = o3d.pipelines.registration.registration_icp(
    #    source, target, threshold, trans_init,
    #    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #print(reg_p2l)
    #print("Transformation is:")
    #print(reg_p2l.transformation)
    #print("")
    #draw_registration_result(source, target, reg_p2l.transformation)