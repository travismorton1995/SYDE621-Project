# examples/Python/Advanced/multiway_registration.py
import open3d as o3d
from open3d import *
import numpy as np
import copy

voxel_size = 0.002
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def load_point_clouds(voxel_size = 0.0):
    '''
    Reads in the 2 different bunny images and downsamples them for manipulation.
    This function can be expanded to import more than 1 point cloud and the script
    will still be able to merge the files.
    '''
    
    pcds = []
    #The Original bunny
    pcd = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun000_Structured.pcd")
    pcd_down = pcd.voxel_down_sample(voxel_size = voxel_size)
    pcds.append(pcd_down)
    
    #The bunny scanned from a 45 degree offset
    pcd = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun045_Structured.pcd")
    pcd_down = pcd.voxel_down_sample(voxel_size = voxel_size)
    pcds.append(pcd_down)
    
    return pcds


def pairwise_registration(source, target):
    '''
    Performs a coarse pass then a fine pass of ICP Point-to-Point.
    Returns the transformation and the information matrix fo the ICP algorithm
    '''
    print("Apply point-to-point ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(source, target,
            max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_fine = o3d.pipelines.registration.registration_icp(source, target,
            max_correspondence_distance_fine, icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds,
        max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = open3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(open3d.pipelines.registration.PoseGraphNode(odometry))
    
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                    pcds[source_id], pcds[target_id])
            print("Build PoseGraph")
            if target_id == source_id + 1: # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(open3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(open3d.pipelines.registration.PoseGraphEdge(source_id, target_id,
                        transformation_icp, information_icp, uncertain = False))
            else: # loop closure case
                pose_graph.edges.append(PoseGraphEdge(source_id, target_id,
                        transformation_icp, information_icp, uncertain = True))
                
            #***
            #Evaluate the transformation here
            #draw_registration_result(pcds[source_id], pcds[target_id], transformation_icp)
            #print(transformation_icp)
            #***
    return pose_graph


if __name__ == "__main__":
    print("Visualizing the initial scan offsets")
    #voxel_size = 0.005 # means 0.5cm for this dataset
    pcds_down = load_point_clouds(voxel_size)
    o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds_down,
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance = max_correspondence_distance_fine,
            edge_prune_threshold = 0.25,
            reference_node = 0)
    optimize = False
    if optimize == True:
        o3d.pipelines.registration.global_optimization(pose_graph,
                #o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationGaussNewton(),                        
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
    
        

    print("Transform points and display")
    threshold = 0.02
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    
    source = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun000_Structured.pcd")
    target = o3d.io.read_point_cloud("./stanford_bunny/stanford_bunny/data/bun045_Structured.pcd")

    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                        threshold, pose_graph.nodes[1].pose)
    print(evaluation)
        
        
    o3d.visualization.draw_geometries(pcds_down)
    
    #transform here

    print("Make a combined point cloud")
    pcds = load_point_clouds(voxel_size)
    pcd_combined = open3d.geometry.PointCloud() 
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size = voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])