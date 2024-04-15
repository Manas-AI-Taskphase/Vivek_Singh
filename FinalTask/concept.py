import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import threading
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import airsim, open3d

# Connect to AirSim
client = airsim.MultirotorClient()

def arm_and_takeoff(targetHeight):
    while vehicle.is_armable != True:
        print("Waiting for vehicle to become armable.")
        time.sleep(1)
    print("Vehicle is now armable")

    vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode != 'GUIDED':
        print("Waiting for drone to enter GUIDED flight mode")
        time.sleep(1)
    print("Vehicle now in GUIDED MODE. Have fun!!")

    vehicle.armed = True
    while vehicle.armed == False:
        print("Waiting for vehicle to become armed.")
        time.sleep(1)
    print("Look out! Virtual props are spinning!!")

    vehicle.simple_takeoff(targetHeight)

    while True:
        print("Current Altitude: %d" % vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= .95 * targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!!")

# Function to create point cloud from LiDAR data
def create_point_cloud():
    x_positions = client.getLidarData().point_cloud
    pts = []
    cloud = PointCloud()
    for i in range(0, int(len(x_positions) / 3)):
        point = Point32()
        point.x = x_positions[3 * i] * -1
        point.y = x_positions[3 * i + 1] * -1
        point.z = x_positions[3 * i + 2] * -1
        pts.append([point.x, point.y, point.z])
        cloud.points.append(point)
    obstacles = open3d.geometry.PointCloud()
    obstacles.points = open3d.utility.Vector3dVector(np.asarray(pts))

    labels = cluster(obstacles)

    return obstacles, cloud, labels

# Function to create clusters
def cluster(pcd):
    #Downsampling
    pcd = pcd.voxel_down_sample(voxel_size=0.2)
    #Segmentation
    _, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=250)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    #Clustering
    with open3d.utility.VerbosityContextManager(open3d.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.05, min_points=5, print_progress=True))
    
    maxlabels=labels.max()
    return maxlabels

# Function to navigate the drone to a target location
def navigate_to_target(vehicle, target_location):
    print("Navigating to target:", target_location)
    vehicle.simple_goto(target_location)

# Function to calculate artificial potential field
def artificial_potential_field(current_position, target_position, obstacles):
    k_att = 0.5  # Attraction gain
    k_rep = 1.0  # Repulsion gain
    d0 = 2.0     # Distance threshold for repulsion
    
    att = k_att * (target_position - current_position)
    rep = np.zeros(3)
    
    for obstacle in obstacles:
        d = np.linalg.norm(current_position - obstacle)
        if d < d0:
            rep += k_rep * ((1 / d) - (1 / d0)) * ((current_position - obstacle) / d)
    
    return att + rep

# Function to update vehicle position
def update_vehicle_position(vehicle_location):
    while True:
        global current_vehicle_position
        current_vehicle_position = np.array([vehicle_location.lat, vehicle_location.lon, vehicle_location.alt])
        time.sleep(1)  # Update rate

# Start a thread to continuously update the vehicle position
position_thread = threading.Thread(target=update_vehicle_position)
position_thread.daemon = True
position_thread.start()

# Connect to the vehicle
connection_string = 'udp:127.0.0.1:14550'  # Adjust this according to your setup
vehicle = connect(connection_string, wait_ready=True)

arm_and_takeoff(12)

# Define target location
target_location = LocationGlobalRelative(-35.3623529000, 149.1662637000, 12.000000)

# Main loop
while True:
    # Obtain LiDAR data and obstacles
    obstacles, cloud = create_point_cloud()
    
    target_location = LocationGlobalRelative(-35.3623529000, 149.1662637000, 12.000000)

    # Calculate artificial potential field
    direction = artificial_potential_field(vehicle.location.global_frame, target_location, obstacles)
    
    # Convert direction to velocity commands
    vx = direction[0]  # Forward velocity
    vy = direction[1]  # Lateral velocity
    vz = direction[2]  # Vertical velocity
    
    # Send velocity commands to the vehicle
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms
        0,       # target system
        0,       # target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0,       # x
        0,       # y
        0,       # z
        vx,      # vx
        vy,      # vy
        vz,      # vz
        0,       # afx
        0,       # afy
        0,       # afz
        0,       # yaw angle
        0        # yaw rate
    )
    vehicle.send_mavlink(msg)
    
    # Sleep to control loop rate
    time.sleep(0.1)
