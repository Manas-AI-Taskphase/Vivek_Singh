import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import threading
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import airsim
import open3d as o3d

# Connect to AirSim
client = airsim.MultirotorClient()

# Function to create point cloud from LiDAR data
def create_point_cloud():
    lidar_data = client.getLidarData()
    if not lidar_data:
        print("Error: Failed to retrieve LiDAR data")
        return None, None, None

    x_positions = lidar_data.point_cloud
    pts = []
    cloud = o3d.geometry.PointCloud()
    for i in range(0, int(len(x_positions) / 3)):
        point = Point32()
        point.x = x_positions[3 * i] * -1
        point.y = x_positions[3 * i + 1] * -1
        point.z = x_positions[3 * i + 2] * -1
        pts.append([point.x, point.y, point.z])
        cloud.points.append(point)
    
    obstacles = o3d.geometry.PointCloud()
    obstacles.points = o3d.utility.Vector3dVector(np.asarray(pts))

    return obstacles, cloud, None  # For now, returning None for labels

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
    global current_vehicle_position
    current_vehicle_position = np.array([vehicle_location.lat, vehicle_location.lon, vehicle_location.alt])

# Start a thread to continuously update the vehicle position
position_thread = threading.Thread(target=update_vehicle_position)
position_thread.daemon = True
position_thread.start()

# Connect to the vehicle
connection_string = 'udp:127.0.0.1:14550'  # Adjust this according to your setup
vehicle = connect(connection_string, wait_ready=True)

# Define target location
target_location = LocationGlobalRelative(TARGET_LATITUDE, TARGET_LONGITUDE, TARGET_ALTITUDE)

# Arm and takeoff
while not vehicle.is_armable:
    print("Waiting for vehicle to become armable.")
    time.sleep(1)
print("Vehicle is now armable")

vehicle.mode = VehicleMode("GUIDED")
while vehicle.mode != 'GUIDED':
    print("Waiting for drone to enter GUIDED flight mode")
    time.sleep(1)
print("Vehicle now in GUIDED MODE.")

vehicle.armed = True
while not vehicle.armed:
    print("Waiting for vehicle to become armed.")
    time.sleep(1)
print("Vehicle is armed.")

vehicle.simple_takeoff(TARGET_ALTITUDE)

while True:
    # Obtain LiDAR data and obstacles
    obstacles, cloud, _ = create_point_cloud()
    if obstacles is None:
        print("Error: Failed to retrieve LiDAR data")
        continue
    
    # Calculate artificial potential field
    direction = artificial_potential_field(current_vehicle_position, target_location, obstacles.points)
    
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
