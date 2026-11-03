import carla
import sys

try:
    client = carla.Client("172.17.240.1", 3000)
    client.set_timeout(5.0)
    world = client.get_world()
    print("SUCCESS: Connected to CARLA:", world.get_map().name)
except Exception as e:
    print("FAILED:", e)
    sys.exit(1)
