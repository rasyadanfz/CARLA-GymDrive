# Pygame Window
IM_WIDTH                = 1280
IM_HEIGHT               = 720
NUM_COLS                = 2  # Number of columns in the grid
NUM_ROWS                = 2  # Number of rows in the grid
MARGIN                  = 30
BORDER_WIDTH            = 10

# Vehicle and Sensors attributes
SENSOR_FPS              = 30
VERBOSE                 = False
VEHICLE_SENSORS_FILE    = 'src/config/default_sensors.json'
VEHICLE_PHYSICS_FILE    = 'src/config/default_vehicle_physics.json'
VEHICLE_MODEL           = "vehicle.tesla.model3"

# Simulation attributes
SIM_HOST                = 'localhost'
SIM_PORT                = 2000
SIM_TIMEOUT             = 30
SIM_LOW_QUALITY         = False
SIM_OFFSCREEN_RENDERING = True
SIM_NO_RENDERING        = False
SIM_FPS                 = 30

# Environment attributes
ENV_SCENARIOS_FILE      = 'src/config/scenario_2.json'
ENV_MAX_STEPS           = 1300 # Max number of steps per episode. I suggest running the helpfull-scipts/check_max_num_steps.py script to get your number
ENV_WAYPOINT_SPACING    = 7.0
