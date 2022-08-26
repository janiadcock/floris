import numpy as np
from floris.tools import FlorisInterface
import matplotlib.pyplot as plt
from floris.tools.visualization import plot_rotor_values

# 1. Load an input file
fi = FlorisInterface("/Users/jani/Documents/research/windFarmControl/code/floris/examples/inputs/gch.yaml")

fi.floris.solver

# 2. Modify the inputs with a more complex wind turbine layout
D = 126.0  # Design the layout based on turbine diameter
x = [0, 0,  6 * D, 6 * D]
y = [0, 3 * D, 0, 3 * D]
# wind_directions = [270.0, 280.0]
wind_directions = [270.0]
wind_speeds = [8.0]

# Pass the new data to FlorisInterface
fi.reinitialize(
    layout=(x, y),
    wind_directions=wind_directions,
    wind_speeds=wind_speeds
)

yaw_angles = np.zeros( (1, 1, 4) )  # Construct the yaw array with dimensions for two wind directions, one wind speed, and four turbines
yaw_angles[0, :, 0] = 25            # At 270 degrees, yaw the first turbine 25 degrees
yaw_angles[0, :, 1] = 25            # At 270 degrees, yaw the second turbine 25 degrees

# 3. Calculate the velocities at each turbine for all atmospheric conditions
# All turbines have 0 degrees yaw
fi.calculate_wake(yaw_angles = yaw_angles)

print("flow_field_u: ", fi.floris.flow_field.u)

# 4. Get the total farm power
turbine_powers_baseline = fi.get_turbine_powers() / 1000.0  # Given in W, so convert to kW
farm_power_baseline = np.sum(turbine_powers_baseline, 2)  # Sum over the third dimension

print("turbine power; total power: ", turbine_powers_baseline, farm_power_baseline)

# yaw_angles = np.zeros( (1, 1, 4) )  # Construct the yaw array with dimensions for two wind directions, one wind speed, and four turbines
# fi.reinitialize(
#     layout=(x, y),
#     wind_directions=wind_directions,
#     wind_speeds=wind_speeds
# )
# fi.calculate_wake( yaw_angles=yaw_angles )
# fi.get_turbine_powers() / 1000.0
fig, _, _ , _ = plot_rotor_values(fi.floris.flow_field.u, wd_index=0, ws_index=0, n_rows=1, n_cols=4, return_fig_objects=True)
fig.suptitle("Wind direction 270")
plt.show()

# from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
#     YawOptimizationSR,
# )
# # Initialize optimizer object and run optimization using the Serial-Refine method
# yaw_opt = YawOptimizationSR(fi)#, exploit_layout_symmetry=False)
# df_opt = yaw_opt.optimize()

# print("Optimization results:")
# print(df_opt)

from floris.tools.visualization import visualize_cut_plane

# fig, axarr = plt.subplots(1, 2, figsize=(15,8))
fig, axarr = plt.subplots(1, 1, figsize=(8,8))

# horizontal_plane = fi.calculate_horizontal_plane( wd=[wind_directions[0]], height=90.0 )
# visualize_cut_plane(horizontal_plane, ax=axarr[0], title="270 - Aligned")

horizontal_plane = fi.calculate_horizontal_plane( wd=[wind_directions[0]], yaw_angles=yaw_angles[0:1,0:1] , height=90.0 )
visualize_cut_plane(horizontal_plane, ax=axarr, title="270 - Yawed")
plt.show()