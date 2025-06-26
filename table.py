import matplotlib.pyplot as plt


# Lambda values
lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# Data series
drone_sat_r1 = [94.53, 95.00, 97.18, 95.23, 94.85, 89.65]
# drone_sat_ap = [95.41, 95.90, 97.78, 96.22, 95.91, 91.44]
sat_drone_r1 = [96.25, 96.25, 98.75, 97.50, 96.25, 93.75]
# sat_drone_ap = [93.21, 94.13, 95.58, 94.75, 92.63, 90.85]

# Configure plot
plt.figure(figsize=(12, 7))
# font = {'family': 'Times New Roman', 'size': 24}
font = {'family': 'STIXGeneral', 'size': 28}
plt.rc('font', **font)


# Plot lines with different styles
plt.plot(lambdas, drone_sat_r1, 'r-o', label='Drone → Satellite R@1', linewidth=3, markersize=9)
# plt.plot(lambdas, drone_sat_ap, 'r--s', label='Drone → Satellite AP', linewidth=2, markersize=8)
plt.plot(lambdas, sat_drone_r1, 'b-o', label='Satellite → Drone R@1', linewidth=3, markersize=9)
# plt.plot(lambdas, sat_drone_ap, 'y--s', label='Satellite → Drone AP', linewidth=2, markersize=10)

# Configure axes
plt.xscale('log')
plt.xticks(lambdas, [str(λ) for λ in lambdas])
plt.ylim(88, 100)
plt.xlabel("λ")
plt.ylabel("R@1 (%)")

ax = plt.gca()

ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)

# 调整刻度线 (tick lines) 的宽度
ax.tick_params(axis='x', width=2)
ax.tick_params(axis='y', width=2)

# Add legend inside the plot
plt.legend(loc='lower left')  # Place legend in the lower right corner

# Add grid lines (dashed)
plt.grid(True, linestyle='-', which='major', axis='both', alpha=0.5)  # Add dashed grid lines for both axes

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#Save the plot
file_path = 'lambda.png'
plt.savefig(file_path)
