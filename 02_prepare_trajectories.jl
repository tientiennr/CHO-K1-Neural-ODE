# Run the data loading and preparation script. รันไฟล์แรก ทำให้ไฟล์ใหม่สามารถใช้ตัวแปร model_data ที่สร้างในไฟล์แรกได้
include(joinpath(@__DIR__, "01_load_and_check_data.jl"))

# Separate the development data from the independent test data. ตรวจค่าในคอลัมน์ dataset_role ทีละแถวว่าเท่ากับ "development" หรือไม่ จุดใน .== หมายถึงเปรียบเทียบทุกแถว model_data[เงื่อนไข, :] ตำแหน่งก่อนเครื่องหมายจุลภาค เลือกแถวตามเงื่อนไข : เลือกทุกคอลัมน์
development_data = model_data[
    model_data.dataset_role .== "development",
    :
]

test_data = model_data[
    model_data.dataset_role .== "independent test",
    :
]

# Count the rows, experiments, and papers in each dataset.
println("\nDevelopment data:")
println("Rows: ", nrow(development_data))
println(
    "Experiments: ",
    length(unique(development_data.experiment_id))
)
println(
    "Papers: ",
    length(unique(development_data.paper_id))
)

println("\nIndependent test data:")
println("Rows: ", nrow(test_data))
println(
    "Experiments: ",
    length(unique(test_data.experiment_id))
)
println(
    "Papers: ",
    length(unique(test_data.paper_id))
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Group the development and test data by experiment. รวบรวมทุกแถวที่มี experiment_id เดียวกันไว้เป็นกลุ่มเดียว เช่น ข้อมูลของ P001_E01 ทุกเวลาเป็นหนึ่ง trajectory
development_trajectories = groupby(
    development_data,
    :experiment_id
)

test_trajectories = groupby(
    test_data,
    :experiment_id
)

# Display the number of trajectories.
println("\nNumber of development trajectories: ",
    length(development_trajectories)
)

println("Number of independent test trajectories: ",
    length(test_trajectories)
)

# Select the first development trajectory for inspection.
first_trajectory = development_trajectories[1]

println("\nFirst trajectory experiment ID: ",
    first(first_trajectory.experiment_id)
)

println("Number of time points: ",
    nrow(first_trajectory)
)

println("Time points in days:")
println(first_trajectory.day)

println("VCC values:")
println(first_trajectory.vcc)

println("Glucose values:")
println(first_trajectory.glucose)

println("Titer values:")
println(first_trajectory.titer)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Load the package used to create plots.
using Plots

# Identify the rows containing observed VCC values.
vcc_rows = .!ismissing.(first_trajectory.vcc)

# Identify the rows containing observed glucose values.
glucose_rows = .!ismissing.(first_trajectory.glucose)

# Identify the rows containing observed titer values.
titer_rows = .!ismissing.(first_trajectory.titer)

# Define the common time range for all plots. # Add space around the first and last data points.
day_range = (
    minimum(first_trajectory.day) - 0.5,
    maximum(first_trajectory.day) + 0.5
)

# Display day labels from 0 to 14 at one-day intervals.
day_ticks = minimum(first_trajectory.day):1:maximum(first_trajectory.day)

# Plot the observed VCC values. legend = false หมายถึง ไม่แสดงกล่องคำอธิบายเส้นกราฟ
vcc_plot = plot(
    first_trajectory.day[vcc_rows],
    first_trajectory.vcc[vcc_rows],
    marker = :circle,
    xlabel = "Day",
    ylabel = "VCC (10^6 cells/mL)",
    title = "Observed VCC",
    xlims = day_range,
    xticks = day_ticks,
    legend = false
)

# Plot the observed glucose values.
glucose_plot = plot(
    first_trajectory.day[glucose_rows],
    first_trajectory.glucose[glucose_rows],
    marker = :circle,
    xlabel = "Day",
    ylabel = "Glucose (g/L)",
    title = "Observed Glucose",
    xlims = day_range,
    xticks = day_ticks,
    legend = false
)

# Plot the observed titer values.
titer_plot = plot(
    first_trajectory.day[titer_rows],
    first_trajectory.titer[titer_rows],
    marker = :circle,
    xlabel = "Day",
    ylabel = "Titer (g/L)",
    title = "Observed Titer",
    xlims = day_range,
    xticks = day_ticks,
    legend = false
)

# Combine the three plots into one figure.
trajectory_plot = plot(
    vcc_plot,
    glucose_plot,
    titer_plot,
    layout = (3, 1),
    size = (800, 900)
)

# Display the combined figure.
display(trajectory_plot)

# Save the figure in the project folder.
savefig(
    trajectory_plot,
    joinpath(project_directory, "first_trajectory.png")
)