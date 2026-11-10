# Display a message to confirm that the Julia script can run.
println("Julia script is working.")

# Load the package used to read CSV files.
using CSV

# Load the package used to manage tabular data.
using DataFrames

# Store the directory containing this Julia script. @__DIR__ คือโฟลเดอร์ที่ไฟล์ Julia นี้อยู่
project_directory = @__DIR__

# Define the path of each CSV file. joinpath() นำตำแหน่งโฟลเดอร์ของไฟล์ .jl มาต่อกับชื่อไฟล์เพื่อสร้างที่อยู่ไฟล์แบบสมบูรณ์ ทำให้ไม่ต้องเขียน C:\Users\... ลงในโค้ด
conditions_path = joinpath(project_directory, "experiment_conditions.csv")
timeseries_path = joinpath(project_directory, "model_timeseries.csv")
split_path = joinpath(project_directory, "paper_split.csv")

# Display the project directory.
println("Project directory: ", project_directory)

# Check whether each CSV file exists. isfile(file_path) ใช้ตรวจสอบว่าเส้นทางที่อยู่ในตัวแปร file_path ชี้ไปยังไฟล์ที่มีอยู่จริงหรือไม่ ก่อนอ่าน CSV โดยให้ผลลัพธ์แบบ Boolean จะได้ทราบว่าปัญหาเกิดจากหาไฟล์ไม่พบหรือจากเนื้อหาภายใน CSV
println("Conditions file found: ", isfile(conditions_path))
println("Time-series file found: ", isfile(timeseries_path))
println("Paper split file found: ", isfile(split_path))

# Read each CSV file and store it as a DataFrame.
conditions = CSV.read(conditions_path, DataFrame)
timeseries = CSV.read(timeseries_path, DataFrame)
paper_split = CSV.read(split_path, DataFrame)

# Display the number of rows and columns in each DataFrame.
println("Conditions size: ", size(conditions))
println("Time-series size: ", size(timeseries))
println("Paper split size: ", size(paper_split))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Display the column names in each DataFrame.
println("\nConditions columns:")
println(names(conditions))

println("\nTime-series columns:")
println(names(timeseries))

println("\nPaper split columns:")
println(names(paper_split))

# Display the data type of each column. eachcol(conditions) แยก DataFrame ออกเป็นแต่ละคอลัมน์ eltype.(...) แสดงชนิดข้อมูลของทุกคอลัมน์ . ให้ทำ eltype กับทุกคอลัมน์ทีละคอลัมน์
println("\nConditions column types:")
println(eltype.(eachcol(conditions)))

println("\nTime-series column types:")
println(eltype.(eachcol(timeseries)))

println("\nPaper split column types:")
println(eltype.(eachcol(paper_split)))

# Display the first five rows of each DataFrame. first(conditions, 5) เลือกข้อมูล 5 แถวแรก show(..., allcols = true) แสดงทุกคอลัมน์
println("\nFirst five rows of conditions:")
show(first(conditions, 5), allrows = true, allcols = true)
println()

println("\nFirst five rows of time-series:")
show(first(timeseries, 5), allrows = true, allcols = true)
println()

println("\nFirst five rows of paper split:")
show(first(paper_split, 5), allrows = true, allcols = true)
println()

# Count missing values in each column.
println("\nMissing values in conditions:")
for column_name in names(conditions)
    println(column_name, ": ", count(ismissing, conditions[!, column_name]))
end

println("\nMissing values in time-series:")
for column_name in names(timeseries)
    println(column_name, ": ", count(ismissing, timeseries[!, column_name]))
end

println("\nMissing values in paper split:")
for column_name in names(paper_split)
    println(column_name, ": ", count(ismissing, paper_split[!, column_name]))
end

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Add experimental conditions to every time-series row.
model_data = leftjoin(
    timeseries,
    conditions,
    on = [:experiment_id, :paper_id]
)

# Add dataset role and cross-validation fold to every row.
model_data = leftjoin(
    model_data,
    paper_split,
    on = :paper_id
)

# Sort the combined data by paper, experiment, and time.
sort!(
    model_data,
    [:paper_id, :experiment_id, :day]
)

# Display the size and column names of the combined data.
println("\nCombined model data size: ", size(model_data))
println("Combined model data columns:")
println(names(model_data))