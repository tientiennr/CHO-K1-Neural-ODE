# Load the combined model data from the first script.
include(joinpath(@__DIR__, "01_load_and_check_data.jl"))

# Load statistical functions used to calculate means and standard deviations.
using Statistics

# Select the cross-validation fold used as the validation set.
# ARGS รายการค่าที่เราพิมพ์ต่อท้ายชื่อไฟล์ตอนสั่งรัน เช่น สั่ง julia 07_train_neural_ode.jl 2 จะได้ ARGS = ["2"] isempty(ARGS) ตรวจว่ารายการนี้ว่างหรือไม่? 1 :	ถ้าว่าง ให้ใช้เลข 1 เป็นค่าเริ่มต้น parse(Int, ARGS[1])	ถ้ามีค่า ให้หยิบค่าแรกแล้วแปลงจากข้อความ เช่น "2" เป็นจำนวนเต็ม 2
validation_fold = isempty(ARGS) ? 1 : parse(Int, ARGS[1])

# Select development data from the folds not used for validation.
# เลือกเฉพาะข้อมูลใน development set
# เลือกเฉพาะแถวที่มีหมายเลข fold เพื่อป้องกันไม่ให้ independent test ซึ่งมี cv_fold = missing ปะปนเข้ามา
# เลือก fold ที่ไม่ใช่ fold 1 เป็น training
# .& หมายความว่าทุกเงื่อนไขต้องเป็นจริงพร้อมกัน : หมายถึงเก็บทุกคอลัมน์ของแถวที่เลือก
training_data = model_data[
    (model_data.dataset_role .== "development") .&
    (.!ismissing.(model_data.cv_fold)) .&
    (model_data.cv_fold .!= validation_fold),
    :
]

# Select development data from the fold used for validation.
validation_data = model_data[
    (model_data.dataset_role .== "development") .&
    (.!ismissing.(model_data.cv_fold)) .&
    (model_data.cv_fold .== validation_fold),
    :
]

# Keep the independent test data separate from model development.
independent_test_data = model_data[
    model_data.dataset_role .== "independent test",
    :
]

# Group rows belonging to the same experiment into trajectories.
# รวมทุก time point ของ experiment เดียวกันเป็นหนึ่ง trajectory โดย paper เดียวกันจะไม่ถูกแบ่งข้าม training และ validation เพราะ cv_fold ถูกกำหนดในระดับ paper แล้ว
training_trajectories = groupby(
    training_data,
    [:paper_id, :experiment_id]
)

validation_trajectories = groupby(
    validation_data,
    [:paper_id, :experiment_id]
)

independent_test_trajectories = groupby(
    independent_test_data,
    [:paper_id, :experiment_id]
)

# Display the selected validation fold.
println("\nValidation fold: ", validation_fold)

# Display the number of papers and trajectories in the training set.
println("\nTraining set:")
println(
    "Papers: ",
    length(unique(training_data.paper_id))
)
println(
    "Trajectories: ",
    length(training_trajectories)
)

# Display the number of papers and trajectories in the validation set.
println("\nValidation set:")
println(
    "Papers: ",
    length(unique(validation_data.paper_id))
)
println(
    "Trajectories: ",
    length(validation_trajectories)
)

# Display the number of papers and trajectories in the independent test set.
println("\nIndependent test set:")
println(
    "Papers: ",
    length(unique(independent_test_data.paper_id))
)
println(
    "Trajectories: ",
    length(independent_test_trajectories)
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Calculate scaling parameters from the training set only. เพื่อไม่ให้ข้อมูลจาก validation หรือ test รั่วเข้าสู่การฝึกโมเดล
# training_data.vcc เลือกข้อมูลในคอลัมน์ vcc ของ training set skipmissing(...) ข้ามช่องว่างหรือ missing mean(...) คำนวณค่าเฉลี่ย std(...) คำนวณค่า standard deviation vcc_mean = เก็บผลลัพธ์ไว้ในตัวแปรชื่อ vcc_mean
vcc_mean = mean(skipmissing(training_data.vcc))
vcc_std = std(skipmissing(training_data.vcc))

glucose_mean = mean(skipmissing(training_data.glucose))
glucose_std = std(skipmissing(training_data.glucose))

titer_mean = mean(skipmissing(training_data.titer))
titer_std = std(skipmissing(training_data.titer))

# Display the scaling parameters.
println("\nScaling parameters calculated from the training set:")

println("VCC mean: ", vcc_mean)
println("VCC standard deviation: ", vcc_std)

println("Glucose mean: ", glucose_mean)
println("Glucose standard deviation: ", glucose_std)

println("Titer mean: ", titer_mean)
println("Titer standard deviation: ", titer_std)

# Scale the training data using parameters calculated from the training set.
# training_data.vcc_scaled สร้างคอลัมน์ vcc_scaled ที่อยู่ในตาราง training_data
# .- และ ./ หมายถึงให้คำนวณกับข้อมูลทุกแถวในคอลัมน์ และหากแถวใดเป็น missing ผลลัพธ์ของแถวนั้นจะยังเป็น missing
training_data.vcc_scaled =
    (training_data.vcc .- vcc_mean) ./ vcc_std

training_data.glucose_scaled =
    (training_data.glucose .- glucose_mean) ./ glucose_std

training_data.titer_scaled =
    (training_data.titer .- titer_mean) ./ titer_std

# Scale the validation data using the same training parameters.
validation_data.vcc_scaled =
    (validation_data.vcc .- vcc_mean) ./ vcc_std

validation_data.glucose_scaled =
    (validation_data.glucose .- glucose_mean) ./ glucose_std

validation_data.titer_scaled =
    (validation_data.titer .- titer_mean) ./ titer_std

# Scale the independent test data using the same training parameters.
independent_test_data.vcc_scaled =
    (independent_test_data.vcc .- vcc_mean) ./ vcc_std

independent_test_data.glucose_scaled =
    (independent_test_data.glucose .- glucose_mean) ./ glucose_std

independent_test_data.titer_scaled =
    (independent_test_data.titer .- titer_mean) ./ titer_std

# Check the scaled training data.
println("\nScaled training data check:")
println(
    "VCC mean: ",
    mean(skipmissing(training_data.vcc_scaled))
)
println(
    "VCC standard deviation: ",
    std(skipmissing(training_data.vcc_scaled))
)
println(
    "Glucose mean: ",
    mean(skipmissing(training_data.glucose_scaled))
)
println(
    "Glucose standard deviation: ",
    std(skipmissing(training_data.glucose_scaled))
)
println(
    "Titer mean: ",
    mean(skipmissing(training_data.titer_scaled))
)
println(
    "Titer standard deviation: ",
    std(skipmissing(training_data.titer_scaled))
)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Convert a DataFrame into trajectories that can be used by the model.
# สร้างตัวแปรชื่อ prepared_trajectories เพื่อเก็บ trajectory ที่เตรียมเสร็จแล้ว
# NamedTuple คือชุดข้อมูลที่แต่ละส่วนมีชื่อ เช่น: (
#    paper_id = "P020",
#    experiment_id = "P020_E01",
#    time = [0.0, 1.0, 2.0]
#) [] หมายถึงเริ่มต้นด้วยรายการว่างก่อน ค่อยเพิ่ม trajectory ทีละรายการ
# groupby() ใช้แบ่งแถวใน DataFrame ตามค่าของคอลัมน์ที่กำหนด
function prepare_model_trajectories(data)

    prepared_trajectories = NamedTuple[]

    grouped_data = groupby(
        data,
        [:paper_id, :experiment_id]
    )
# for หมายถึงทำขั้นตอนต่อไปนี้ซ้ำกับทุกกลุ่ม แต่ละรอบ trajectory_group (เป็นชื่อที่เราตั้งเอง) จะหมายถึงข้อมูลหนึ่ง experiment เท่านั้น
# เปลี่ยนกลุ่มเป็น DataFrame และเรียงแถวตามคอลัมน์ day จากน้อยไปมาก แม้หน้า 1 จะเรียงข้อมูลมาแล้ว การเรียงซ้ำตรงนี้เป็นการป้องกันว่าแต่ละ trajectory อยู่ตามลำดับเวลาก่อนส่งเข้า ODE solver
    for trajectory_group in grouped_data

        trajectory = sort(
            DataFrame(trajectory_group),
            :day
        )

        has_observation = any(.!ismissing.(trajectory.vcc_scaled)) ||
                          any(.!ismissing.(trajectory.glucose_scaled)) ||
                          any(.!ismissing.(trajectory.titer_scaled))
# has_observation ตรวจว่าการทดลองนี้มีค่าที่วัดจริงอย่างน้อยหนึ่งค่าจากตัวแปรใดก็ได้ if ... continue ข้ามเฉพาะการทดลองที่มีเวลาเพียงจุดเดียว หรือไม่มีค่าที่วัดจริงเลย การทดลองที่มีหลายเวลาและมีข้อมูลเพียง VCC หรือ glucose ก็ผ่านเงื่อนไขนี้ได้ เมื่อลบ start_index และการตัดแถวออก โมเดลจะเริ่มจากเวลาแรกของการทดลอง จึงเก็บค่าที่วัดได้ในวันก่อนหน้าที่เคยถูกตัดทิ้งไว้ด้วย || หรือ
        if first(trajectory.day) == last(trajectory.day) || !has_observation
            continue
        end

# หาวันแรกที่มี initial state ครบสามค่า findfirst() ใช้ค้นหาตำแหน่งแรกที่ตรงตามเงื่อนไข i หมายถึงหมายเลขแถวที่กำลังตรวจสอบ && หมายถึง “และ” ดังนั้นทั้งสามเงื่อนไขต้องเป็นจริงพร้อมกัน
# eachindex(trajectory.day) สร้างลำดับหมายเลขแถวของคอลัมน์ day Julia จะสร้างลำดับ “ตำแหน่ง” ที่ใช้เข้าถึงสมาชิกในคอลัมน์ day: ไม่ได้ส่งค่าของวัน 0, 1, 2, 3, 4 เข้าไปตรวจ แต่ส่งหมายเลขตำแหน่งแถว 1, 2, 3, 4, 5 จากนั้น findfirst() ตรวจทีละตำแหน่ง
#        start_index = findfirst(
#            i ->
#                !ismissing(trajectory.vcc_scaled[i]) &&
#                !ismissing(trajectory.glucose_scaled[i]) &&
#                !ismissing(trajectory.titer_scaled[i]),
#            eachindex(trajectory.day)
#        )
# หากไม่พบ ให้ข้าม trajectory นี้ และไปทำ trajectory ถัดไปทันที
#        if start_index === nothing
#            continue
#        end
# start_index:end เลือกแถวตั้งแต่จุดเริ่มต้นจนถึงแถวสุดท้าย : เลือกทุกคอลัมน์ หากข้อมูลมี Day 0–14 แต่ initial state ครบครั้งแรกที่ Day 1 ผลลัพธ์จะเหลือ Day 1–14 ค่า day ยังเป็นวันจริง ไม่ได้เปลี่ยน Day 1 ให้เป็น Day 0
#        trajectory = trajectory[start_index:end, :]
# nrow(trajectory) นับจำนวนแถว หากเหลือน้อยกว่า 2 แถว จะไม่มีข้อมูลการเปลี่ยนแปลงตามเวลาให้ ODE เรียนรู้
#        if nrow(trajectory) < 2
#            continue
#        end

# นำค่าจากคอลัมน์ day มาเก็บใน time_points Float64.(...) แปลงค่าทุกตัวเป็นเลขทศนิยม Neural ODE และ ODE solver ทำงานกับเลขทศนิยมได้สะดวกและสม่ำเสมอกว่า
        time_points = Float32.(trajectory.day)
# observation_mask = permutedims( สร้างเมทริกซ์ที่บอกว่าตำแหน่งใดมีข้อมูลจริง permutedims() สลับแถวกับคอลัมน์ hcat() นำคอลัมน์ต่าง ๆ มาวางต่อกันในแนวนอน จุดหน้า ! หมายถึงให้ดำเนินการกับทุกค่าในคอลัมน์
        observation_mask = permutedims(
            hcat(
                .!ismissing.(trajectory.vcc_scaled),
                .!ismissing.(trajectory.glucose_scaled),
                .!ismissing.(trajectory.titer_scaled)
            )
        )
# สร้างเมทริกซ์ค่าที่สังเกตได้ เริ่มสร้างเมทริกซ์ข้อมูลจริงหลัง scaling hcat( วางข้อมูลสามตัวแปรต่อกัน coalesce.(..., 0.0) หากมีค่าจริง ให้ใช้ค่าจริง หากเป็น missing ให้ใส่ 0.0 ชั่วคราว
# ค่า 0.0 นี้ไม่ใช่การตีความว่าค่าจริงเท่ากับศูนย์ แต่ใช้เพื่อให้เมทริกซ์มีชนิดข้อมูลเป็นตัวเลขทั้งหมด โดย observation_mask จะกันไม่ให้ตำแหน่งที่ไม่มีข้อมูลถูกนำไปคำนวณ loss
# hcat() รวมคอลัมน์ และ permutedims() จัดรูปแบบเป็น 3 แถวตาม state variables Float32.(...) หมายถึงแปลงตัวเลขทุกค่าในวงเล็บให้เป็น Float32
        observations = Float32.(
            permutedims(
                 hcat(
                    coalesce.(trajectory.vcc_scaled, 0.0),
                    coalesce.(trajectory.glucose_scaled, 0.0),
                    coalesce.(trajectory.titer_scaled, 0.0)
                 )
            )
        )
# กำหนด initial state : เลือกทุกแถว ซึ่งก็คือทั้งสาม state 1 เลือกคอลัมน์แรก ซึ่งคือวันเริ่มต้น
        initial_state = observations[:, 1]
# push!() เพิ่มข้อมูลหนึ่งรายการเข้าไปใน prepared_trajectories ! หมายถึงฟังก์ชันนี้เปลี่ยนแปลงตัวแปรเดิมโดยตรง
# เก็บ paper_id ของ trajectory โดยใช้ค่าจากแถวแรก เก็บวันจริงทั้งหมดของ trajectory เก็บค่าเริ่มต้นสาม state เก็บเมทริกซ์ค่าที่วัดได้หลัง scaling เก็บเมทริกซ์ที่ระบุว่าตำแหน่งใดมีข้อมูลจริง
        push!(
            prepared_trajectories,
            (
                paper_id = trajectory.paper_id[1],
                experiment_id = trajectory.experiment_id[1],
                time = time_points,
                initial_state = initial_state,
                observations = observations,
                observation_mask = observation_mask
            )
        )
    end
# จบการทำงานของ trajectory ปัจจุบัน แล้วกลับไปทำกลุ่มถัดไป
    return prepared_trajectories
end

# Prepare the training, validation, and independent test trajectories.
training_model_trajectories =
    prepare_model_trajectories(training_data)
# ส่ง training_data เข้าไปในฟังก์ชัน และเก็บผลลัพธ์ไว้ใน training_model_trajectories ข้อมูลชุดนี้จะใช้ปรับพารามิเตอร์ของ Neural ODE
validation_model_trajectories =
    prepare_model_trajectories(validation_data)
# ข้อมูลชุดนี้ใช้ประเมินโมเดลระหว่างการพัฒนา แต่ไม่ใช้ปรับน้ำหนักโดยตรง
independent_test_model_trajectories =
    prepare_model_trajectories(independent_test_data)
# ข้อมูลนี้จะเก็บไว้ประเมินโมเดลครั้งสุดท้าย หลังจากเลือกโครงสร้างและวิธีฝึกเสร็จแล้ว

# Display the number of prepared trajectories.
println("\nPrepared model trajectories:")
println(
    "Training: ",
    length(training_model_trajectories)
)
println(
    "Validation: ",
    length(validation_model_trajectories)
)
println(
    "Independent test: ",
    length(independent_test_model_trajectories)
)


open(joinpath(@__DIR__, "fold_$(validation_fold)_split.txt"), "w") do io
    for (label, trajectories) in (
        ("train", training_model_trajectories),
        ("validation", validation_model_trajectories)
    )
        entries = sort([
            "$(tr.paper_id) | $(tr.experiment_id)"
            for tr in trajectories
        ])
        for entry in entries
            println(io, label, " | ", entry)
        end
    end
end