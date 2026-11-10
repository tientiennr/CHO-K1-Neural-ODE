# โหลดสมการ Neural ODE และเตรียมชุด independent test
include(joinpath(@__DIR__, "06_define_loss.jl"))

# โหลดโมดูลของ Julia สำหรับอ่านและเขียนข้อมูลที่บันทึกแบบ serialized
using Serialization

# โหลดพารามิเตอร์และค่า scaling ของ final model
final_model = deserialize(joinpath(@__DIR__, "final_neural_ode.jls"))

# ตรวจว่าข้อมูลถูกสเกลด้วยค่าเดียวกับที่ใช้ตอนฝึกโมเดล
if final_model.means != (vcc_mean, glucose_mean, titer_mean) ||
   final_model.stds != (vcc_std, glucose_std, titer_std)
    error("Scaling differs from the saved model. Check the input data and use argument 0.")
end

variable_names = ("VCC", "Glucose", "Titer")

# สร้างตารางเปล่าชื่อ results เพื่อทยอยเพิ่มผลทีละแถว
results = DataFrame(
    experiment_id = String[],
    day = Float32[],
    variable = String[],
    actual = Float64[],
    predicted = Float64[],
    is_initial_time = Bool[]
)

# วนผ่านการทดลองใน independent test ทีละชุด โดย tr คือ trajectory ของการทดลองที่กำลังประมวลผล
# ใช้พารามิเตอร์ที่ฝึกแล้วแก้สมการ ODE สำหรับการทดลองนี้ ได้เมทริกซ์ค่าทำนายบนสเกล z-score โดย แถวคือตัวแปร และ คอลัมน์คือเวลา
for tr in independent_test_model_trajectories
    prediction = predict_trajectory(final_model.parameters, tr)

# วนทีละตัวแปรตามลำดับ VCC, glucose และ titer
# tr.observation_mask[i, :] เลือกทุกเวลาของตัวแปร i และ find all คืนตำแหน่งที่มีค่าที่วัดจริง เช่น [1, 3, 5] จึงวนเฉพาะตำแหน่งเหล่านั้น j คือตำแหน่งในอาร์เรย์ ส่วนวันจริงอ่านจาก tr.time[j]
# ดึงค่าที่วัดจริงของตัวแปร i ณ ตำแหน่งเวลา j แล้วแปลงจาก z-score กลับเป็นหน่วยเดิมด้วยสูตร z × SD + mean
# แปลงค่าทำนายของ ตัวแปรเดียวกัน ณ เวลาเดียวกัน ด้วย mean และ SD ชุดเดียวกัน จึงเทียบกับ actual ได้โดยตรง
# เพิ่มหนึ่งแถวลงใน results ! ใน push! บอกว่าฟังก์ชันแก้ตารางเดิมโดยตรง ใส่รหัสการทดลองลงคอลัมน์แรก
# ได้ true เมื่อเป็นตำแหน่งเวลาแรกของ trajectory แม้เวลาเริ่มจริงจะเป็น Day 1 หรือ Day 2 ก็ตาม
    for i in 1:3
        for j in findall(tr.observation_mask[i, :])
            actual = tr.observations[i, j] * final_model.stds[i] + final_model.means[i]
            predicted = prediction[i, j] * final_model.stds[i] + final_model.means[i]

            push!(results, (
                tr.experiment_id,
                tr.time[j],
                variable_names[i],
                actual,
                predicted,
                j == 1
            ))
        end
    end
end

# บันทึกค่าจริงและค่าทำนายทุกจุดที่มีการวัด
# กำหนดให้ไฟล์ผลอยู่ในโฟลเดอร์เดียวกับ 09 เขียนตารางทั้งหมดลง CSV รวมเวลาเริ่มต้นที่มีค่าที่วัดจริง
output_path = joinpath(@__DIR__, "independent_test_predictions.csv")
CSV.write(output_path, results)

# คำนวณ RMSE แยกตัวแปร เฉพาะเวลาหลังจุดเริ่มจำลอง
# เลือกแถวที่ ตรงกับตัวแปรนั้น และอยู่หลังเวลาเริ่มจำลอง โดย .== เทียบทีละแถว .& คือ “และ” ทีละแถว .! กลับค่า true กับ false ทีละแถว
# คำนวณ ค่าทำนายลบค่าจริง ทีละคู่ ถ้าผลเป็นบวกแปลว่าทำนายสูงเกิน ถ้าเป็นลบแปลว่าทำนายต่ำเกิน
# ตรวจว่าตัวแปรนี้มีจุดที่ใช้ประเมินหลังเวลาเริ่มต้นหรือไม่ ถ้ารายการว่าง ให้พิมพ์ว่าไม่มีจุดที่ใช้ประเมิน ถ้ามีข้อมูล ให้คำนวณ RMSE พิมพ์ชื่อตัวแปร จำนวนจุดที่ใช้ และ RMSE โดยปัดเป็น 4 ตำแหน่ง
for name in variable_names
    rows = (results.variable .== name) .& .!results.is_initial_time
    errors = results.predicted[rows] .- results.actual[rows]

    if isempty(errors)
        println(name, " | No measured points after initialization")
    else
        rmse = sqrt(mean(errors .^ 2))
        println(name, " | Points: ", length(errors), " | RMSE: ", round(rmse; digits=4))
    end
end

println("Saved predictions: ", output_path)