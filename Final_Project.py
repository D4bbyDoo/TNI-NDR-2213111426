import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# โหลดข้อมูล
df = pd.read_excel(r"Coca-Cola_6Mdaily.xlsx")

# ตั้งชื่อคอลัมน์ใหม่
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change%", "NYSE Index"]

# แปลงคอลัมน์ Date เป็น datetime และเก็บเฉพาะวันที่ (ลบเวลา)
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce").dt.date

# ตั้ง index ให้เริ่มที่ 1 สำหรับข้อมูลทั้งหมด
df.index = range(1, len(df) + 1)

# ลบแถวที่มี NaN
df = df.dropna()

# เพิ่ม CSS เพื่อปรับขนาดตารางและสไตล์กล่องราคาสูงสุด/ต่ำสุด
st.markdown(
    """
    <style>
    .dataframe {
        width: 900px !important;  # ปรับให้เท่ากับความกว้างของกราฟ (12 นิ้ว * 72 พิกเซล)
        overflow-x: auto;
        margin: 0 auto;  # จัดกึ่งกลาง
    }
    .max-box {
        background-color: #e6ffed;  # สีเขียวอ่อนสำหรับราคาสูงสุด
        padding: 10px;  # ลด padding ให้เล็กลง
        border-radius: 5px;
        text-align: center;
        margin-right: 10px;
        font-size: 14px;  # ลดขนาดตัวอักษร
    }
    .min-box {
        background-color: #ffe6e6;  # สีแดงอ่อนสำหรับราคาต่ำสุด
        padding: 10px;  # ลด padding ให้เล็กลง
        border-radius: 5px;
        text-align: center;
        margin-left: 10px;
        font-size: 14px;  # ลดขนาดตัวอักษร
    }
    h4 { font-size: 16px; }  # ปรับขนาดหัวข้อ
    h2 { font-size: 20px; }  # ปรับขนาดราคา
    p { font-size: 12px; }   # ปรับขนาดข้อความวันที่และ % เปลี่ยนแปลง
    .text {
        font-size: 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# สร้าง dropdown menu สำหรับเลือกช่วงเวลา
st.title("📊 วิเคราะห์ข้อมูลหุ้น Coca-Cola")

# เพิ่มข้อความสีแดงด้านบนสุด
st.markdown(
    '<p class="text">***ข้อมูลทั้งหมดนี้จะใช้เป็นสกุลเงิน USD (US Dollars)</p>',
    unsafe_allow_html=True
)

time_period = st.selectbox("เลือกช่วงเวลาย้อนหลัง:", ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months"])

# กำหนดจำนวนวันและกรองข้อมูลตามตัวเลือก
if time_period == "1 Week":
    df_filtered = df.head(7)  
elif time_period == "2 Weeks":
    df_filtered = df.head(14)
elif time_period == "1 Month":
    df_filtered = df.head(30)    
elif time_period == "3 Months":
    df_filtered = df.head(90)
else:  # 6 เดือน
    df_filtered = df.head(180)  # ประมาณ 6 เดือน (30 วัน x 6)

# รีเซ็ต index ให้เริ่มจาก 1 สำหรับข้อมูลที่กรอง
df_filtered.index = range(1, len(df_filtered) + 1)

# สร้างกราฟ
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
df_sorted = df_filtered.sort_values("Date")

# แปลง Date เป็น ordinal สำหรับ Linear Regression
X = pd.to_datetime(df_sorted["Date"]).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = df_sorted["Price"].values

# ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับการฟิตโมเดล
if len(X) > 0:
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)

    # แสดงกราฟก่อนตาราง
    st.subheader(f"กราฟ {time_period} ย้อนหลังของหุ้น Coca-Cola")
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df_sorted["Date"]), y, label="Actual Closing Price")
    plt.plot(pd.to_datetime(df_sorted["Date"]), trend, label="Trend (Linear Regression)", linestyle="--", color="red")
    plt.title(f"Coca-Cola LineChart ({time_period})")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # แสดงกราฟใน Streamlit
    st.pyplot(plt)
else:
    st.warning("ไม่มีข้อมูลสำหรับช่วงเวลาที่เลือก")

# คำนวณราคาสูงสุดและต่ำสุดในช่วงเวลาที่เลือก
max_price_row = df_filtered[df_filtered["Price"] == df_filtered["Price"].max()]
min_price_row = df_filtered[df_filtered["Price"] == df_filtered["Price"].min()]

max_price = max_price_row["Price"].iloc[0]
max_date = max_price_row["Date"].iloc[0]
max_change = max_price_row["Change%"].iloc[0]

min_price = min_price_row["Price"].iloc[0]
min_date = min_price_row["Date"].iloc[0]
min_change = min_price_row["Change%"].iloc[0]

# แสดงราคาสูงสุดและต่ำสุดในกล่องสองกล่อง (ใต้กราฟ)
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="max-box">
            <h4>สูงสุดในรอบ {time_period}</h4>
            <h2>${max_price:.2f}</h2>
            <p>วันที่ {max_date}</p>
            <p>เปลี่ยนแปลง {max_change:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="min-box">
            <h4>ต่ำสุดในรอบ {time_period}</h4>
            <h2>${min_price:.2f}</h2>
            <p>วันที่ {min_date}</p>
            <p>เปลี่ยนแปลง {min_change:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# แสดงตารางข้อมูลตามช่วงเวลาที่เลือก
st.subheader(f"ข้อมูลย้อนหลัง {time_period}")
st.dataframe(df_filtered)

# แสดงสถิติที่ด้านล่างสุด
st.subheader("สถิติข้อมูลหุ้น Coca-Cola (ข้อมูลทั้งหมด)")
st.write(df["Price"].describe())
