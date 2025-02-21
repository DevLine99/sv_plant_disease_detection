## Setup

1. Install the required packages:

```bash
$ pip install -r requirements.txt
```

2. Prepare your dataset in the `data/` directory.

3. Train the model:

```bash
$ python src/model_training.py
```

4. Run the prediction API:

```bash
$ python src/predict_api.py
```

## Usage

To use the prediction API, send a POST request to `/predict` with an image file. The API will return the predicted disease label.

ลบ cash gitignore

```bash
$ git rm --cached models\resnet50_finetuned_model.h5
$ git rm --cached data -r
```

หรือ

```bash
git rm -r --cached .
git add .
git commit -am 'git cache cleared'
git push
```

การย้อนกลับการกระทำครั้งล่าสุด

```bash
git reset HEAD^ --hard
```

---

# Virtual Environment

```bash
python -m venv venv  # สร้าง virtual environment
source venv/bin/activate  # สำหรับ macOS/Linux
venv\Scripts\activate  # สำหรับ Windows
```

- เช็ค Python ว่ากำลังใช้จาก Virtual Environment หรือไม่

```bash
python -c "import sys; print(sys.prefix)"
```

- ถ้า Virtual Environment ถูกเปิดอยู่ มันควรจะแสดง path ของ env เช่น

```bash
C:\Users\YourUser\YourProject\env
```

---

## Execution Policy ของ PowerShell ที่ไม่อนุญาตให้เรียกใช้สคริปต์ (.ps1)

- เปิด PowerShell ด้วยสิทธิ์ Admin
- ตรวจสอบ Execution Policy พิมพ์คำสั่งนี้เพื่อตรวจสอบค่า Policy ปัจจุบัน

```bash
Get-ExecutionPolicy
```

- เปลี่ยน Execution Policy ใช้คำสั่งนี้เพื่ออนุญาตให้รันสคริปต์:

```bash
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

- ลอง Activate Virtual Environment อีกครั้ง

```bash
<ชื่อ>\Scripts\Activate
```

- หรือหากใช้ WSL หรือ macOS/Linux:

```bash
source env/bin/activate
```

## ยกเลิก หรือ ตั้งค่า Execution Policy กลับเป็นค่าเริ่มต้น

- ตั้งค่าเป็นค่าเริ่มต้นของ Windows

```bash
Set-ExecutionPolicy Restricted -Scope CurrentUser
```

---

# Run project

```bash
uvicorn main:app --reload
```
