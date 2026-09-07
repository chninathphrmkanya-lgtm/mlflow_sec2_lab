import mlflow
from sklearn.datasets import load_breast_cancer  # 1. เปลี่ยนเป็น Breast Cancer[cite: 2]


def validate_data():
    """
    Loads the breast cancer dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    # Set the experiment name for this step
    mlflow.set_experiment("Breast Cancer - Data Validation")  # 2. เปลี่ยนชื่อ Experiment[cite: 2]

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data as a Pandas DataFrame
        cancer_data = load_breast_cancer(as_frame=True)  # 3. เปลี่ยนตัวแปรโหลดข้อมูล[cite: 2]
        df = cancer_data.frame
        print("Data loaded successfully.")

        # 2. Perform simple validation checks
        num_rows, num_cols = df.shape
        num_classes = df['target'].nunique()
        missing_values = df.isnull().sum().sum()

        # เพิ่มคำนวณสัดส่วนคลาสน้อยสุดตามโจทย์[cite: 2]
        min_class_ratio = df['target'].value_counts(normalize=True).min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Min class ratio: {min_class_ratio:.4f}")

        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", min_class_ratio)  # เพิ่ม log class_balance ตามโจทย์[cite: 2]
        mlflow.log_param("num_classes", num_classes)

        # Check if the data passes our defined criteria
        validation_status = "Success"
        # 4. แก้เงื่อนไข: เปลี่ยนจาก < 3 เป็น != 2 และเช็คสัดส่วนคลาส < 0.20 ตามโจทย์[cite: 2]
        if missing_values > 0 or num_classes != 2 or min_class_ratio < 0.20:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # 4. ทำให้ CI จับได้จริง
        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()