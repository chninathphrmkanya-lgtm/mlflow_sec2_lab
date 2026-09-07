import mlflow
from sklearn.datasets import load_breast_cancer  # [จุดที่ 1] เปลี่ยนจาก load_wine เป็น load_breast_cancer


def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "cancer-classifier-prod"  # [จุดที่ 1] เปลี่ยนชื่อโมเดลให้ตรงกับสคริปต์ขั้นที่ 3
    MODEL_ALIAS = "staging"  # MLflow 3 ใช้ Alias แทน Stage เดิม

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # Load the model from the Model Registry ด้วย Alias URI
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI.")
        return

    # [จุดที่ 1 & 2] โหลดข้อมูล Breast Cancer และดึงมา คลาสละ 1 ตัวอย่าง (รวมเป็น 2 ตัวอย่าง)
    cancer_data = load_breast_cancer(as_frame=True)
    X, y = cancer_data.data, cancer_data.target
    target_names = cancer_data.target_names  # ['malignant', 'benign']

    # หาดัชนีของคลาส 0 (Malignant) และ คลาส 1 (Benign) อย่างละ 1 ตัวอย่าง
    idx_class_0 = (y == 0).idxmax()
    idx_class_1 = (y == 1).idxmax()

    sample_data = X.loc[[idx_class_0, idx_class_1]]
    actual_labels = y.loc[[idx_class_0, idx_class_1]].values

    # Use the loaded model to make predictions
    predictions = model.predict(sample_data)

    # [จุดที่ 3] แสดงผลลัพธ์พร้อมแปลงชื่อคลาสให้อ่านง่าย
    print("-" * 50)
    for i in range(2):
        act_str = target_names[actual_labels[i]]
        pred_str = target_names[predictions[i]]
        status = "Correct" if actual_labels[i] == predictions[i] else "Incorrect"
        print(f"Sample {i+1}: Actual = {act_str} | Predicted = {pred_str} [{status}]")
    print("-" * 50)


if __name__ == "__main__":
    load_and_predict()