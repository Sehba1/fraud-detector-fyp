from app.models.xai import explain_prediction

text = "urgent hiring work from home earn 500 dollars daily send bank account details and passport copy to apply immediately limited positions"
result = explain_prediction(text)
print("label:", result["label"])
print("fraud_probability:", result["fraud_probability"])
print("local_fidelity:", result["local_fidelity"])
