import pickle

with open("models/label_encoder.pkl","rb") as f:
    le = pickle.load(f)

print(le.classes_)