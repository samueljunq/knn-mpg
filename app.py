import numpy as np
import pickle
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder  # Importar para codificação de rótulos

app = Flask(__name__)
CORS(app)

# Carregar modelo e scaler para normalização
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Inicializar LabelEncoder para variáveis categóricas
le_origin = LabelEncoder()

# Supomos que você tem essas categorias possíveis para 'origin'
le_origin.fit(["USA", "Europe", "Japan"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_values = list(request.form.values())
        print(f"Valores do formulário: {form_values}")

        # Remover 'origin' antes de processar os dados
        features = [float(x) for x in form_values[:-1]]  # Remove o último valor ('USA')

        print(f"Features enviadas ao modelo: {features}")

        # Normalizar os dados (o scaler foi treinado com 6 features)
        final_features = scaler.transform([features])

        # Fazer a previsão
        pred = model.predict(final_features)
        output = round(pred[0], 2)
        print(f"Previsão: {output}")

        return render_template("index.html", prediction_text=f"MPG previsto: {output}")

    except Exception as e:
        print("Erro durante a previsão:", str(e))
        return render_template("index.html", prediction_text="Erro na previsão!")

if __name__ == "__main__":
    app.run(debug=True)
