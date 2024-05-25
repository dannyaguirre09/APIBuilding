
from flask import Flask, json, jsonify, request, make_response
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys
import io

app = Flask(__name__)

longitud, altura = 150, 150
modelo = 'modelo.h5'
pesos = 'pesos.h5'
try:
    cnn = load_model(modelo) 
    cnn.load_weights(pesos) 
    print("Modelo y pesos cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo o los pesos: {e}")

@app.route('/')
def Home():  
  return 'API to detect crash in buildings '

@app.route('/predict', methods=['POST'] )
def Predict():
  try:
    file = request.files['file'].read()
    image = Image.open(io.BytesIO(file))    
    resizedImage = image.resize((longitud, altura))
    x = np.expand_dims(resizedImage, axis=0)   
    predictions = cnn.predict(x)  
    predicted_class = np.argmax(predictions, axis=1) 
    class_names = ['COLLAPSE', 'HEALTHY', 'PARTIAL']
    predicted_label = class_names[predicted_class[0]]
    res = jsonify(result=predicted_label, status = True)
    return make_response(res, 200)
  except :
    res = jsonify(status = False, message = str(sys.exc_info() ) )
    return make_response(res, 500)
 

if __name__ == '__main__':
  app.run(debug=True) 
