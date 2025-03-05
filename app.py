import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.efficientnet import preprocess_input

# Charger les modèles (remplacez les chemins par vos fichiers de modèles)
model1 = tf.keras.models.load_model('model_v2_Q1.keras')
model2 = tf.keras.models.load_model('model_v2_Q2.keras')
model3 = tf.keras.models.load_model('model_Q3.keras')
model4 = tf.keras.models.load_model('model_Q4.keras')


# Gère le formattage des images
def format_image_type1(image):
    image = image.resize((64, 64)) # Redimensionner selon la taille attendue
    image = np.array(image) / 255.0 # Normalisation
    return image_single_fix(image)

def format_image_type2(image):
    image = ImageOps.grayscale(image) # Charge en noir et blanc mais avec une meilleure résolution
    image = image.resize((128, 128), Image.LANCZOS) # Redimensionner selon la taille attendue
    image = np.array(image) / 255.0 # Normalisation
    return image_single_fix(image)

def format_image_type3(image):
    image = tf.image.resize(image, (128, 128)) # Redimensionner selon la taille attendue
    image = preprocess_input(image) # Normalisation
    return image_single_fix(image)

def image_single_fix(image_array):
    return np.expand_dims(image_array, axis=0) # Ajouter batch dimension (demandé par tensorflow/keras qui demande à recevoir un groupe d'image plutôt que juste une seule)


# Configuration des infos des modèles
models = {
    "Modèle 1 (genre uniquement)": {
        "model": model1,
        "type": "gender_only",
        "image_format": format_image_type1
    },
    "Modèle 2 (âge uniquement)": {
        "model": model2,
        "type": "age_only",
        "image_format": format_image_type1
    },
    "Modèle 3 (genre et âge)": {
        "model": model3,
        "type": "gender_age",
        "image_format": format_image_type2
    },
    "Modèle 4 (genre et âge) [Transfert d'apprentissage]": {
        "model": model4,
        "type": "age_gender",
        "image_format": format_image_type3
    }
}


# Fonction de prédiction
def predict(image, model_name):
    model_data = models[model_name]
    
    image_array = model_data["image_format"](image) # Formatter l'image de la façon qui correspond
    
    model = model_data["model"]
    model_type = model_data["type"]
    
    prediction = model.predict(image_array)
    
    return_text = ""
    if model_type == "gender_only":
        gender_value = prediction[0][0]
        return_text = display_gender_prediction(gender_value)
    elif model_type == "age_only":
        age_value = prediction[0][0]
        return_text = display_age_prediction(age_value)
    elif model_type == "gender_age":
        gender_value = prediction[0][0][0]
        age_value = prediction[1][0][0]
        return_text = display_gender_prediction(gender_value) + "\n" + display_age_prediction(age_value)
    elif model_type == "age_gender": # gender_age but prediction data order is reversed
        gender_value = prediction[1][0][0]
        age_value = prediction[0][0][0]
        return_text = display_gender_prediction(gender_value) + "\n" + display_age_prediction(age_value)
    else:
        raise Exception(f"Unsupported model_type '{model_type}'")
    
    return f"{return_text}"


# Fonctions utiles
def get_gender_confidence(gender_value) -> str:
    return f"{round(abs(gender_value - 0.5) * 2 * 100)}%"

def display_gender_prediction(gender_value) -> str:
    gender = "Homme" if gender_value < 0.5 else "Femme" # Genre basé sur probabilité
    rounded_gender_value_str = str(round(gender_value, 4))
    return f"Genre: {gender} ({rounded_gender_value_str} - {get_gender_confidence(gender_value)} certitude)"

def display_age_prediction(age_value) -> str:
    age = round(age_value) # Arroundi à l'entier le plus proche "half away zero"
    rounded_age_value_str = str(round(age_value, 3))
    return f"Age: {age} ({rounded_age_value_str})"


# Interface Gradio
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Dropdown(choices=list(models.keys()), label="Choisir un modèle")],
    outputs=gr.Textbox(label="Prédictions des modèles")
)


# Lancer l'interface
if __name__ == "__main__":
    iface.launch()
