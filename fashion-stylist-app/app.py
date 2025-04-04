import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("fashion_model.h5")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def recommend(class_name):
    recs = {
        'T-shirt/top': ["Pair with jeans", "Layer under a jacket"],
        'Dress': ["Heels or boots go well", "Try a belt for shape"],
        'Sneaker': ["Casual or streetwear fit", "Not for formal occasions"],
    }
    return recs.get(class_name, ["No specific tips available"])

def predict(image):
    img = Image.fromarray(image).convert("L").resize((28, 28))
    img_array = np.expand_dims(np.expand_dims(np.array(img) / 255.0, axis=-1), axis=0)
    prediction = model.predict(img_array)
    class_name = class_names[np.argmax(prediction)]
    tips = recommend(class_name)
    return f"Prediction: {class_name}", "Styling Tips:\n" + "\n".join("• " + tip for tip in tips)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Clothing Image"),
    outputs=["text", "text"],
    title="AI Fashion Stylist",
    description="Upload a fashion image and get AI-powered classification + styling advice!"
)

demo.launch()
