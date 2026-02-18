import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model once
model = load_model("Trained_model.h5")

# Class labels (IMPORTANT: must match training order)
class_names = [
    'Be My Friend',
    'Do not Worry',
    'Get well soon',
    'Give Me Water',
    'Good Morning',
    'Good Night',
    'Help Me',
    'How are You',
    'Hungry',
    'Hurry Up',
    'I am Fine',
    'I am Late',
    'I am Not Feeling Well',
    'I am Tired',
    'I Love Reading',
    'I Love Writing',
    'I will Try my level Best',
    'Its Raining Outside',
    'Lets Dance',
    'Lets Go',
    'Lets Play',
    'Long Time no See',
    'Look Down',
    'Look Up',
    'Medicine',
    'Music'
]

img_name = input("Enter image name (example: 1.png): ")
image_path = f"./{img_name}"

test_image = image.load_img(image_path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)

predicted_class = class_names[np.argmax(result)]

print("\nPredicted Sign is:", predicted_class)
