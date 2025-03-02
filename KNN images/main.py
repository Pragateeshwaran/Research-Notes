import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

def load_images(folder, label, image_size):
 
    data = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            with Image.open(file_path) as img:
                img = img.resize(image_size)
                img_array = np.array(img).flatten()  # Flatten the image
                data.append((img_array, label))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return data

apple_folder = r"F:\works\A-important\A-neurals\KNN images\apple"
mango_folder = r"F:\works\A-important\A-neurals\KNN images\mango"
test_image_path = r"F:\works\A-important\A-neurals\KNN images\apple\apple.jpg"
image_size = (64, 64)  
k_neighbors = 3  

apple_data = load_images(apple_folder, label=0, image_size=image_size)  
mango_data = load_images(mango_folder, label=1, image_size=image_size)  

data = apple_data + mango_data
X_train = np.array([item[0] for item in data])  
y_train = np.array([item[1] for item in data])  

knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_train, y_train)

with Image.open(test_image_path) as test_img:
    test_img = test_img.resize(image_size)
    test_array = np.array(test_img).flatten().reshape(1, -1)  
    prediction = knn.predict(test_array)
    class_name = "Apple" if prediction[0] == 0 else "Mango"
    print(f"The test image is classified as: {class_name}")
