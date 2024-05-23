import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Загрузка предобученной модели
model = tf.keras.models.load_model('deeplab_900.h5', compile=False) #, compile=False

def load_image(label):
    uploaded_file = st.file_uploader(label=label)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        return np.array(image)
    else:
        return None

def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

st.title('Сегментация печени')

# Загрузка снимка для сегментации
input_image = load_image("Выберите снимок для сегментации")

# Загрузка истинной маски
ground_truth = load_image("Выберите маску с истиной разметкой")

if input_image is not None:
    result = st.button('Сегментация печени')
    if result:
        st.write('Результаты сегментации')
        input_image_batch = np.expand_dims(input_image, axis=0)  
        predicted_mask = model.predict(input_image_batch, verbose=1)
        predicted_mask = np.squeeze(predicted_mask, axis=0)  
        
        # Отображение предсказанной маски
        st.image(predicted_mask, caption='Сегментированное изображение', use_column_width=True)
        
        if ground_truth is not None:
            # Расчет Индекса Жакара, если загружена истинная маска
            jac_index = jaccard_index(ground_truth, predicted_mask > 0.5)
            st.write(f'Индекс Жакара: {jac_index:.4f}')
        else:
            st.write('Истинная маска не загружена, Индекс Жакара не будет рассчитан.')
else:
    st.write('Пожалуйста, загрузите снимок для продолжения.')