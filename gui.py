import os
import time

import streamlit as st
from PIL import Image

import tensorflow as tf
import keras
import numpy as np

import matplotlib.pyplot as plt

import cv2
import joblib
from scipy.special import softmax  # Per convertire i punteggi SVM in pseudo-probabilità

from task_3 import get_sift_descriptors, extract_lbp, extract_color_moments

# classi
CLASS_NAMES = [
    'air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing',
    'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
    'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding',
    'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket',
    'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men',
    'figure skating pairs', 'figure skating women', 'fly fishing', 'football',
    'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw',
    'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping',
    'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing',
    'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
    'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling',
    'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo',
    'pommel horse', 'rings', 'rock climbing', 'roller derby', 'rollerblade racing',
    'rowing', 'rugby', 'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing',
    'ski jumping', 'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing',
    'speed skating', 'steer wrestling', 'sumo wrestling', 'surfing', 'swimming',
    'table tennis', 'tennis', 'track bicycle', 'trapeze', 'tug of war', 'ultimate',
    'uneven bars', 'volleyball', 'water cycling', 'water polo', 'weightlifting',
    'wheelchair basketball', 'wheelchair racing', 'wingsuit flying'
]

# page name and layout
st.set_page_config(layout="wide", page_title="Image Classification Dashboard")

USE_DENSE_SIFT = True
SVM_IMG_SIZE = (512, 512)


@st.cache_resource
def load_task1_model():
    model_path = os.path.join("models", "mymodel_task1.keras")
    try:
        model = keras.models.load_model(model_path)
        return model
    except:
        return None


@st.cache_resource
def load_task2_model():
    model_path = os.path.join("models", "mymodel_task2.keras")
    try:
        model = keras.models.load_model(model_path)
        return model
    except:
        return None


@st.cache_resource
def load_mini_resnet():
    model_path = os.path.join("models", "miniresnet.keras")
    try:
        model = keras.models.load_model(model_path)
        return model
    except:
        return None


@st.cache_resource
def load_svm_pipeline():
    model_path = os.path.join("models", "cv_model_pipeline.pkl")
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        return None


# GRADCAM FUNCTION
def get_last_conv_layer(model):
    # cerca l'ultimo layer del modello convoluzionale
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.cast(img_array, tf.float32)

    try:
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            outputs = grad_model(img_tensor)
            last_conv_layer_output = outputs[0]
            preds = outputs[1]

            # unwrap delle liste
            if isinstance(last_conv_layer_output, (list, tuple)):
                last_conv_layer_output = last_conv_layer_output[0]
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            tape.watch(last_conv_layer_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])

            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

    except Exception as e:
        # se rete costruita con sequential
        conv_model = keras.models.Model(
            inputs=model.layers[0].input,
            outputs=model.get_layer(last_conv_layer_name).output
        )

        conv_output_shape = conv_model.output.shape[1:]
        classifier_input = keras.Input(shape=conv_output_shape)
        x = classifier_input

        layer_names = [layer.name for layer in model.layers]
        conv_idx = layer_names.index(last_conv_layer_name)

        for layer in model.layers[conv_idx + 1:]:
            x = layer(x)
        classifier_model = keras.models.Model(inputs=classifier_input, outputs=x)

        with tf.GradientTape() as tape:
            last_conv_layer_output = conv_model(img_tensor)
            tape.watch(last_conv_layer_output)

            preds = classifier_model(last_conv_layer_output)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            if pred_index is None:
                pred_index = tf.argmax(preds[0])

            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

    # valutazione finale dei gradienti
    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    if grads is None:
        raise ValueError("Gradients are None.")

    # costruzione delle heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(img_pil, heatmap, alpha=0.5):
    # sovrappone la gradcam all'immagine
    img_array = np.array(img_pil)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img


# UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border: 2px solid #000;
        font-weight: bold;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# colonns destra delle impostazioni
with st.sidebar:
    st.title("Settings")

    # menu per selezionare i modelli
    st.subheader("Select Model")
    model_option = st.selectbox(
        "models:",
        ["Task-1 CNN", "Task-2 CNN", "Mini-Resnet", "Task-3 SVM Fused"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # caricamento dell'immagine
    st.subheader("Load Image")
    uploaded_file = st.file_uploader(
        "image selection",
        type=['png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.write("##")
    run_button = st.button("Run")

# carico l'immagine e la mostro
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Image loaded")
        st.image(image, width='stretch')

    if run_button:
        # necessarie per la gradcam
        cnn_model_loaded = None
        img_batch_global = None
        top5_indices_global = []
        target_size_global = (299, 299)

        # nella colonna vicino all'immagine mostro i risultati
        with col2:
            st.markdown("### Classification Results")

            top5_results = []
            inference_time = ""

            # se selezionata una CNN
            if model_option in ["Task-1 CNN", "Task-2 CNN", "Mini-Resnet"]:

                if model_option == "Task-1 CNN":
                    cnn_model_loaded = load_task1_model()
                elif model_option == "Task-2 CNN":
                    cnn_model_loaded = load_task2_model()
                elif model_option == "Mini-Resnet":
                    cnn_model_loaded = load_mini_resnet()

                # se non viene trovato il modello
                if cnn_model_loaded is None:
                    st.error(f"Error: Model {model_option} not found.")
                else:
                    with st.spinner('classificazione...'):
                        # immagine ridimensionata e passata al modello
                        image_resized = image.resize(target_size_global)
                        img_array = np.array(image_resized)
                        img_batch_global = np.expand_dims(img_array, axis=0)

                        start_time = time.time()
                        predictions = cnn_model_loaded.predict(img_batch_global)
                        end_time = time.time()

                        # calcolo tempo di inferenza ed estrazione delle probabilità
                        inference_time = f"{(end_time - start_time) * 1000:.1f}ms"
                        probs = predictions[0]

                        top5_indices_global = np.argsort(probs)[-5:][::-1]
                        top5_probs = probs[top5_indices_global]

                        for idx, prob in zip(top5_indices_global, top5_probs):
                            class_name = CLASS_NAMES[idx]
                            top5_results.append((class_name, prob))

            # se non selezione una rete neurale e sto risolvendo il
            # task 3 definisco SVM
            elif model_option == "Task-3 SVM Fused":
                pipeline = load_svm_pipeline()

                if pipeline is None:
                    st.error("Model 'cv_model_pipeline.pkl' not found.")
                else:
                    with st.spinner('Estrazione feature ed inferenza SVM...'):
                        start_time = time.time()

                        # converto l'immagine in BGR, tutti i metodi la gestiranno correttamente
                        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        img_cv_resized = cv2.resize(img_cv, SVM_IMG_SIZE)
                        gray = cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2GRAY)

                        # estraggo feature sift
                        sift = cv2.SIFT_create()
                        des = get_sift_descriptors(gray, sift)

                        # se non ci sono descrittori creo vettore di 0
                        if des is None or len(des) == 0:
                            des = np.zeros((1, 128))

                        # dal modello addestrato importo tutto il necessario
                        kmeans = pipeline['kmeans_vocabulary']
                        words = kmeans.predict(des.astype(np.float64))
                        hist_sift, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1), density=True)

                        # estraggo feature texture e colore
                        feat_lbp = extract_lbp(gray)
                        feat_color = extract_color_moments(img_cv_resized)

                        # standardizzo i dati media 0 e std 1
                        scaler_sift = pipeline['scaler_sift']
                        scaler_lbp = pipeline['scaler_lbp']
                        scaler_color = pipeline['scaler_color']

                        hist_sift_sc = scaler_sift.transform(hist_sift.reshape(1, -1))
                        feat_lbp_sc = scaler_lbp.transform(np.nan_to_num(feat_lbp).reshape(1, -1))
                        feat_color_sc = scaler_color.transform(np.nan_to_num(feat_color).reshape(1, -1))

                        # fondo le features
                        fused_features = np.hstack((hist_sift_sc, feat_lbp_sc, feat_color_sc))
                        model_fused = pipeline['model_fused']

                        # eseguo la previsione ed estraggo le probabilità di classificazione
                        decision_scores = model_fused.decision_function(fused_features)[0]
                        pseudo_probs = softmax(decision_scores)
                        classes_svm = model_fused.classes_

                        top5_indices_svm = np.argsort(pseudo_probs)[-5:][::-1]

                        for idx in top5_indices_svm:
                            c_name = classes_svm[idx]
                            c_prob = pseudo_probs[idx]
                            top5_results.append((c_name, c_prob))

                        end_time = time.time()
                        inference_time = f"{(end_time - start_time) * 1000:.1f}ms"

            # di fianco all'immagine inserisco le classi con le probabilità
            if len(top5_results) > 0:
                st.markdown(f"**Top-5 Classi Predette:**")
                for i, (c_name, c_prob) in enumerate(top5_results):
                    st.metric(label=f"Rank #{i + 1}", value=c_name, delta=f"{c_prob * 100:.1f}%", delta_color="off")
                    st.write("")

                st.write("---")
                st.metric(label=f"Tempo di inferenza ({model_option})", value=inference_time)

        # se il modello impiegato è convoluzionale vengono generate delle grad-cam
        if model_option in ["Task-1 CNN", "Task-2 CNN", "Mini-Resnet"] and cnn_model_loaded is not None:
            st.markdown("---")
            st.markdown("### Model Explainability (Grad-CAM)")

            last_conv_layer_name = get_last_conv_layer(cnn_model_loaded)

            if last_conv_layer_name is None:
                st.warning("Nessun layer convoluzionale trovato. Impossibile generare la Grad-CAM.")
            else:
                st.markdown(
                    "**Clicca sulle schede qui sotto per esplorare la mappa di attivazione per ogni singola classe predetta:**")

                # creo le tabs in cui caricare le gradcam
                tab_titles = [f"#{i + 1} {name}" for i, (name, _) in enumerate(top5_results)]
                tabs = st.tabs(tab_titles)

                for i, tab in enumerate(tabs):
                    with tab:
                        try:
                            # calcolo la gradcam considerando la predizione fatta
                            heatmap = make_gradcam_heatmap(
                                img_batch_global,
                                cnn_model_loaded,
                                last_conv_layer_name,
                                pred_index=top5_indices_global[i]
                            )

                            # sovrapposizione con l'immagine caricata opportunamente ridimensionata
                            base_img_resized = image.resize(target_size_global)
                            cam_image = overlay_heatmap(base_img_resized, heatmap, alpha=0.6)

                            # Layout centrato per la heatmap
                            c1, c2, c3 = st.columns([1, 2, 1])
                            with c2:
                                st.image(cam_image, width='stretch')

                        except Exception as e:
                            st.warning(f"Errore durante la generazione della Grad-CAM: {e}")

else:
    st.info("Seleziona un modello e carica un'immagine")
    st.markdown(
        """
        <div style='display: flex; justify-content: center; align-items: center; height: 400px; border: 2px dashed #ccc; color: #ccc;'>
            Visualization area
        </div>
        """,
        unsafe_allow_html=True
    )