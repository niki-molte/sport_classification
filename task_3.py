import cv2
import numpy as np
import os
import glob
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score # <-- Aggiunto top_k_accuracy_score
from sklearn.model_selection import GridSearchCV
from skimage.feature import local_binary_pattern
from scipy.stats import skew
import time
import joblib

PATH_TRAIN = "/home/niki/Documenti/sport dataset/exam_dataset-20260123T091607Z-3-001/exam_dataset/valid/valid"
PATH_TEST_DEGRADED = "/home/niki/Documenti/sport dataset/exam_dataset-20260123T091607Z-3-001/exam_dataset/test_degradato/test_degradato"
PATH_TEST_CLEAN = "/home/niki/Documenti/sport dataset/exam_dataset-20260123T091607Z-3-001/exam_dataset/test/test"

IMG_SIZE = (512, 512)
K_VOCABULARY = 700

# definisce ed usa una griglia densa con un determinato
# passo di campionamento.
USE_DENSE_SIFT = True


def get_sift_descriptors(img_gray, sift_obj):
    if USE_DENSE_SIFT:
        # passo di campionamento della griglia
        step_size = 10
        kps = [cv2.KeyPoint(x, y, step_size)
               for y in range(0, img_gray.shape[0], step_size)
               for x in range(0, img_gray.shape[1], step_size)]
        kps, des = sift_obj.compute(img_gray, kps)
    else:
        # estrazione automatica delle feature sift
        kps, des = sift_obj.detectAndCompute(img_gray, None)
    return des


def extract_lbp(img_gray):
    radius = 1
    n_points = 8 * radius

    # 'uniform' genera un istogramma più compatto (10 bin per r=1, p=8)
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')

    # calcolo istogramma
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_color_moments(img_bgr):
    # conversione nello spazio colore HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    features = []

    # immagine suddivisa in quadranti
    h, w, _ = hsv.shape
    h_mid, w_mid = h // 2, w // 2

    regions = [
        hsv[0:h_mid, 0:w_mid],  # alto sinistra
        hsv[0:h_mid, w_mid:w],  # alto destra
        hsv[h_mid:h, 0:w_mid],  # basso sinistra
        hsv[h_mid:h, w_mid:w]   # basso destra
    ]

    for region in regions:
        # ciclo sui canali H, S, V
        for i in range(3):
            channel_pixels = region[:, :, i].ravel()

            # se la regione è vuota
            if len(channel_pixels) == 0:
                features.extend([0, 0, 0])
                continue

            # valuto media e std del canale
            mean_val = np.mean(channel_pixels)
            std_val = np.std(channel_pixels)

            features.append(mean_val)
            features.append(std_val)

            # se la deviazione standard è quasi 0 (colori uniformi), lo skew è 0.
            # impostato per evitare warning
            if std_val < 1e-6:
                features.append(0.0)
            else:
                try:
                    s = skew(channel_pixels)
                    if np.isnan(s): s = 0.0
                    features.append(s)
                except:
                    features.append(0.0)

    return np.array(features)



def load_dataset(dataset_path, kmeans_model=None):
    """
    Carica immagini ed estrae TUTTE le features separatamente.
    Se kmeans_model è None, stiamo facendo training e raccogliamo i descrittori SIFT per il vocabolario.
    Se kmeans_model esiste, stiamo facendo test o creando gli istogrammi BoW.
    """
    print(f"   Working on folder: {dataset_path} ...")

    sift = cv2.SIFT_create()

    # features
    sift_raw_descriptors = []  # k-means training

    data_sift_bow = []  # istogrammi SIFT
    data_lbp = []  # vettori LBP
    data_color = []  # vettori Colore
    labels = []

    # Caricamento ordinato
    class_folders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    # Se stiamo processando le immagini per creare i vettori (non solo raccogliere descrittori)
    temp_sift_des_per_image = []

    for class_name in class_folders:
        folder_path = os.path.join(dataset_path, class_name)
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue

            img = cv2.resize(img, IMG_SIZE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. SIFT Extraction
            des = get_sift_descriptors(gray, sift)
            if des is None: des = np.zeros((1, 128))  # Fallback

            if kmeans_model is None:
                # Fase Training: accumula descrittori grezzi
                sift_raw_descriptors.append(des)
                temp_sift_des_per_image.append(des)
            else:
                # Fase Test/Finalizzazione: calcola subito BoW
                words = kmeans_model.predict(des.astype(np.float64))
                hist_bow, _ = np.histogram(words, bins=np.arange(kmeans_model.n_clusters + 1), density=True)
                data_sift_bow.append(hist_bow)

            # 2. LBP Extraction
            lbp_feat = extract_lbp(gray)
            data_lbp.append(lbp_feat)

            # 3. Color Extraction
            color_feat = extract_color_moments(img)
            data_color.append(color_feat)

            labels.append(class_name)

    if kmeans_model is None:
        return sift_raw_descriptors, temp_sift_des_per_image, np.array(data_lbp), np.array(data_color), np.array(labels)
    else:
        return np.array(data_sift_bow), np.array(data_lbp), np.array(data_color), np.array(labels)


def train_svm(X, y, name):
    print(f"   Training SVM on {name} features (Dim: {X.shape[1]})...")

    param_grid = {'C': [0.1, 0.001, 1, 10, 100], 'kernel': ['rbf', 'linear']}
    grid = GridSearchCV(SVC(random_state=42, class_weight='balanced'), param_grid, cv=3, n_jobs=-1)
    grid.fit(X, y)
    print(f"     -> Best Params: {grid.best_params_} | CV Score: {grid.best_score_:.3f}")

    return grid.best_estimator_

if __name__ == "__main__":
    start_time = time.time()
    print(f"STARTING ANALYSIS - SIFT: {'DENSE' if USE_DENSE_SIFT else 'AUTO'}")

    print("\n--- 1. Loading Training Set ---")

    # caricamento dati
    raw_sift, train_sift_des_list, X_train_lbp, X_train_color, y_train = load_dataset(PATH_TRAIN, kmeans_model=None)

    # Verifica numero di classi per la Top-5
    unique_classes = np.unique(y_train)
    k_val = 5 if len(unique_classes) >= 5 else len(unique_classes)

    # sift BOVW
    print(f"  BOFW SIFT (K={K_VOCABULARY})...")
    if len(raw_sift) > 0:
        all_descriptors = np.vstack(raw_sift).astype(np.float64)
        kmeans = MiniBatchKMeans(n_clusters=K_VOCABULARY, random_state=42, batch_size=1000, n_init='auto').fit(
            all_descriptors)
    else:
        raise ValueError("Error: No SIFT descriptor found in the training set")

    # SIFT -> BoW Histograms
    print("   SIFT conversion in BoW Histograms...")
    X_train_sift = []
    for des in train_sift_des_list:
        if des is not None and len(des) > 0:
            words = kmeans.predict(des.astype(np.float64))
            hist, _ = np.histogram(words, bins=np.arange(K_VOCABULARY + 1), density=True)
            X_train_sift.append(hist)
        else:
            # immagini senza keypoints
            X_train_sift.append(np.zeros(K_VOCABULARY))

    X_train_sift = np.array(X_train_sift)

    # data cleaning, NaN sostituiti con 0 prima dello
    # scaling
    X_train_lbp = np.nan_to_num(X_train_lbp)
    X_train_color = np.nan_to_num(X_train_color)

    # media e varianza sul training set per ogni tipo di feature
    print("   Scaling delle features...")
    scaler_sift = StandardScaler().fit(X_train_sift)
    scaler_lbp = StandardScaler().fit(X_train_lbp)
    scaler_color = StandardScaler().fit(X_train_color)

    X_train_sift_sc = scaler_sift.transform(X_train_sift)
    X_train_lbp_sc = scaler_lbp.transform(X_train_lbp)
    X_train_color_sc = scaler_color.transform(X_train_color)

    # concatenazione delle features
    X_train_fused = np.hstack((X_train_sift_sc, X_train_lbp_sc, X_train_color_sc))

    # addestramento SVM
    print("\n--- 2. Addestramento Modelli SVM ---")

    model_sift = train_svm(X_train_sift_sc, y_train, "ONLY SIFT")
    model_lbp = train_svm(X_train_lbp_sc, y_train, "ONLY LBP (Texture)")
    model_color = train_svm(X_train_color_sc, y_train, "ONLY COLOR")
    model_fused = train_svm(X_train_fused, y_train, "FUSION (Sift+Lbp+Color)")

    print("\n--- 3. Evaluation on Test Sets ---")


    def process_and_evaluate(path, label_desc):
        print(f"   Evaluation on {label_desc}...")

        # carico i dati, passo il kmeans creato in training
        X_s, X_l, X_c, y = load_dataset(path, kmeans_model=kmeans)

        X_l = np.nan_to_num(X_l)
        X_c = np.nan_to_num(X_c)

        # scaling con scaler addestrati sul training
        X_s = scaler_sift.transform(X_s)
        X_l = scaler_lbp.transform(X_l)
        X_c = scaler_color.transform(X_c)

        # fusion
        X_f = np.hstack((X_s, X_l, X_c))

        # --- Predizioni Top-1 ---
        pred_s = model_sift.predict(X_s)
        pred_l = model_lbp.predict(X_l)
        pred_c = model_color.predict(X_c)
        pred_f = model_fused.predict(X_f)

        acc_s = accuracy_score(y, pred_s)
        acc_l = accuracy_score(y, pred_l)
        acc_c = accuracy_score(y, pred_c)
        acc_f = accuracy_score(y, pred_f)

        # --- Calcolo Punteggi (Scores) per Top-5 ---
        # decision_function restituisce la distanza dall'iperpiano per ogni classe
        scores_s = model_sift.decision_function(X_s)
        scores_l = model_lbp.decision_function(X_l)
        scores_c = model_color.decision_function(X_c)
        scores_f = model_fused.decision_function(X_f)

        # --- Predizioni Top-5 ---
        classes = model_sift.classes_
        top5_s = top_k_accuracy_score(y, scores_s, k=k_val, labels=classes)
        top5_l = top_k_accuracy_score(y, scores_l, k=k_val, labels=classes)
        top5_c = top_k_accuracy_score(y, scores_c, k=k_val, labels=classes)
        top5_f = top_k_accuracy_score(y, scores_f, k=k_val, labels=classes)

        return (acc_s, top5_s), (acc_l, top5_l), (acc_c, top5_c), (acc_f, top5_f)


    # test
    res_degraded = process_and_evaluate(PATH_TEST_DEGRADED, "DEGRADED")

    print("\n" + "=" * 70)
    print(f"{'STRATEGIA':<22} | {'Top-1 Accuracy':<15} | {'Top-5 Accuracy':<15} |")
    print("-" * 70)

    metrics = [
        ("SIFT Only", res_degraded[0]),
        ("LBP Only (Texture)", res_degraded[1]),
        ("Color Only", res_degraded[2]),
        ("FUSION (ALL)", res_degraded[3])
    ]

    for name, (acc_d_1, acc_d_5) in metrics:
        print(f"{name:<22} | {acc_d_1:.2%}     | {acc_d_5:.2%}     |")

    print("=" * 70)
    print(f" Execution time: {time.time() - start_time:.1f} seconds")

    pipeline_export = {
        'kmeans_vocabulary': kmeans,
        'scaler_sift': scaler_sift,
        'scaler_lbp': scaler_lbp,
        'scaler_color': scaler_color,
        'model_sift': model_sift,
        'model_lbp': model_lbp,
        'model_color': model_color,
        'model_fused': model_fused
    }
    #joblib.dump(pipeline_export, 'models/cv_model_pipeline.pkl')