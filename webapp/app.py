# Petite application web pour tester le modèle de détection des défauts PCB.
# Lancement : python app.py  puis ouvrir http://127.0.0.1:7860 dans le navigateur.
import time
from pathlib import Path

import gradio as gr
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent/'best.pt'
IMGSZ = 1280      # même taille qu'à l'entraînement
REF_WIDTH = 3000  # largeur approximative des photos d'entraînement

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Copier best.pt (entraîné sur Colab) dans : {MODEL_PATH.parent}')
model = YOLO(str(MODEL_PATH))

def analysis_size(image):
    # Une photo recadrée (ex. 600 px) doit être moins agrandie, pour que les défauts
    # gardent la même taille en pixels qu'à l'entraînement. Une photo complète
    # (environ 2500 à 3000 px) est traitée exactement comme à l'entraînement.
    if max(image.size) >= 2000:
        return IMGSZ
    size = round(max(image.size) * IMGSZ / REF_WIDTH / 32) * 32
    return min(IMGSZ, max(128, size))

def detect(image, conf):
    if image is None:
        return None, pd.DataFrame(), ''
    imgsz = analysis_size(image)
    start = time.perf_counter()
    result = model.predict(image, imgsz=imgsz, conf=conf, device='cpu', verbose=False)[0]
    elapsed = time.perf_counter() - start
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB pour l'affichage
    rows = [{'Défaut': result.names[int(c)], 'Confiance': round(float(s), 2),
             'x1': int(b[0]), 'y1': int(b[1]), 'x2': int(b[2]), 'y2': int(b[3])}
            for c, s, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]
    table = pd.DataFrame(rows, columns=['Défaut', 'Confiance', 'x1', 'y1', 'x2', 'y2'])
    summary = (f'{len(rows)} défaut(s) détecté(s) en {elapsed:.1f} s '
               f'(photo {image.size[0]}×{image.size[1]} px, analysée à {imgsz} px)')
    return annotated, table, summary

with gr.Blocks(title='Détection de défauts PCB') as demo:
    gr.Markdown('# Détection de défauts PCB (YOLO11n)\n'
                'Déposer une photo de carte électronique, puis cliquer sur **Analyser**.')
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type='pil', label='Photo de la carte')
            conf_in = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                label='Seuil de confiance (plus haut = moins de fausses alertes)')
            button = gr.Button('Analyser', variant='primary')
        with gr.Column():
            image_out = gr.Image(label='Résultat')
            summary_out = gr.Markdown()
    table_out = gr.Dataframe(label='Détections')
    button.click(detect, [image_in, conf_in], [image_out, table_out, summary_out])

if __name__ == '__main__':
    demo.launch()
