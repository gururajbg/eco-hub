import torch
import numpy as np
from PIL import Image, ImageDraw

def load_model(path):
    model = torch.hub.load('ultralytics/yolov12', 'custom', path=path, force_reload=True)  # replace with YOLOv12
    model.eval()
    return model

def run_inference_and_combine(image, model1, model2):
    results1 = model1(image)
    results2 = model2(image)

    # Convert results to pandas
    df1 = results1.pandas().xyxy[0]
    df2 = results2.pandas().xyxy[0]

    # Adjust class names to avoid overlap
    df2['name'] = df2['name'].apply(lambda x: f"m2_{x}")

    # Combine both
    combined_df = df1._append(df2, ignore_index=True)

    # Draw on image
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for _, row in combined_df.iterrows():
        draw.rectangle([row['xmin'], row['ymin'], row['xmax'], row['ymax']], outline='red', width=2)
        draw.text((row['xmin'], row['ymin'] - 10), row['name'], fill='yellow')

    return {"df": combined_df, "image": img_draw}
