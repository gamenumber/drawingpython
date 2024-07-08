import webbrowser
import threading
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def anime_style_filter(img):
    # RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # bilateral 필터로 노이즈 제거 및 엣지 보존
    img_color = cv2.bilateralFilter(img_rgb, 9, 300, 300)
    
    # 엣지 검출
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    
    # 색상 양자화
    img_small = cv2.resize(img_color, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    for _ in range(2):
        img_small = cv2.pyrDown(img_small)
    img_quant = img_small.copy()
    img_quant = img_quant // 32 * 32 + 16
    for _ in range(2):
        img_quant = cv2.pyrUp(img_quant)
    img_quant = cv2.resize(img_quant, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # 엣지 블러 및 강화
    edge = cv2.GaussianBlur(edge, (3, 3), 1)
    edge = cv2.Laplacian(edge, cv2.CV_8U, ksize=5)
    edge = 255 - edge
    
    # 색상 블렌딩
    result = cv2.bitwise_and(img_quant, img_quant, mask=edge)
    
    # 색상 향상
    result = cv2.addWeighted(result, 0.7, img_rgb, 0.3, 0)
    
    # 선명도 향상
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    result = cv2.filter2D(result, -1, kernel)
    
    # 밝기 및 대비 조정
    result = cv2.convertScaleAbs(result, alpha=1.1, beta=15)
    
    # 피부톤 보정
    skin_lower = np.array([0, 20, 70], dtype=np.uint8)
    skin_upper = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(cv2.cvtColor(result, cv2.COLOR_RGB2HSV), skin_lower, skin_upper)
    skin = cv2.bitwise_and(result, result, mask=skin_mask)
    skin = cv2.GaussianBlur(skin, (3, 3), 0)
    result = np.where(skin_mask[..., None], skin, result)
    
    # 머리카락 디테일 강화
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    hair_mask = cv2.inRange(gray, 0, 80)
    hair = cv2.bitwise_and(result, result, mask=hair_mask)
    hair = cv2.addWeighted(hair, 1.2, np.zeros_like(hair), 0, 0)
    result = np.where(hair_mask[..., None], hair, result)
    
    # 색상 보정
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            
            img = cv2.imread(img_path)
            result = anime_style_filter(img)
            
            result_path = os.path.join(RESULT_FOLDER, 'anime_' + file.filename)
            cv2.imwrite(result_path, result)
            
            return render_template('index.html', original_img=img_path, filtered_img=result_path)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def send_result_file(filename=''):
    return send_from_directory(RESULT_FOLDER, filename)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
