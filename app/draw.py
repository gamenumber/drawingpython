import webbrowser
import threading
from flask import Flask, request, render_template, send_from_directory, url_for
import cv2
import numpy as np
import os
from skimage import color, filters, exposure

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def anime_style_filter(img):
    try:
        # RGB 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # bilateral 필터로 노이즈 제거 및 엣지 보존
        img_color = cv2.bilateralFilter(img_rgb, 9, 75, 75)
        
        # 엣지 검출
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edge = filters.sobel(gray)
        edge = np.uint8(255 * edge)
        edge = cv2.adaptiveThreshold(edge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        
        # 색상 양자화
        img_lab = color.rgb2lab(img_color)
        
        # Normalize the image to [0, 1] before applying adaptive histogram equalization
        img_lab[..., 0] = exposure.equalize_adapthist(img_lab[..., 0] / 100.0) * 100.0
        img_color = color.lab2rgb(img_lab)
        img_color = np.uint8(img_color * 255)
        
        # 엣지 블러 및 강화
        edge = cv2.GaussianBlur(edge, (3, 3), 1)
        edge = cv2.Laplacian(edge, cv2.CV_8U, ksize=5)
        edge = 255 - edge
        
        # 색상 블렌딩
        result = cv2.bitwise_and(img_color, img_color, mask=edge)
        
        # 선명도 향상
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel_sharpen)
        
        # 밝기 및 대비 조정
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=20)
        
        # 피부톤 보정
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin = cv2.bitwise_and(result, result, mask=skin_mask)
        skin = cv2.GaussianBlur(skin, (3, 3), 0)
        result = np.where(skin_mask[..., None], skin, result)
        
        # 머리카락 디테일 강화
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        hair_mask = cv2.inRange(gray, 0, 80)
        hair = cv2.bitwise_and(result, result, mask=hair_mask)
        hair = cv2.addWeighted(hair, 1.3, np.zeros_like(hair), 0, 0)
        result = np.where(hair_mask[..., None], hair, result)
        
        # 색상 보정
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error in anime_style_filter: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            
            img = cv2.imread(img_path)
            if img is not None:
                result = anime_style_filter(img)
                if result is not None:
                    result_path = os.path.join(RESULT_FOLDER, 'anime_' + file.filename)
                    cv2.imwrite(result_path, result)
                    
                    return render_template('index.html', original_img=file.filename, filtered_img='anime_' + file.filename)
                else:
                    return "Error processing image", 500
            else:
                return "Error processing image", 400
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
