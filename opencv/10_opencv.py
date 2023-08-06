import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def myPutText(src, text, pos, font_size, font_color):
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('fonts/gulim.ttc', font_size)
    draw.text(pos, text, font=font, fill=font_color)
    return np.array(img_pil)

img = np.zeros((480, 640, 3), dtype=np.uint8)

#SCALE = 1
FONT_SIZE = 30
COLOR = (255, 255, 255)
#THICKNESS = 1

# cv2.putText(img, "나도코딩", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR, THICKNESS)
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께

img = myPutText(img, '나도코딩', (20, 50), FONT_SIZE, COLOR)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()