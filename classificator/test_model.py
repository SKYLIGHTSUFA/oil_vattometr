import time

from ultralytics import YOLO
import cv2
import  os
import shutil
import customtkinter


customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue") # Themes: blue (default), dark-blue, green


model = YOLO("runs/classify/train2/weights/best.pt")
dirr = r"C:\Users\User\Documents\projects\belt\dataset\valid\bad"

for filename in os.listdir(dirr):
    if not filename.endswith("png"):
        continue
    results = model.predict(source=os.path.join(dirr, filename), verbose=False)
    conf = results[0].probs.data.tolist()
    idx = conf.index(max(conf))

    cv2.imshow("0", cv2.imread(os.path.join(dirr, filename)))
    print(f'idx = {idx}, filename = {filename}')
    image = cv2.imread(os.path.join(dirr, filename))
    #if int(idx) == 1:
    cv2.imshow("0", image)
    cv2.waitKey(0)
        #time.sleep(100)
    #    shutil.move(os.path.join(dirr, filename), "dataset/tmp/true/"+filename)
