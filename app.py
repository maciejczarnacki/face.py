import time

import tkinter as Tk
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np

from face.face import Faceanalyzer

# window definition
root = Tk.Tk()
root.geometry("650x600")
root.wm_title("face.py package test")

# starting black image
first = np.full((480,640), 0)
first_ = Image.fromarray(first)
imgtk = ImageTk.PhotoImage(image=first_)

# camera control
camera = cv2.VideoCapture(1 + cv2.CAP_DSHOW)  # opencv camera inicialisation
camera.set(cv2.CAP_PROP_BRIGHTNESS, 100) #set brightness
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set height

pTime = 0

face = Faceanalyzer()

# function executed in a loop                   
def getface():
    global pTime
    s, img = camera.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    l_e, r_e, l_i, r_i, l_b, r_b, lips = face.face_features(
        img, eyes_.get(), irises_.get(), eyebrows_.get(), lips_.get())
    cTime = time.time()
    if r_e != None:
        for i in r_e:
            cv2.circle(img, (i[0], i[1]), 1, (0, 0, 255), 2)
        if face.eyes_status(l_e, r_e)[1]:
            cv2.putText(img, f"open", (10,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
        else:
            cv2.putText(img, f"closed", (10,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
    if l_e != None: 
        for j in l_e:
            cv2.circle(img, (j[0], j[1]), 1, (0, 255, 0), 2)
        if face.eyes_status(l_e, r_e)[0]:
            cv2.putText(img, f"open", (500,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
        else:
            cv2.putText(img, f"closed", (500,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
    if r_i != None:
        for j in r_i:
            cv2.circle(img, (j[0], j[1]), 1, (0, 0, 255), 2)  
    if l_i != None: 
        for j in l_i:
            cv2.circle(img, (j[0], j[1]), 1, (0, 255, 0), 2)
    if r_b != None:
        for j in r_b:
            cv2.circle(img, (j[0], j[1]), 1, (0, 0, 255), 2)  
    if l_b != None: 
        for j in l_b:
            cv2.circle(img, (j[0], j[1]), 1, (0, 255, 0), 2)
    if lips != None: 
        for j in lips:
            cv2.circle(img, (j[0], j[1]), 1, (255, 0, 0), 2)
        if face.lips_status(lips):
            cv2.putText(img, f"open", (280,400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        else:
            cv2.putText(img, f"closed", (280,400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
    img_ = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img_)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    root.after(1, getface)
    

lmain = Tk.Label(master=root, height=480, width=640)
lmain.pack(side=Tk.TOP, fill="both")
lmain.imgtk = imgtk
lmain.configure(image=imgtk)                       

getPhotoButton = Tk.Button(master=root, text='Get photos', command=getface)
getPhotoButton.pack(side=Tk.TOP, fill="both")

eyes_ = Tk.IntVar()
eyes_check = Tk.Checkbutton(master=root, text="eyes", variable=eyes_)
eyes_check.pack(side=Tk.LEFT)

irises_ = Tk.IntVar()
irises_check = Tk.Checkbutton(master=root, text="irises", variable=irises_)
irises_check.pack(side=Tk.LEFT)

eyebrows_ = Tk.IntVar()
eyebrows_check = Tk.Checkbutton(master=root, text="eyebrows", variable=eyebrows_)
eyebrows_check.pack(side=Tk.LEFT)

lips_ = Tk.IntVar()
lips_check = Tk.Checkbutton(master=root, text="lips", variable=lips_)
lips_check.pack(side=Tk.LEFT)

Tk.mainloop()