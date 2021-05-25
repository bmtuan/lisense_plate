from tkinter import *
from tkinter.ttk import *
import tkinter
from tkinter import filedialog
from PIL import Image,ImageTk
import cv2
import test1
import numpy as np

window = Tk()
window.title("Nhận diện biển số xe - Nhóm 10")
window.geometry("1200x550")

nameProgram = tkinter.Label(window,text = "Nhận diện biển số xe - Nhóm 10")
nameProgram.config(font=("Arial", 30))
nameProgram.pack(pady=10)

list_images = []
list_images.append(2)
ID = ""
LAME = ""
P = ""


def handleButton(list_images):
    if len(list_images) > 2:
        list_images = list_images[0:1]
    file = filedialog.askopenfilename()
    image = cv2.imread(file,cv2.IMREAD_COLOR)
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    choseImage.configure(image = imgtk)
    choseImage.image = imgtk
    conImage.configure(image=imgtk)
    conImage.image = imgtk
    list_images.append(image)
    return

def processBtn():
    image_copy = np.copy(list_images[1])
    list_image = test1.predict(image_copy)
    for i in range(len(list_image)):
        image2 = cv2.resize(list_image[i],(472,303))
        list_images.append(image2)
    return

def nextBtn():
    if (list_images[0] == 5):
        list_images[0] = 1

    img = Image.fromarray(list_images[list_images[0]])
    list_images[0] = list_images[0] + 1
    imgtk = ImageTk.PhotoImage(image=img)
    conImage.configure(image=imgtk)
    conImage.image = imgtk
    return

# def checkBtn():
#     (P, ID, LAME) = matching.find(list_images[1])
#     P = P * 100
#     if(P < 90):
#         LAME = "Không tồn tại vân tay trong CSDL!"
#         nametxt.configure(text = LAME , fg ="red")
#     else:
#         nametxt.configure(text= "Họ Và Tên: " + LAME)
#         idtxt.configure(text = "ID: " + str(ID))
#         xxp.configure(text= "Tỷ lệ chính xác: " + str(P) + "%")
#
#     return

load = Image.open("white.png")
render = ImageTk.PhotoImage(load)


# anhmau
anhmau = tkinter.Label(window,text = "Ảnh Nhận Diện")
anhmau.config(font=("Arial", 15))
anhmau.place(x=250, y = 70)

choseImage = tkinter.Label(window,image=render)
choseImage.place(x= 50, y = 100)

# ảnh quá trình
anhmau1 = tkinter.Label(window,text = "Ảnh Quá Trình")
anhmau1.config(font=("Arial", 15))
anhmau1.place(x=800, y = 70)


conImage = tkinter.Label(window,image=render)
conImage.place(x= 600, y = 100)



# ảnh kết quả
# anhKQ = tkinter.Label(window,text = "Ảnh Kết Qủa")
# anhKQ.config(font=("Arial", 15))
# anhKQ.place(x=1150, y = 70)
#
# KQImage = tkinter.Label(window,image=render)
# KQImage.place(x= 1150, y = 100)

#theem hop thoai tep
btnChose = Button(window, text = "Chọn ảnh", command = handleButton(list_images))
btnChose.place(x = 250 , y =500)

# kiểm tra
# btnCheck = Button(window,text = "Check", command =  checkBtn)
# btnCheck.place(x = 830, y = 500)


# nut xu ly
processBtn = Button(window, text = "Xử lý ảnh" , command = processBtn)
processBtn.place(x = 520 , y =500)

# button neext
btnNext = Button(window, text = "Next" , command = nextBtn)
btnNext.place(x = 800 , y =500)

# idtxt = tkinter.Label(window,text = "")
# idtxt.config(font=("Arial", 20))
# idtxt.place(x= 950, y = 245)
#
# nametxt = tkinter.Label(window,text = "")
# nametxt.config(font=("Arial", 20))
# nametxt.place(x= 950, y = 345)
#
# xxp = tkinter.Label(window,text = "")
# xxp.config(font=("Arial", 20))
# xxp.place(x= 950, y = 445)

window.mainloop()



