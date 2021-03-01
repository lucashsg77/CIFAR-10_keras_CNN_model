
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model
from keras.preprocessing import image

mainWindow = tk.Tk()
mainWindow.geometry('800x600')
mainWindow.title('CIFAR10 CNN Model Testing')
mainWindow.configure(background = '#092834')
label = Label(mainWindow, background='#092834', font = ('arial', 15, 'bold'))
sign_image = Label(mainWindow)

#the labels to all the dataset classes
classes = { 
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck' 
}

def classify(image_path, model):
    img = image.load_img(image_path, target_size=(32,32))
    img = image.img_to_array(img)
    img = img.reshape((1,) + img.shape)
    img = img/255.
    prediction = model.predict(img)
    class_label = classes[numpy.argmax(prediction)]
    print(numpy.argmax(prediction))
    label.configure(foreground = '#347B98', text = class_label) 

def show_classify_b(image_path, model):
    classify_b = Button(mainWindow,text = "Classify Image", command = lambda: classify(image_path, model), padx = 10, pady=5)
    classify_b.configure(background = '#66B032', foreground = 'white', font = ('arial', 10, 'bold'))
    classify_b.place(relx = 0.84, rely = 0.46)

def show_select_image_b(model):
    select_image_b = Button(mainWindow, text = "Select an image", command = lambda: select_image(model), padx = 10, pady = 5)
    select_image_b.configure(background = '#66B032', foreground = 'white', font = ('arial', 10, 'bold'))
    select_image_b.place(relx = 0.01, rely = 0.46)

def select_image(model):
    try:
        image_path = filedialog.askopenfilename()
        selected_image = Image.open(image_path)
        selected_image.thumbnail(((mainWindow.winfo_width()/2.25), (mainWindow.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(selected_image)
        sign_image.configure(image = im)
        sign_image.image = im
        label.configure(text = '')
        show_classify_b(image_path, model)
        sign_image.pack(expand = True)
        label.pack(expand = True)
    except:
        pass

model_path = filedialog.askopenfilename()
model = load_model(model_path)
show_select_image_b(model)

mainWindow_header = Label(mainWindow, text = "Image Classification CIFAR10", pady = 20, font = ('arial',20,'bold'))
mainWindow_header.configure(background = '#092834', foreground = '#66B032')
mainWindow_header.pack()

mainWindow.mainloop()
