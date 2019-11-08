from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL, X
from PIL import Image, ImageTk
from plasticClassifier import plasticClassifier
import os

debug = False

class MainDisplay:
    def __init__(self, master):
        self.master = master
        master.title("Plastic Classifier")

        self.path = "imgs/test/Milk/milk6.jpg"
        img = Image.open(self.path)
        img = img.resize((1280,720))
        photo = ImageTk.PhotoImage(img)
        self.image = Label(master, image=photo)
        self.image.image = photo
        self.image.pack(side="top")
        self.confidenceSlider = Scale(master, from_=0, to=1, orient=HORIZONTAL, tickinterval=0.1, resolution=0.01, length=580, label='Confidence')
        self.confidenceSlider.set(0.5)
        self.confidenceSlider.pack(side='left')
        self.nonMaximaSlider = Scale(master, from_=0, to=1, orient=HORIZONTAL, tickinterval=0.1, resolution=0.01, length=580, label='NMS')
        self.nonMaximaSlider.set(0.3)
        self.nonMaximaSlider.pack(side='right')
        self.changePic = Button(master, text="Change Picture", command=self.changePhoto)
        self.changePic.pack(side="top", pady=5, fill=X, padx=5)
        self.updatePic = Button(master, text="Update", command=self.updatePhoto)
        self.updatePic.pack(side="top", pady=5, fill=X, padx=5)

    def changePhoto(self):
        self.path = filedialog.askopenfilename(initialdir=os.path.dirname(os.path.abspath(__file__)), title="Select Photo", filetypes = (('jpeg files','*.jpg'),('all files','*.*')))
        img = Image.open(self.path)
        img = img.resize((1280,720))
        photo = ImageTk.PhotoImage(img)
        self.image.configure(image=photo)
        self.image.image = photo
        if debug == True:
            print("updated")

    def updatePhoto(self):
        path = self.path
        confidence = self.confidenceSlider.get()
        nms = self.nonMaximaSlider.get()
        predictor = plasticClassifier(confidence, nms, path)
        predictor.label_image()
        path = "object-detection.jpg"
        img = Image.open(path)
        img = img.resize((1280,720))
        photo = ImageTk.PhotoImage(img)
        self.image.configure(image=photo)
        self.image.image = photo
        if debug == True:
            print('Image predicted')

    
if __name__ == "__main__":
    root = Tk()
    root.geometry('1280x800')
    app = MainDisplay(root)
    root.mainloop()