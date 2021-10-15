import numpy as np
import tkinter as tk
import cv2
from PIL import ImageTk, Image
from tkinter import filedialog
from keras.datasets import mnist           #pip install tensorflow     pip install keras
from matplotlib import pyplot

class Application(tk.Frame):
    #Construtor do objeto Application
    def __init__(self, master):
        self.master = master
        self.image = None
        self.default = None
        self.photoImage = None
        self.Widgets()

    #Método para inserir imagem
    def inserir(self):
        fileTypes = [("Image PNG, JPG", ".png .jpg")]
        filename = filedialog.askopenfilename(initialdir = "Prática", title = "Select a image", filetypes = fileTypes)

        if filename != '':
            self.image = Image.open(filename)
            self.atualizarTela()
    
    #Método para voltar a imagem anterior
    def desfazer(self):
        self.image = self.default
        self.atualizarTela()

    #Método para converter a imagem para abrir no Tkinter
    def convertTkinter(self, image):
        self.image = Image.fromarray(image)
        self.atualizarTela()

    #Método para atualizar a label com a imagem
    def atualizarTela(self):    
        self.photoImage = ImageTk.PhotoImage(self.image)
        self.lbl_Image.configure(image = self.photoImage)

    #Método para converter a imagem para tons de cinza
    def cinza(self):
        if self.image != None:
            self.default = self.image
            aux = np.array(self.image)
            gray = cv2.cvtColor(aux, cv2.COLOR_BGR2GRAY)
            self.convertTkinter(gray)

    #Método para quantizar a imagem
    def quantizacao(self):
        if self.image != None:
            self.default = self.image

    #Método para carregar base de dados mnist
    def load_mnist_dataset(self):
        # load the MNIST dataset and stack the training data and testing
        # data together (we'll create our own training and testing splits
        # later in the project)
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        # return a 2-tuple of the MNIST data and labels

        print('X_train: ' + str(train_X.shape))
        print('Y_train: ' + str(train_y.shape))
        print('X_test:  '  + str(test_X.shape))
        print('Y_test:  '  + str(test_y.shape))
        # return (data, labels)

    #Método para criação do Canvas e menus
    def Widgets(self):
        self.master.attributes("-fullscreen", True)           

        menu = tk.Menu(self.master)
        self.master.config(menu = menu)

        #Menu arquivo
        optionMenu = tk.Menu(menu, tearoff = 0)
        menu.add_cascade(label = "Arquivo", menu = optionMenu)
        optionMenu.add_command(label = "Inserir", command = self.inserir)
        optionMenu.add_command(label = "Desfazer", command = self.desfazer)
        optionMenu.add_separator()
        optionMenu.add_command(label = "Sair", command = self.master.destroy)

        #Submenu Quantização
        quantMenu = tk.Menu(menu, tearoff = 0)
        quantMenu.add_command(label = '128', command = self.load_mnist_dataset)
        #quantMenu.add_command(label = '64', command = self.quantizacao)
        #quantMenu.add_command(label = '32', command = self.quantizacao)
        #quantMenu.add_command(label = '16', command = self.quantizacao)

        #Menu Ferramentas
        toolsMenu = tk.Menu(menu, tearoff = 0)
        menu.add_cascade(label = "Ferramentas", menu = toolsMenu)
        toolsMenu.add_command(label = "Tons de Cinza", command = self.cinza)
        toolsMenu.add_cascade(label = 'Quantização', menu = quantMenu)

        #Label da imagem
        self.lbl_Image = tk.Label(self.master)
        self.lbl_Image.pack()


#Criação do objeto Application e loop principal  
root = tk.Tk()
Application(root)
root.title("Processamento de Imagens") #Nome da janela
root.mainloop()