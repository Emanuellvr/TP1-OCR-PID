import numpy as np
import tkinter as tk
import cv2
from PIL import ImageTk, Image
from tkinter import Frame, filedialog
#from keras.datasets import mnist           #pip install tensorflow     pip install keras
#from matplotlib import pyplot

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
            #self.image = Image.open(filename)
            #self.atualizarTela()
            self.image = cv2.imread(filename)
            self.convertTkinter(self.image)
    
    #Método para voltar a imagem anterior
    def desfazer(self):
        self.image = self.default
        self.atualizarTela()

    #Método para converter a imagem para abrir no Tkinter
    def convertTkinter(self, image):
        try:
            b,g,r = cv2.split(image)
            image = cv2.merge((r,g,b))
        except:
            print("Conversão inválida (Imagem em tons de cinza).")

        #self.image = Image.fromarray(image)
        self.image = image
        self.atualizarTela()

    #Método para atualizar a label com a imagem
    def atualizarTela(self):    
        self.photoImage = ImageTk.PhotoImage(Image.fromarray(self.image))
        self.lbl_Image.configure(image = self.photoImage)

    #Método para converter a imagem para tons de cinza
    def cinza(self):
        try:
            self.default = self.image
            #aux = np.array(self.image)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.convertTkinter(gray)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para quantizar a imagem
    def quantizacao(self, n):
        try:
            self.default = self.image
            aux = np.float32(self.image)
            bucket = 256 / n
            quantizacao = aux / bucket
            quantizado = np.uint8(quantizacao) * bucket
            self.convertTkinter(quantizado)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para remoção de ruído
    def ruido(self):
        try:
            self.default = self.image
            #aux = np.array(self.image)
            noise = cv2.medianBlur(self.image, 5)
            self.convertTkinter(noise)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para binarizar a imagem
    def binarizacao(self):
        try:
            self.default = self.image
            #aux = np.array(self.image)
            thresholding = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.convertTkinter(thresholding)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para criação do Canvas e menus
    def Widgets(self):
        self.master.attributes("-fullscreen", True)           

        frameMain = Frame(self.master)
        frameMain.pack()

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
        quantMenu.add_command(label = '128', command = lambda: self.quantizacao(128))
        quantMenu.add_command(label = '64', command = lambda: self.quantizacao(64))
        quantMenu.add_command(label = '32', command = lambda: self.quantizacao(32))
        quantMenu.add_command(label = '16', command = lambda: self.quantizacao(16))
        quantMenu.add_command(label = '8', command = lambda: self.quantizacao(8))
        quantMenu.add_command(label = '4', command = lambda: self.quantizacao(4))
        quantMenu.add_command(label = '2', command = lambda: self.quantizacao(2))

        #Menu Ferramentas
        toolsMenu = tk.Menu(menu, tearoff = 0)
        menu.add_cascade(label = "Ferramentas", menu = toolsMenu)
        toolsMenu.add_command(label = "Tons de cinza", command = self.cinza)
        toolsMenu.add_cascade(label = 'Quantização', menu = quantMenu)
        toolsMenu.add_command(label = "Remoção de ruído", command = self.ruido)
        toolsMenu.add_command(label = "Binarização", command = self.binarizacao)

        #Label da imagem
        self.lbl_Image = tk.Label(frameMain)
        self.lbl_Image.pack()


#Criação do objeto Application e loop principal  
root = tk.Tk()
Application(root)
root.title("Processamento de Imagens") #Nome da janela
root.mainloop()