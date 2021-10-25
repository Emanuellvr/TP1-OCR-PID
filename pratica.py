import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import Frame, filedialog
from numpy.core.defchararray import asarray
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import interpolation as inter
from matplotlib import pyplot as plt
# from keras.datasets import mnist           #pip install tensorflow     pip install keras


'''
Git:
git pull                        Atualiza com a versão do github.
git add 'nomeDoArquivo'         Prepara o arquivo para ser enviado.
git commit -m "Comentário"      Comita o arquivo.
git push                        Envia o arquivo para o github.

Erros:
Rever quantização (uma solução pra colorida e uma solução para preto e branco) {
    Na quantização para a imagem preto e branco: Remoção de ruído após quantização.
}
Na projeção:
os valores da matriz possuem um ponto no final. Não sei se isso ocorre pelos números serem float, ou se é erro
'''

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
        filename = filedialog.askopenfilename(title="Select a image", filetypes=fileTypes)

        if filename != '':
            self.image = cv2.imread(filename)
            self.cinza()
            self.binarizacao()
            self.convertTkinter(self.image)

    #Método para voltar a imagem anterior
    def desfazer(self):
        self.image = self.default
        self.convertTkinter(self.image)

    #Método para converter a imagem para abrir no Tkinter
    def convertTkinter(self, image):
        self.image = image
        try:
            b, g, r = cv2.split(image)
            image = cv2.merge((r, g, b))
        except:
            print("Conversão inválida (Imagem em tons de cinza).")

        self.atualizarTela(image)

    #Método para atualizar a label com a imagem
    def atualizarTela(self, image):
        self.photoImage = ImageTk.PhotoImage(Image.fromarray(image))
        self.lbl_Image.configure(image=self.photoImage)

    #Método para converter a imagem para tons de cinza
    def cinza(self):
        try:
            self.default = self.image
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray
            #self.convertTkinter(gray)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para quantizar a imagem (preto e branco)
    def quantizacao(self, n):
        try:
            self.default = self.image
            aux = np.float32(self.image)
            bucket = 256 / n
            quantizacao = aux / bucket
            quantizado = np.uint8(quantizacao) * bucket
            self.convertTkinter(np.array(quantizado))
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    '''
    #Método para quantizar a imagem (colorida)
    def quantizacao(self, n):
        #try:
            self.default = self.image
            (h, w) = self.image.shape[:2]
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            image = image.reshape((image.shape[0] * image.shape[1], 3))
            clt = MiniBatchKMeans(n)
            labels = clt.fit_predict(image)
            quantizado = clt.cluster_centers_.astype("uint8")[labels]
            quantizado = quantizado.reshape((h, w, 3))
            quantizado = cv2.cvtColor(quantizado, cv2.COLOR_LAB2BGR)
            #image = image.reshape((h, w, 3))
            #image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
            self.convertTkinter(quantizado)
        #except:
            #print("Erro: Nenhuma imagem foi selecionada.")
    '''

    #Método para remoção de ruído
    def ruido(self):
        try:
            self.default = self.image
            noise = cv2.medianBlur(self.image, 5)
            self.convertTkinter(noise)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para binarizar a imagem
    def binarizacao(self):
        try:
            self.default = self.image
            thresholding = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.image = thresholding
            #self.convertTkinter(thresholding)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para extrair a projeção horizontal da imagem
    def projHorizontal(self):
        try:
            print(np.sum(self.image, axis=1, keepdims=True) / 255)
            return np.sum(self.image, axis=1, keepdims=True) / 255
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para extrair a projeção vertical da imagem
    def projVertical(self):
        try:
            print(np.sum(self.image, axis=0, keepdims=True) / 255)
            return np.sum(self.image, axis=0, keepdims=True) / 255
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para extrair as projeções da imagem e concatená-las
    def projecao(self):
        try:
            self.default = self.image
            horizontal = self.projHorizontal()
            vertical = self.projVertical()
            concatenate = np.concatenate((np.array(horizontal).transpose(), np.array(vertical)), axis=None)
            print(concatenate)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para inverter a paleta de cores da imagem
    def inverterTons(self):
        try:
            self.default = self.image
            invert = 255 - self.image
            self.convertTkinter(invert)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para extração do histograma da imagem
    def find_score(self, arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    #Método para correção do ângulo das imagens
    def correcaoAngulo(self):
        try:
            self.default = self.image
            image = self.image
            delta = 1
            limit = 30
            angles = np.arange(-limit, limit + delta, delta)
            scores = []
            for angle in angles:
                hist, score = self.find_score(image, angle)
                scores.append(score)
            best_score = max(scores)
            best_angle = angles[scores.index(best_score)]
            data = inter.rotate(image, best_angle, reshape=False, order=0)
            self.convertTkinter(data)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para erosão da imagem
    def erosao(self):
        try:
            self.default = self.image
            kernel = np.ones((5, 5), np.uint8)
            erode = cv2.erode(self.image, kernel, iterations=1)
            self.convertTkinter(erode)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para rotacionar a imagem 90 graus para a esquerda
    def rot90anti(self):
        try:
            self.default = self.image
            rotate = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.convertTkinter(rotate)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para rotacionar a imagem 90 graus para a direita
    def rot90hor(self):
        try:
            self.default = self.image
            rotate = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.convertTkinter(rotate)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para inverter a imagem horizontalmente
    def invertHor(self):
        try:
            self.default = self.image
            invert = cv2.flip(self.image, 0)
            self.convertTkinter(invert)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para inverter a imagem verticalmente
    def invertVert(self):
        try:
            self.default = self.image
            invert = cv2.flip(self.image, 1)
            self.convertTkinter(invert)
        except:
            print("Erro: Nenhuma imagem foi selecionada.")

    #Método para criação do Canvas e menus
    def Widgets(self):
        self.master.attributes("-fullscreen", True)

        frameMain = Frame(self.master)
        frameMain.pack()

        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        #Menu arquivo
        optionMenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Arquivo", menu=optionMenu)
        optionMenu.add_command(label="Inserir", command=self.inserir)
        optionMenu.add_command(label="Desfazer", command=self.desfazer)
        optionMenu.add_separator()
        optionMenu.add_command(label="Sair", command=self.master.destroy)

        #Submenu Quantização
        quantMenu = tk.Menu(menu, tearoff=0)
        quantMenu.add_command(label="128", command=lambda: self.quantizacao(128))
        quantMenu.add_command(label="64", command=lambda: self.quantizacao(64))
        quantMenu.add_command(label="32", command=lambda: self.quantizacao(32))
        quantMenu.add_command(label="16", command=lambda: self.quantizacao(16))
        quantMenu.add_command(label="8", command=lambda: self.quantizacao(8))
        quantMenu.add_command(label="4", command=lambda: self.quantizacao(4))
        quantMenu.add_command(label="2", command=lambda: self.quantizacao(2))

        #SubMenu Projeção
        projMenu = tk.Menu(menu, tearoff=0)
        projMenu.add_command(label="Horizontal", command=self.projHorizontal)
        projMenu.add_command(label="Vertical", command=self.projVertical)
        projMenu.add_command(label="Horizontal + Vertical", command=self.projecao)

        #Menu Ferramentas
        toolsMenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Ferramentas", menu=toolsMenu)
        #toolsMenu.add_command(label="Tons de cinza", command=self.cinza)
        #toolsMenu.add_command(label="Binarização", command=self.binarizacao)
        #toolsMenu.add_cascade(label='Quantização', menu=quantMenu)
        toolsMenu.add_command(label="Remoção de ruído", command=self.ruido)
        toolsMenu.add_command(label="Erosão", command=self.erosao)
        toolsMenu.add_command(label="Inverter tons", command=self.inverterTons)
        toolsMenu.add_command(label="Correção de ângulo", command=self.correcaoAngulo)
        toolsMenu.add_cascade(label="Projeção", menu=projMenu)

        #Submenu Rotação
        rotationMenu = tk.Menu(menu, tearoff=0)
        rotationMenu.add_command(label="90° esquerda", command=self.rot90anti)
        rotationMenu.add_command(label="90° direita", command=self.rot90hor)

        #Submenu Eixo
        flipMenu = tk.Menu(menu, tearoff=0)
        flipMenu.add_command(label="Horizontal", command=self.invertHor)
        flipMenu.add_command(label="Vertical", command=self.invertVert)

        #Menu Visualisação
        visualMenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Visualização", menu=visualMenu)
        visualMenu.add_cascade(label="Rotação", menu=rotationMenu)
        visualMenu.add_cascade(label="Inverter eixo", menu=flipMenu)

        #Label da imagem
        self.lbl_Image = tk.Label(frameMain)
        self.lbl_Image.pack()

#Criação do objeto Application e loop principal
root = tk.Tk()
Application(root)
root.title("Processamento de Imagens")  # Nome da janela
root.mainloop()