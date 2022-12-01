# Importar as dependecias do Kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Importar os componentes UX do Kivy
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Importar dependencias auxiliares do Kivy
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Importar outras dependencias úteis
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Criar o app e o layout do app
class CamApp(App):

    def build(self):
        # Aqui nós temos os componentes que serão exibidos: a imagem da câmera, o botão de verificação e o botão de resultado da verificação. 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Aqui nós adicionamos os botões ao layout do app.
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Carregamos o modelo salvo e treinado.
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

        # Aqui utilizamos a câmera para captura da imagem.
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Esse código roda de forma contínua para capturar e atualizar o frame obtido da webcam
    def update(self, *args):

        # Faz a leitura do frame pelo OpenCv
        ret, frame = self.capture.read()
        
        frame = frame[120:120+250, 200:200+250, :]

        # Inverte a imagem para horizontalmente e converte a imagem para texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Esse método carrega a imagem do arquivo e converte ela para 100x100px
    def preprocess(self, file_path):
        # Faz a leitura da imagem do arquivo
        byte_img = tf.io.read_file(file_path)
        # Carrega a imagem
        img = tf.io.decode_jpeg(byte_img)
        
        # Faz conversão da imagem para a resolução 100x100x3
        img = tf.image.resize(img, (100,100))
        # Transforma a imagem em uma escala entre 0 e 1.
        img = img / 255.0
        
        # Retorna a imagem
        return img

    # Esse método serve para imprimir o progresso da predição do modelo.
    def print_progress(self, value):
        print(f'Verifying: {value}%')

    # Método de verificação
    def verify(self, *args):
        # Especifica o limiar de verificação
        detection_threshold = 0.99
        verification_threshold = 0.7
        # Captura a imagem da webcam e salva no diretório 'input_image'
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        
        # No código abaixo iremos percorres as imagens do diretório verification_images
        # Para cada imagem desse diretório iremos pedir para o modelo realizar a predição comparando a imagem com a imagem recebida da câmera
        # Para cada predição iremos guardar os resultados preditos pelo modelo para verificarmos depois
        results = []
        progress = 0
        total = len(os.listdir(os.path.join('application_data', 'verification_images')))
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            if (progress/total) % 0.10 == 0:
                self.print_progress((progress/total) * 100)
            progress += 1
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
            
        
        # Após coletar os resultados obtidos pelas predições do modelo
        # Fazemos a comparação do resultado obtido com o nosso limiar(threshold) definido anteriormente
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Fazemos a verificação da seguinte forma: Proporção das predições positivas / total de imagens de exemplo 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        # Dizemos que é verificado se o valor de verificação obtido for maior que o nosso limiar(threshold) definido anteriormente
        verified = verification > verification_threshold

        # Se for verificado exibimos no nosso layout o valor 'Verified', caso contrário exibimos 'Unverified'
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Mostramos o log dos valores obtidos
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified



if __name__ == '__main__':
    CamApp().run()
