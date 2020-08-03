#Información
#Port = 9080
#IP= 127.0.0.1 || LocalHost

#Godot 3.2



#¿Como funciona?'
# > Iniciar el proyecto en godot para lanzar el servidor
# >Despues ejecutar este script 




#

 #Area de Modulos
from threading import *
import cv2 as cv
import numpy as np
import asyncio
import websockets
import time

#Diccionarios y listas

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


def Rutina():
	#Cargamos el modelo pre-entrenado MOBILENET 
	net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

	#Iniciamos la lectura de la webcam
	cap = cv.VideoCapture(0)
	
	while cv.waitKey(1) < 0:
		ret, frame = cap.read()
		
		#Obtenemos sus dimensiones
		frameWidth = frame.shape[1]
		frameHeight = frame.shape[0]
		
		#Configuramos la dnnn
		net.setInput(cv.dnn.blobFromImage(frame, 1.0, (frameWidth, frameHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
		out = net.forward()
		
		# El modelo MobileNet arroja [1, 57, -1, -1], solo ocupamos los primeros 19 puntos [Diccionario]
		out = out[:, :19, :, :]
		points = []
		assert(len(BODY_PARTS) == out.shape[1])
		for i in range(len(BODY_PARTS)):
			heatMap = out[0, i, :, :]
			#Solo podremos detectar 1 postura a la vez
			_, conf, _, point = cv.minMaxLoc(heatMap)
			x = (frameWidth * point[0]) / out.shape[3]
			y = (frameHeight * point[1]) / out.shape[2]
			# Agrega un punto si su semejanza es mayor que el limite
			points.append((int(x), int(y)) if conf > 0.3 else None)
			#print(BODY_PARTS)

		for pair in POSE_PAIRS:
			partFrom = pair[0]
			partTo = pair[1]
			assert(partFrom in BODY_PARTS)
			assert(partTo in BODY_PARTS)

			idFrom = BODY_PARTS[partFrom]
			idTo = BODY_PARTS[partTo]

			if points[idFrom] and points[idTo]:
				cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
				cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
				cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

		t, _ = net.getPerfProfile()

		cv.imshow('Avatar1', frame)
		print(points)


async def Puntos(uri):
    async with websockets.connect(uri) as websocket:
        mensaje="Hola"
        for i in range(1, 100):
            await websocket.send(mensaje)
            await websocket.recv()
            time.sleep(1)
		    
def IniciarSocket():
	asyncio.get_event_loop().run_until_complete(Puntos('ws://127.0.0.1:9080'))
	

		

r=Thread(target= Rutina())
s=Thread(target= IniciarSocket())


s.start
r.start

s.join()
r.join()


#La parte de la comunicación Python-Godot funciona, el multithreding no funciona  por los while loops , investigar sobre multiprocesamiento (?)
