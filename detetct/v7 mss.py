import mss
import numpy as np
import cv2
import torch
import time


model = torch.hub.load(r'C:\Users\winz1\Desktop\yolov7', 'custom',r'C:/Users/winz1/Desktop/detetct v7/vc.pt', source='local')
model.conf = 0.3
#model.classes  = [0] # which classe the model detects
model.maxdet = 100
model.apm = True

with mss.mss() as sct:
    monitor = {'top': 330, 'left': 640, 'width': 700, 'height': 500}

    while True:
        t = time.time()

        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        results.render()
        enemyNum = results.xyxy[0].shape[0]
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"FPS: {int(1 / (time.time() - t))}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
        cv2.putText(img, f"NUM: {int(enemyNum)}", (155, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
        cv2.imshow("img", img)
        
       

        

        if(cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()
            break
