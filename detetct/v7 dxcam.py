import mss
import numpy as np
import cv2
import torch
import time
import dxcam

#yolo part
model = torch.hub.load('WongKinYiu/yolov7','custom','migliore.pt',force_reload=True)
model.conf = 0.35
model.classes  = [0] # which classe the model detects
model.maxdet = 100
#model.apm = True

#dxcampart
left, top = (1920 - 640) // 2, (1080 - 640) // 2
right, bottom = left + 640, top + 640
region = (left, top, right, bottom)
camera = dxcam.create(region=region)
camera.start(target_fps=160, video_mode=True)

while True:
        t = time.time()
        img = np.array(camera.get_latest_frame())
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = model(img)
        results.render()
        enemyNum = results.xyxy[0].shape[0]
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(out, f"FPS: {int(1 / (time.time() - t))}", (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
        cv2.putText(out, f"NUM: {int(enemyNum)}", (155, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (113, 116, 244), 2)
        cv2.imshow('s', out)

        if cv2.waitKey(1) == 27:
            break
    
cv2.destroyAllWindows()
