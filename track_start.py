from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

# TRACKER_CONFIG = "bytetrack.yaml"
# TRACKER_CONFIG = "botsort.yaml"
# TRACKER_CONFIG = "ocsort.yaml"
#TRACKER_CONFIG = "tracktrack.yaml"
DETECTION_INTERVAL = 5
model = YOLO("yolov8n.pt")

video_path = "md1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ошибка открытия {video_path}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  # количество кадров в секунду
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # ширина кадра в пикселях
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # высота кадра

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output_track.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

track_history = defaultdict(lambda: [])
frame_counter = 0

while cap.isOpened():
    success, frame = cap.read()  # считывает кадр из видео
    if not success:
        print("конец видео")
        break

    '''
        results = model.track(source="path/to/video.mp4", tracker="bytetrack.yaml")
        results = model.track(source="path/to/video.mp4", tracker="ocsort.yaml")
        results = model.track(source="path/to/video.mp4", tracker="tracktrack.yaml")
        
        results = model.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")
        '''
    if frame_counter % DETECTION_INTERVAL == 1:
    #трекинг на текущем кадре, с сохранением треков между кадрами
        results = model.track(frame,
                             persist=True,
                            conf=0.3,
                            iou=0.5,
                            verbose=False)
                          #tracker=TRACKER_CONFIG)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
            track_ids = results[0].boxes.id.cpu().tolist()  # идентификаторы треков

            annotated_frame = results[0].plot()  # визуализация в кадре

        # отрисовка треков
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box  # координаты центра и размеры бокса
                track = track_history[track_id]
                track.append((float(x), float(y)))  # добавление координат центра объекта в историю
                if len(track) > 30:  # ограничение длины истории до 30 кадров
                    track.pop(0)

            # рисование линий трека
                if len(track) > 1:
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

            cv2.imshow("Tracking", annotated_frame)  # отображение в отдельном окне
            out.write(annotated_frame)  # запись обратн кадра
        else:
            cv2.imshow("Tracking", frame)
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()