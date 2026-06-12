from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import threading
import queue

# TRACKER_CONFIG = "bytetrack.yaml"
# TRACKER_CONFIG = "botsort.yaml"
# TRACKER_CONFIG = "ocsort.yaml"
#TRACKER_CONFIG = "tracktrack.yaml"

QUEUE_SIZE = 30
DETECTION_INTERVAL = 5
DSIZE=(1280,720)
VIDEO_PATH = "md1.mp4"

track_history = defaultdict(lambda: [])
frame_counter = 0
out=None

frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
result_queue = queue.Queue(maxsize=QUEUE_SIZE)
running = True

model = YOLO("yolov8n.pt")
video_path = "md1.mp4"

#1поток:заъват кадров
def capture_thread_func(path):
    cap = cv2.VideoCapture(path)
    global running, frame_counter

    if not cap.isOpened():
        print(f"ошибка открытия {path}")
        running = False
        return
    '''
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # количество кадров в секунду
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # ширина кадра в пикселях
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # высота кадра
   '''

    local_frame_counter = 0
    while running:
        success, frame = cap.read()
        if not success:
            print("конец света")
            break

        local_frame_counter += 1
        frame: np.ndarray

        frame = cv2.resize(src=frame, dsize=DSIZE, interpolation=cv2.INTER_NEAREST)

        try:
            frame_queue.put((frame_counter, frame), timeout=0.5)
        except queue.Full:
            try:
                frame_queue.get_nowait()
                frame_queue.put((local_frame_counter, frame))
            except queue.Empty:
                pass

    cap.release()
    frame_queue.put(None)


#2поток для детекции и трекинга
def process_thread_func():
    global running, track_history, model, out

    while running:
        item = frame_queue.get()
        if item is None:
            result_queue.put(None)
            break

        frame_idx, frame = item

        try:

            if frame_idx % DETECTION_INTERVAL == 1:


            #трекинг на текущем кадре, с сохранением треков между кадрами
                results = model.track(
                    frame,
                    persist=True,
                    conf=0.3,
                    iou=0.5,
                    verbose=False)
            #tracker=TRACKER_CONFIG)
            else:
                results = model.track(frame, persist=True, conf=0.3, iou=0.5, verbose=False)

            if results is not None and len(results) > 0:
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

                    result_queue.put(annotated_frame)

                else:
                    result_queue.put(frame)
            else:
                # results пустой
                result_queue.put(frame)

        except Exception as e:
            print(f"Ошибка в process_thread_func: {e}")
            result_queue.put(frame)



def main():
    global running, out, DSIZE
    #video_path = "md1.mp4"

    capture_thread = threading.Thread(target=capture_thread_func, args=(VIDEO_PATH,))
    process_thread = threading.Thread(target=process_thread_func, args=())

    capture_thread.start()
    process_thread.start()


    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(
        "output_track.mp4",
        fourcc,
        30,
        DSIZE
    )

    '''
    while cap.isOpened():
        success, frame = cap.read()  # считывает кадр из видео
        if not success:
            print("конец видео")
            break

        
            results = model.track(source="path/to/video.mp4", tracker="bytetrack.yaml")
            results = model.track(source="path/to/video.mp4", tracker="ocsort.yaml")
            results = model.track(source="path/to/video.mp4", tracker="tracktrack.yaml")
            
            results = model.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")
            '''
    print("Начинаем отображение...")
    frame_count = 0

    while running:
        try:
            frame = result_queue.get(timeout=1)
            item = result_queue.get(timeout=1)

            if frame is None:
                print("Получен сигнал завершения")
                break

            if isinstance(item, np.ndarray):
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Показано кадров: {frame_count}")
                cv2.imshow("Tracking", frame)
                if out is not None:
                    out.write(frame)

            else:
                print(f"ПРЕДУПРЕЖДЕНИЕ: получен не кадр, а {type(item)}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

        except queue.Empty:
            continue

    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    capture_thread.join()
    process_thread.join()

if __name__ == "__main__":
    main()