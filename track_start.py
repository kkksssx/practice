from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import threading
import queue
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from boxmot.trackers.ocsort.ocsort import OcSort
import sys

# TRACKER_CONFIG = "bytetrack.yaml"
# TRACKER_CONFIG = "botsort.yaml"
# TRACKER_CONFIG = "ocsort.yaml"
#TRACKER_CONFIG = "tracktrack.yaml"

#минимальные размеры для фильтрации ложных срабатываний
MIN_BOX_WIDTH = 20      #мин ширина бокса в пикселях
MIN_BOX_HEIGHT = 35     #мин высота бокса в пикселях
MIN_BOX_AREA = 700      #мин площадь бокса
MAX_BOX_AREA = 200000
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.2
MIN_CONFIDENCE = 0.25

QUEUE_SIZE = 0
DETECTION_INTERVAL = 5
DSIZE=(1280,720)
VIDEO_PATH = "md1.mp4"

SLICE_HEIGHT = 480      # Высота каждого куска (тайла)
SLICE_WIDTH = 480      # Ширина каждого куска
OVERLAP_RATIO = 0.25     # Перекрытие между кусками

CONFIDENCE_THRESHOLD = 0.25 # уверенность(умен, если не находит)

track_history = defaultdict(lambda: [])
out=None

frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
result_queue = queue.Queue(maxsize=QUEUE_SIZE)
running = True

model_path = "yolov8s_openvino_model/"
model = YOLO(model_path)

# SAHI модель для детекции
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    device="cpu",
)

# BoxMOT трекер
tracker = OcSort(
    det_thresh=0.1,      # Порог уверенности для детекций
    max_age=20,           # Максимальный возраст трека в кадрах
    min_hits=1,           # Минимум детекций для подтверждения трека
    iou_threshold=0.3,    # Порог IoU для связывания
    obs_thresh=0.5,       # Порог уверенности для наблюдения
    #track_thresh=0.1,     # Порог для уверенных треков
    #match_thresh=0.8,     # Порог для сопоставления
    #track_buffer=30,      # Буфер для потерянных треков
    #frame_rate=6          # Частота кадров 30 FPS видео / 5 (интервал детекции) = 6 FPS для трекера
)

video_path = "md1.mp4"

#повышение контраста и резкости кадра
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    gaussing = cv2.GaussianBlur(enhanced, (0,0), 2)
    sharpened = cv2.addWeighted(enhanced, 1.3, gaussing, -0.3, 0)

    return sharpened


#1поток:заъват кадров
def capture_thread_func(path):
    cap = cv2.VideoCapture(path)
    global running

    if not cap.isOpened():
        print(f"ошибка открытия {path}")
        running = False
        return
    '''
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # количество кадров в секунду
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # ширина кадра в пикселях
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # высота кадра
   '''
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Исходная частота кадров видео: {video_fps} FPS")

    local_frame_counter = 0

    while running:
        success, frame = cap.read()
        if not success:
            print("конец света")
            break

        local_frame_counter += 1
        #frame: np.ndarray

        frame = cv2.resize(src=frame, dsize=DSIZE, interpolation=cv2.INTER_NEAREST)

        frame_queue.put((local_frame_counter, frame))

    cap.release()
    frame_queue.put(None)


#2поток для детекции и трекинга
def process_thread_func():
    global running, track_history, model, out
    process_counter = 0
    processed_counter = 0
    last_tracks = [] #сохранение последних известных треков для отрисовки на промежуточных кадрах

    while running:
        item = frame_queue.get()
        if item is None:
            result_queue.put(None)
            break

        frame_idx, frame = item
        process_counter += 1
        processed_counter += 1

        if process_counter % DETECTION_INTERVAL == 1:

            enhanced_frame = enhance_frame(frame)
             #трекинг на текущем кадре, с сохранением треков между кадрами

            result = get_sliced_prediction(
                enhanced_frame,
                sahi_model,
                slice_height=SLICE_HEIGHT,
                slice_width=SLICE_WIDTH,
                overlap_height_ratio=OVERLAP_RATIO,
                overlap_width_ratio=OVERLAP_RATIO,
                postprocess_type="GREEDYNMM",  # Тип алгоритма
                postprocess_match_metric="IOU",  # Метрика сравнения
                postprocess_match_threshold=0.3
            )

            detections = []  ## перевод SAHI результат в формат для BoxMOT
            for obj in result.object_prediction_list:
                if obj.category.name == 'person':
                    bbox = obj.bbox.to_xyxy()
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = width / max(height, 1)

                    if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
                        continue
                    if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
                        continue
                    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                        continue
                    if obj.score.value < MIN_CONFIDENCE:
                        continue

                    detections.append([
                         bbox[0], bbox[1], bbox[2], bbox[3],
                         obj.score.value,
                         0  # класс айди для людей
                    ])
            #обновление трекера новыми детекциями
            if len(detections) > 0:
                 tracks = tracker.update(np.array(detections), frame)
                 print(f"Кадр {frame_idx}: найдено {len(detections)} человек, треков: {len(tracks)}")
            else:
                 tracks = tracker.update(np.empty((0, 6)), frame)
                 print(f"Кадр {frame_idx}: людей не найдено")

            last_tracks = tracks

            for track in tracks:
                 x1, y1, x2, y2, track_id, conf, cls = track[:7]
                 center_x = (x1 + x2) // 2
                 center_y = (y1 + y2) // 2
                 track_line = track_history[track_id]
                 track_line.append((center_x, center_y))
                 if len(track_line) > 30:
                     track_line.pop(0)

        else:
            #просто использует сохраненные треки, чтобы они не исчезали
            tracks = tracker.update(np.empty((0, 6)), frame)

            if len(tracks) == 0 and len(last_tracks) > 0:
                tracks = last_tracks
            else:
                last_tracks = tracks


        annotated_frame = frame.copy()
            #persons_found = 0

        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls = track[:7]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)

                    #цвет для каждого ID (уникальный)
                color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)


                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID:{track_id} ({conf:.2f})",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #рис траекторию
                track_line = track_history[track_id]
                if len(track_line) > 1:
                    points = np.array(track_line).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False,
                                  color=(230, 230, 230), thickness=3)

        if process_counter % DETECTION_INTERVAL == 1:
            cv2.putText(annotated_frame, f"DETECTION FRAME (SAHI)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, f"TRACKING ONLY",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(annotated_frame, f"Frame: {frame_idx}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        result_queue.put(annotated_frame)


def main():
    global running, out, DSIZE
    #video_path = "md1.mp4"

    temp_cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
    temp_cap.release()

    print(f"Частота кадров видео: {video_fps} FPS")

    capture_thread = threading.Thread(target=capture_thread_func, args=(VIDEO_PATH,), daemon=True)
    process_thread = threading.Thread(target=process_thread_func, args=(), daemon=True)

    capture_thread.start()
    process_thread.start()


    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(
        "output_track.avi",
        fourcc,
        video_fps,
        DSIZE
    )

    print("Начинаем отображение...")
    frame_count = 0

    while running:
        try:
            frame = result_queue.get(timeout=0.1)

            if frame is None:
                print("Получен сигнал завершения")
                break

            if isinstance(frame, np.ndarray):
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Показано кадров: {frame_count}")
                cv2.imshow("Tracking", frame)
                if out is not None:
                    out.write(frame)


        except queue.Empty:
             pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n Прервано пользователем (нажата 'q')")
            running = False
            if out is not None:
                out.release()
                print(f"Видео сохранено: output_track.avi ({frame_count} кадров)")
            cv2.destroyAllWindows()
            sys.exit(0)

    print("Ожидание завершения потоков...")
    running = False

    capture_thread.join()
    process_thread.join()

    if out is not None:
        out.release()
        print(f"Видео сохранено: output_track.mp4 ({frame_count} кадров)")

    cv2.destroyAllWindows()
    print("Готово!")

if __name__ == "__main__":
    main()