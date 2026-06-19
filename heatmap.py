import  cv2
import numpy as np
from collections import deque

class HeatmapGenerator:

    def __init__(self, width=640, height=480, decay=0.999, radius=15):
        self.width = width
        self.height = height
        self.decay = decay
        self.radius = radius
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.points = []
        self.colors = []
        self.track_trails = {}
        self.all_time_points = []
        self.max_all_points = 10000
        # Матрица накопленной интенсивности (для нижней карты)
        self.intensity_map = np.zeros((height, width), dtype=np.float32)
        self.intensity_max = 1.0  #текущий максимум для нормализации

    def add_points(self, points_topdown):
        self.points = points_topdown

        #точки в постоянное хранилище верхней карты
        for x, y in points_topdown:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.all_time_points.append((int(x), int(y)))

            #ограниченный размер хранилища
        if len(self.all_time_points) > self.max_all_points:
            self.all_time_points = self.all_time_points[-self.max_all_points:]

            # Создаём маску текущих точе/нижняя карта
        current_mask = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in points_topdown:
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(current_mask, (int(x), int(y)), self.radius, 1.0, -1)

            # КЛЮЧЕВОЕ: Просто ДОБАВЛЯЕМ маску к интенсивности (без затухания!)
        self.intensity_map += current_mask

            # Обновляем максимум для нормализации
        current_max = np.max(self.intensity_map)
        if current_max > self.intensity_max:
            self.intensity_max = current_max

        self.heatmap = self.intensity_map / max(self.intensity_max, 1.0)

    def get_colored_heatmap(self):
        #нормализация по текущему максимуму
        if self.intensity_max > 0:
            normalized = self.intensity_map / self.intensity_max
        else:
            normalized = self.intensity_map

        heatmap_uint8 = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return heatmap_color

    def get_points_overlay(self, color =(255,0,255), size =5):
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for x, y in self.points:
            if 0 <= x < self.width and 0 <= y < self.height:
                 cv2.circle(overlay, (int(x), int(y)), size, color, -1)

        return overlay

    #комбинированный вид heatmap + точки
    def get_combined_view(self, alpha =0.7):

        heatmap_color = self.get_colored_heatmap()
        points_overlay = self.get_points_overlay()
        combined = cv2.addWeighted(heatmap_color, 1.0, points_overlay, alpha, 0)
        return combined

    #разделённый вид: верх - точки+траектории, низ - тепловая карта
    def get_split_view(self, point_color=(255, 0, 255), point_size=5, trail_thickness=2):
        split_view = np.zeros((self.height * 2, self.width, 3), dtype=np.uint8)
        #все точки
        for x, y in self.all_time_points:
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(split_view[:self.height], (x, y), point_size, point_color, -1)

        for track_id, points_list in self.track_trails.items():
            if len(points_list) > 1:
                points_array = np.array(points_list, dtype=np.int32).reshape((-1, 1, 2))
                # Все траектории одного цвета
                cv2.polylines(split_view[:self.height], [points_array], isClosed=False,
                              color=point_color, thickness=trail_thickness)

        #текущие точки
        for x, y in self.points:
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(split_view[:self.height], (int(x), int(y)), point_size, point_color, -1)

        #разделитель
        cv2.line(split_view, (0, self.height), (self.width, self.height), (255, 255, 255), 2)

        #Тепловая карта
        heatmap_color = self.get_colored_heatmap()
        split_view[self.height:, :] = heatmap_color

        return split_view

    def reset(self):
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.points = []

    def get_heatmap_array(self):
        return self.heatmap.copy()

#вид сверху
class PerspectiveTransformer:

    def __init__(self, src_points, dst_points):
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def transform_point(self, x, y):
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)
        return transformed[0][0]

    def transform_points(self, points):
        if len(points) == 0:
            return []
        points_array = np.array([points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points_array, self.matrix)
        return transformed[0].tolist()

    def inverse_transform_point(self, x, y):
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_matrix)
        return transformed[0][0]

#трансформер по умолчанию (для типичной камеры сверху)
def create_default_transformer(frame_shape, heatmap_width=640, heatmap_height=480):
    h, w = frame_shape[:2]

    # Примерные точки/нужно подстроить под камеру!
    src_points = [
        [w * 0.1, h * 0.4],  # Левый верх
        [w * 0.9, h * 0.4],  # Правый верх
        [w * 0.9, h * 0.9],  # Правый низ
        [w * 0.1, h * 0.9],  # Левый низ
    ]

    dst_points = [
        [0, 0],
        [heatmap_width, 0],
        [heatmap_width, heatmap_height],
        [0, heatmap_height],
    ]

    return PerspectiveTransformer(src_points, dst_points)

#overlay с heatmap для данного кадра
def create_heatmap_overlay(heatmap_gen, transformer, tracks, track_history, frame_shape, heatmap_size=(640, 480)):

    current_points_topdown = []

    #сбор точек из текущих треков
    if len(tracks) > 0:
        for track in tracks:
            x1, y1, x2, y2 = track[:4].astype(int)
            track_id = int(track[4])

            #точка контакта с полом (низ бокса)
            bottom_center = ((x1 + x2) // 2, y2)

            #добавлениев историю
            if track_id not in track_history:
                track_history[track_id] = []

            track_line = track_history[track_id]
            track_line.append(bottom_center)
            if len(track_line) > 30:
                track_line.pop(0)

            #трансформация  в вид сверху
            x_top, y_top = transformer.transform_point(bottom_center[0], bottom_center[1])
            current_points_topdown.append((x_top, y_top))
            #для змейки
            if track_id not in heatmap_gen.track_trails:
                heatmap_gen.track_trails[track_id] = []

            trail = heatmap_gen.track_trails[track_id]
            trail.append((x_top, y_top))
            if len(trail) > 30:
                trail.pop(0)

    # обновление heatmap
    heatmap_gen.add_points(current_points_topdown)

    # возврат комбинированного вид
    return heatmap_gen.get_split_view(point_color=(255, 0, 255), point_size=5, trail_thickness=2)

#легенда
def draw_heatmap_legend(frame, position=(10, 30)):

    x, y = position

    #градиент для легенды
    legend_width = 200
    legend_height = 20
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

    for i in range(legend_width):
        ratio = i / legend_width
        #от синего к красному
        if ratio < 0.25:
            color = (int(255 * ratio * 4), 0, 255)
        elif ratio < 0.5:
            color = (255, 0, int(255 * (1 - (ratio - 0.25) * 4)))
        elif ratio < 0.75:
            color = (255, int(255 * (ratio - 0.5) * 4), 0)
        else:
            color = (int(255 * (1 - (ratio - 0.75) * 4)), 255, 0)
        cv2.rectangle(legend, (i, 0), (i + 1, legend_height), color, -1)

    h, w = frame.shape[:2]
    if y + legend_height + 40 < h and x + legend_width < w:
        frame[y:y + legend_height, x:x + legend_width] = legend

        # Подписи
        cv2.putText(frame, "Low", (x, y + legend_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "High", (x + legend_width - 50, y + legend_height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

