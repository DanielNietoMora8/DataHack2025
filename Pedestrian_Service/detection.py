import cv2
import numpy as np
import torch
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from scipy.spatial import distance

regions = []
current_region = []

def click_event(event, x, y, flags, param):
    global current_region, regions
    if event == cv2.EVENT_LBUTTONDOWN:
        current_region.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_region) >= 3:
            regions.append(current_region.copy())
            print(f"Región {len(regions)} creada con {len(current_region)} puntos.")
        current_region = []

def draw_regions(img, regions):
    for i, region in enumerate(regions):
        pts = np.array(region, np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)


def draw_region_counts(frame, counted_centroids_per_region):
    start_x = 10
    start_y = 30
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    bg_color = (0, 0, 0)       # Negro para fondo del texto
    text_color = (0, 255, 255) # Amarillo para texto

    # Calculamos ancho máximo para el fondo del recuadro
    max_width = 0
    labels = []
    for idx, counted in enumerate(counted_centroids_per_region):
        label = f"Región {idx + 1}: {len(counted)}"
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        max_width = max(max_width, w)
        labels.append(label)

    padding = 10
    box_width = max_width + padding * 2
    box_height = line_height * len(labels) + padding

    # Dibuja fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x - padding, start_y - line_height),
                  (start_x - padding + box_width, start_y - line_height + box_height),
                  bg_color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Dibuja cada línea de texto encima del fondo
    for i, label in enumerate(labels):
        y = start_y + i * line_height
        cv2.putText(frame, label, (start_x, y), font, font_scale, text_color, thickness)


def main(video_source=0):
    global current_region

    model = YOLO("yolov8n.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(video_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    output_path = "salida_detectada.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        print("Error al abrir video o cámara.")
        return

    clone = frame.copy()
    cv2.namedWindow("Selecciona regiones")
    cv2.setMouseCallback("Selecciona regiones", click_event)

    print("🖱️ Click izquierdo para puntos. Click derecho para cerrar región. Pulsa Q para continuar.")

    while True:
        temp = clone.copy()
        for i, pt in enumerate(current_region):
            cv2.circle(temp, pt, 4, (255, 0, 0), -1)
            if i > 0:
                cv2.line(temp, current_region[i - 1], pt, (255, 0, 0), 2)
        if len(current_region) >= 3:
            cv2.polylines(temp, [np.array(current_region)], False, (0, 255, 255), 1)
        draw_regions(temp, regions)
        cv2.imshow("Selecciona regiones", temp)
        key = cv2.waitKey(1)
        if key == ord('q'):
            if current_region:
                print("⚠️ Región sin cerrar descartada.")
                current_region.clear()
            break

    cv2.destroyWindow("Selecciona regiones")

    print("Iniciando detección y conteo de personas...")

    region_polygons = [Polygon(r) for r in regions]
    min_conf = 0.6  # baja confianza para detectar más personas lejanas

    tracked_centroids = []
    counted_centroids_per_region = [ [] for _ in regions ]
    max_distance = 20  # distancia máxima para considerar mismo objeto

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=min_conf)[0]

        current_centroids = []

        for box in results.boxes:
            if int(box.cls) == 0 and box.conf.item() > min_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                current_centroids.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        new_tracked = []
        for c in current_centroids:
            if tracked_centroids:
                dists = distance.cdist([c], tracked_centroids)
                min_dist = dists.min()
                if min_dist < max_distance:
                    new_tracked.append(tracked_centroids[dists.argmin()])
                else:
                    new_tracked.append(c)
            else:
                new_tracked.append(c)
        tracked_centroids = new_tracked

        # Conteo por región sin IDs
        for idx, polygon in enumerate(region_polygons):
            for c in tracked_centroids:
                point = Point(c)
                if polygon.contains(point):
                    counted_before = False
                    for pc in counted_centroids_per_region[idx]:
                        if distance.euclidean(c, pc) < max_distance:
                            counted_before = True
                            break
                    if not counted_before:
                        counted_centroids_per_region[idx].append(c)
                        print(f"🧍 Persona contada en región {idx + 1}")

        # Dibujar regiones y texto de conteo
        draw_regions(frame, regions)
        draw_region_counts(frame, counted_centroids_per_region)

        out.write(frame)
        cv2.imshow("Detección y Conteo", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video guardado en: {output_path}")

if __name__ == "__main__":
    import sys
    video_path = "files/sample3.mp4"
    main(video_path)