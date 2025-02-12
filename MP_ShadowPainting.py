"""***************************************************
SILHOUETTE GAME 
A small Application for Microsoft Kinect Azure v2 That compares a shadow with a reference Image and give an ok if the shadow Matches the Image.
Code: Anselm Fritzsche
Idea: Lion Isele and Anselm Fritzsche
**************************************************"""




import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
from skimage.metrics import structural_similarity as ssim

def preprocess_silhouette(image, display_steps=False, window_name="Processing"):
    """
    Verarbeitet das Bild, um die Silhouette stabiler zu extrahieren und zeigt optional
    Zwischenschritte der Verarbeitung in einem separaten Fenster.
    """
    if image.shape[-1] == 4:  # Falls ein Alphakanal existiert
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Graustufenbild
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogramm-Equalisierung, um den Kontrast zu verbessern
    gray = cv2.equalizeHist(gray)

    # Adaptive Schwellenwertbestimmung für stabilere Binärdarstellung
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

    # Morphologische Transformationen zur Verbesserung der Silhouette
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    if display_steps:
        cv2.imshow(f"{window_name} - Binärbild", binary)

    # Konturen finden
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Keine Konturen im {window_name} gefunden.")
        return None, None

    # Größte Kontur wählen
    largest_contour = max(contours, key=cv2.contourArea)

    # Rechteck um die größte Kontur
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Region ausschneiden und auf feste Größe skalieren
    cropped = binary[y:y + h, x:x + w]
    resized = cv2.resize(cropped, (200, 200))

    # Zeichne die Kontur zur Visualisierung
    contour_image = np.zeros_like(binary)
    cv2.drawContours(contour_image, [largest_contour], -1, 255, thickness=2)

    return resized, contour_image

def accumulate_frames(frames, num_frames=10):
    """
    Führt Mittelwertbildung auf mehreren Frames durch, um Rauschen zu reduzieren.
    """
    frame_accumulated = np.zeros_like(frames[0], dtype=np.float32)
    for frame in frames:
        frame_accumulated += frame / num_frames

    return cv2.convertScaleAbs(frame_accumulated)

def compare_silhouettes(camera_image, path, roi_camera, display_steps=False):
    """
    Vergleicht die binarisierten Silhouetten aus dem Kamerabild (innerhalb einer ROI) und dem Vergleichsbild.
    Zeigt Verarbeitungsschritte in separaten Fenstern.
    """
    # Lade das Vergleichsbild
    jpg_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if jpg_image is None:
        print("Fehler: Vergleichsbild konnte nicht geladen werden.")
        return None

    # Extrahiere den Bereich (ROI) aus dem Kamerabild
    x, y, w, h = roi_camera
    cropped_camera_image = camera_image[y:y + h, x:x + w]

    # Verarbeite Kamera- und Vergleichsbilder zu binarisierten Silhouetten
    camera_binary, _ = preprocess_silhouette(cropped_camera_image, display_steps, "Kamera")
    jpg_binary, _ = preprocess_silhouette(jpg_image, display_steps, "Vergleichsbild")

    # Sicherstellen, dass beide Bilder gültig sind
    if camera_binary is None or jpg_binary is None:
        print("Fehler: Eine der Silhouetten konnte nicht verarbeitet werden.")
        return None

    # Größe der Bilder anpassen, um den Vergleich zu ermöglichen
    h, w = jpg_binary.shape
    camera_resized = cv2.resize(camera_binary, (w, h), interpolation=cv2.INTER_AREA)

    # SSIM-Vergleich der binarisierten Silhouetten
    similarity, _ = ssim(camera_resized, jpg_binary, full=True)

    return similarity

def start_kinect():
    print("Kinect gestartet!")
    kinect = PyK4A(Config(color_resolution=ColorResolution.RES_720P, 
                          depth_mode=DepthMode.NFOV_2X2BINNED, 
                          camera_fps=2))
    kinect.start()

    roi_camera = [320, 180, 640, 360]  # Startwerte für x, y, Breite, Höhe des Kamerabildes
    scale_step = 20

    try:
        while True:
            try:
                capture = kinect.get_capture()
                if capture.color is None:
                    continue
                color_image = capture.color

                # Validierung der ROI
                x, y, w, h = roi_camera
                if x < 0 or y < 0 or x + w > color_image.shape[1] or y + h > color_image.shape[0]:
                    print(f"Ungültige ROI: {roi_camera}. Überspringe Frame.")
                    continue

                path_to_jpg = r"C:\Users\Ansn\AppData\Local\Programs\Python\Python39\Lib\site-packages\pyk4a\MD2\Dreieck.jpg"
                try:
                    # ROI im Kamerabild anzeigen
                    camera_image_with_roi = color_image.copy()
                    cv2.rectangle(camera_image_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ROI Rechteck in grün
                    cv2.imshow("Kamera mit ROI", camera_image_with_roi)

                    # Silhouetten vergleichen
                    similarity = compare_silhouettes(color_image, path_to_jpg, roi_camera, display_steps=True)
                    if similarity is not None:
                        print(f"Formvergleichsdifferenz (SSIM): {similarity:.2f}")
                        
                        # Originalbilder anzeigen
                        cv2.imshow("Original Kamera", cv2.resize(color_image[:, :, :3], (640, 360)))
                        cv2.imshow("Original Vergleichsbild", cv2.resize(cv2.imread(path_to_jpg), (640, 360)))

                except Exception as e:
                    print(f"Fehler beim Vergleichen der Silhouetten: {e}")

                # ROI-Anpassung
                key = cv2.waitKey(1) & 0xFF
                if key == ord('+'):
                    roi_camera[0] = max(0, roi_camera[0] - scale_step)
                    roi_camera[1] = max(0, roi_camera[1] - scale_step)
                    roi_camera[2] = min(1280 - roi_camera[0], roi_camera[2] + 2 * scale_step)
                    roi_camera[3] = min(720 - roi_camera[1], roi_camera[3] + 2 * scale_step)
                    print(f"ROI vergrößert: {roi_camera}")
                elif key == ord('-'):
                    roi_camera[0] += scale_step
                    roi_camera[1] += scale_step
                    roi_camera[2] = max(scale_step, roi_camera[2] - 2 * scale_step)
                    roi_camera[3] = max(scale_step, roi_camera[3] - 2 * scale_step)
                    print(f"ROI verkleinert: {roi_camera}")
                elif key == ord('q'):
                    break
            except Exception as e:
                print(f"Allgemeiner Fehler im Kinect-Programm: {e}")
                break
    except Exception as e:
        print(f"Allgemeiner Fehler im Kinect-Programm: {e}")
    finally:
        print("Kinect gestoppt!")
        kinect.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_kinect()
