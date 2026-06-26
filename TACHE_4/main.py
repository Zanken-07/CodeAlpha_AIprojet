import cv2
from ultralytics import YOLO

# 1. Charger le modèle d'IA 
# (Nous utilisons YOLOv8n : 'n' pour nano. C'est le plus léger et le plus rapide pour la vidéo en temps réel)
print("Chargement du modèle d'IA en cours...")
model = YOLO('yolov8n.pt')

# Initialiser la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

print("Modèle chargé ! Appuyez sur 'q' pour fermer la fenêtre.")

while True:
    # Capturer l'image
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur de flux vidéo.")
        break
        
    # 2. Donner l'image à l'IA pour la détection
    # YOLO analyse l'image et trouve les objets
    results = model.track(frame, persist=True)
    
    # 3. Dessiner les résultats
    # La fonction plot() d'ultralytics dessine automatiquement les boîtes de délimitation et les étiquettes
    annotated_frame = results[0].plot()
    
    # 4. Afficher l'image annotée
    cv2.imshow('Detection et Suivi YOLOv8', annotated_frame)
    
    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()