from ultralytics import YOLO
import cv2


def fruit_detection(filename):
    model = YOLO("weights/peachlemonv3.pt")
    labels = ["freshpeach", "freshlemon", "rottenpeach", "rottenlemon"]

    results = model(filename, save_json=True)

    img_label_results = []
    img = cv2.imread(filename)
    img_2 = img.copy()
    for result in results:
        for i, cls in enumerate(result.boxes.cls):
            crop_img = img[int(result.boxes.xyxy[i][1]):int(result.boxes.xyxy[i][3]),
                       int(result.boxes.xyxy[i][0]):int(result.boxes.xyxy[i][2])]
            cv2.rectangle(img_2, (int(result.boxes.xyxy[i][0]), int(result.boxes.xyxy[i][1])),
                          (int(result.boxes.xyxy[i][2]), int(result.boxes.xyxy[i][3])), (0, 255, 0), 2)
            cv2.putText(img_2, labels[int(cls)] + str(i), (int(result.boxes.xyxy[i][0]), int(result.boxes.xyxy[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            img_label_results.append({"label": labels[int(cls)] + str(i), "crop_img": crop_img})
    return img_2, img_label_results
