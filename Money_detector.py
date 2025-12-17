from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model
model = YOLO("best.pt")

# Load image
image_path = "m3a9ed.jpg"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Denomination mapping (CHANGE if needed)

total_money = 0

for result in results:
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls)

        class_name = model.names[cls_id]

    

        # Convert string → number
        value = int(class_name.replace("k", "000")) if "k" in class_name else int(class_name)

        total_money += value

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Label value
        cv2.putText(
            image,
            str(value),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


# Save and display result
cv2.imwrite("money_detected_sum.png", image)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

print("Total detected money:", total_money)
