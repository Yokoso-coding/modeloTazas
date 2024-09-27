from ultralytics import YOLO
import os
import cv2

def train_model(dataset_path):
    # Load the YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model on our mug dataset
    results = model.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs=200,
        imgsz=640,
        batch=16,
        name='mug_detector'
    )

    # Print the results
    print(results)

    # Validate the model
    val_results = model.val()
    print(val_results)

    # Test on a single image
    test_images_path = os.path.join(dataset_path, "test", "images")
    test_images = os.listdir(test_images_path)
    if test_images:
        test_img_path = os.path.join(test_images_path, test_images[0])
        results = model(test_img_path)
        
        # Load the image
        img = cv2.imread(test_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot the results on the image
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(img, f'{r.names[int(box.cls)]} {float(box.conf):.2f}', 
                            (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("Test Image with Detections", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the image with detections
        output_path = os.path.join(dataset_path, "test_result.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Test result saved to: {output_path}")
    else:
        print("No test images found.")

    # Export the model
    model.export(format="onnx", dynamic=True, simplify=True)
    print(f"Model exported to: {os.path.join(model.export(), 'weights', 'best.onnx')}")

if __name__ == "__main__":
    # Read the dataset path from the file
    with open("dataset_path.txt", "r") as f:
        dataset_path = f.read().strip()
    
    train_model(dataset_path)