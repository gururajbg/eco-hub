import cv2
import os
#test

def show_image(image_path):
    try:
        # Read the image
        print(f"Reading image from {image_path}...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Display the image
        print("Displaying image...")
        cv2.imshow('Image Viewer', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

def show_webcam():
    try:
        # Open webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit")
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display the frame
            cv2.imshow('Webcam Viewer', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Show image")
    print("2. Show webcam")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        image_path = input("Enter the path to your image: ")
        show_image(image_path)
    elif choice == "2":
        show_webcam()
    else:
        print("Invalid choice") 