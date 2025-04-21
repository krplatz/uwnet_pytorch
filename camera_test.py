import cv2

def continuously_update_window():
    # Open a connection to the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame (if needed)
        # For example, convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show the original frame in one window
        cv2.imshow('Original Frame', frame)

        # Show the processed frame in another window
        cv2.imshow('Grayscale Frame', gray_frame)

        # Wait for a short period to allow the window to refresh
        # This will also allow you to break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    continuously_update_window()