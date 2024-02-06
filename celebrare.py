import cv2

def remove_background(image_path):
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return image
    
    mask = np.zeros_like(image[:, :, 0])
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)
    
    mask = cv2.bitwise_not(mask)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

if __name__ == "__main__":
    image_path = "C:\Users\91798\Downloads\celibrare.png"
    result = remove_background(image_path)
    
    cv2.imshow("Original", cv2.imread(image_path))
    cv2.imshow("Background Removed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
