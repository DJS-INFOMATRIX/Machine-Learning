import cv2
import cvzone
import urllib.request
from cvzone.ColorModule import ColorFinder
import numpy as np

# IP Webcam URL
url = "http://192.168.29.216:8080/shot.jpg"        

# Color finer class
myColorFinder = ColorFinder(trackBar=False)
# Specifying the HSV Values
hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 167, 'hmax': 179, 'smax': 255, 'vmax': 255}

def empty(a):
    pass

# Changing the threshold values
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold 1", "Settings", 50, 255, empty)
cv2.createTrackbar("Threshold 2", "Settings", 100, 255, empty)
cv2.createTrackbar("Threshold 3", "Settings", 100, 255, empty)

# Preprocessing function
def preProcessing(img):
    # S 1: Gaussian Blur
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    
    # S 2: Canny for feature (edge) extraction
    thr1 = cv2.getTrackbarPos("Threshold 1", "Settings")
    thr2 = cv2.getTrackbarPos("Threshold 2", "Settings")
    imgPre = cv2.Canny(imgPre, thr1, thr2)

    # S 3: Make edges thicker
    kernel = np.ones((2, 2), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)

    # S 4: Close gaps in the edges
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre

# Main loop
while True:
    total_coins = 0
    total_money = 0
    imgColor = None
    
    # Tyr except for Phone's camera
    try:
        img_resp = urllib.request.urlopen(url)  # Request image from the IP Webcam URL
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)  # Convert to NumPy array
        img = cv2.imdecode(img_np, -1)  # Decode image to OpenCV format
    except Exception as e:
        print("Could not retrieve image from URL:", e)
        continue  # Skip to the next iteration if image retrieval fails

    # Preprocess the image
    imgPre = preProcessing(img)
    
    thr3 = cv2.getTrackbarPos("Threshold 3", "Settings")
    
    # Find contours on the preprocessed image
    contours, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the original image for visualization
    imgContours = img.copy()
    
    # Main stuff here
    for count, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > thr3:                                 # Only consider contours above the threshold area
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the edges in the shape of the counter is > 6, then consider it a circle
            if len(approx) > 6:
                total_coins += 1
                
                cv2.drawContours(imgContours, [contour], -1, (0, 255, 0), 3)    # Draw contour on the image
                
                # For finding the value of the coin
                # For croped image
                x, y, w, h = cv2.boundingRect(contour)  # Get bounding box coordinates
                imgCrop = img[y:y+h, x:x+w]             # Crop the detected coin area
                # cv2.imshow(f"Coin {count}", imgCrop)  # Show EACH cropped coin
                
                imgColor, mask = myColorFinder.update(imgCrop, hsvVals)         # Apply color filter
                
                pixelCount = cv2.countNonZero(mask)
                # print(pixelCount)
                
                if pixelCount < 37000:
                    total_money += 5
                else:
                    total_money += 2
                    
    # Dispaly the FINAL ANSWER
    print(f"Number of coins: {total_coins}, Total Money: {total_money}")
    
    # Resize each image
    img_resized = cv2.resize(img, (580, 300))
    imgPre_resized = cv2.resize(imgPre, (580, 300))
    imgContours_resized = cv2.resize(imgContours, (580, 300))
    
    # Stacking the windows
    imgStacked = cvzone.stackImages([img_resized, imgPre_resized, imgContours_resized], 2, 1)
    cv2.imshow("Processed Images", imgStacked)  # Show the stacked images in a single window
    
    # For seeing the imgColor (masked) image
    # if imgColor is not None:
    #     cv2.imshow("imgColor", imgColor)  # Show imgColor if available
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
