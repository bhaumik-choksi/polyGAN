import urllib.request as req
import cv2

urls = open('imageurls.txt')
urls = list(url.strip() for url in urls)
count = 0
for url in urls:
    try:
        req.urlretrieve(url, "images/" + str(count) + ".jpg")
        img = cv2.imread("images/" + str(count) + ".jpg", cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (36, 36)) # Change resolution here
        cv2.imwrite("images/" + str(count) + ".jpg", resized)
        count += 1
        print(count, "done")

    except Exception as e:
        print(str(e))
