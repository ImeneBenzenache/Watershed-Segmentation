from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

import cv2

# loading the image and applying pyramid Mean Shift Filtering
image = cv2.imread("Images/coins_01.png")
shifted = cv2.pyrMeanShiftFiltering(image, 21, 80)
# cv2.imshow("Input", image)
# cv2.imshow("Shifted", shifted)


# Converting the shifted image to grayscale then appling the Otsu thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
# cv2.imshow("gray", gray)

# Competing the Euclidean distance transform on the binary image
D = ndimage.distance_transform_edt(thresh)

# Calculating local maximas
localMax = peak_local_max(D, indices=False, min_distance=20)

# performing a connected component analysis on the local peaks,
markers = ndimage.label(localMax)[0]

# for m in markers:
#     for n in m:
#         if n != 0:
#             print(n)


# Applying the watershed algorithm on the markers
labels = watershed(-D, markers, mask=thresh)

# Giving a color to the non black pixels
for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        if labels[i, j] != 0:
            image[i, j] = [labels[i, j]*5, 0, labels[i, j]*10]

cv2.imshow("Segmented Image", image)
cv2.waitKey(0)
