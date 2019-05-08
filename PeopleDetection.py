import cv2 as cv

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def draw_person(img, person):
  x, y, w, h = person
  cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

img = cv.imread("people.jpg")
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

found, w = hog.detectMultiScale(img, winStride=(8,8),scale=1.05)

found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_filtered.append(r)

for person in found_filtered:
  draw_person(img, person)

cv.imshow("people detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
