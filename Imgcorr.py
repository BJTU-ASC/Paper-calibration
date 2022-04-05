import numpy as np
import cv2


def CrossPoint(line1, line2):
    x0, y0, x1, y1 = line1[0]
    x2, y2, x3, y3 = line2[0]

    dx1 = x1 - x0
    dy1 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2

    D1 = x1 * y0 - x0 * y1
    D2 = x3 * y2 - x2 * y3

    y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - D1) / dy1

    return (int(x), int(y))


def SortPoint(points):
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]

    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]

    return sp


def imgcorr(src):
    rgbsrc = src.copy()
    graysrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurimg = cv2.GaussianBlur(src, (3, 3), 0)
    Cannyimg = cv2.Canny(blurimg, 10, 100)
    # 对于不同图片的处理阈值需调整（对于一张图片进行多个阈值的求解？）
    # cv2.imshow("image_01", Cannyimg)
    cv2.imwrite("canny_res.jpg", Cannyimg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # 膨胀处理
    # pos_img = cv2.dilate(Cannyimg.copy(), kernel, 10)
    pos_img = Cannyimg
    # cv2.imshow("image_1", dilated)
    cv2.imwrite("dilate_res.jpg", pos_img)
    # # 循环寻找最大边框（这个方法必须要先保证线段闭合）
    # contours, hier = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # maxArea = 0
    # index = 0
    # k = 0
    # for i in contours:
    #     currentArea = cv2.contourArea(np.array(contours)[i])
    #     if currentArea > maxArea:
    #         maxArea = currentArea
    #         index = i
    # img = cv2.drawContours(src, contours, index, (0, 255, 0), 0)
    # # 轮廓图
    # cv2.imshow("image_3", img)
    lines = cv2.HoughLinesP(pos_img, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=40)
    # 这里可以引用有效字符区域的判断来减少中间区域的识别 https://blog.csdn.net/yangzm/article/details/81105844
    for i in range(int(np.size(lines) / 4)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(rgbsrc, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # cv2.imshow("image_2", rgbsrc)
    cv2.imwrite("Hough_res.jpg", rgbsrc)

    points = np.zeros((4, 2), dtype="float32")
    points[0] = CrossPoint(lines[0], lines[2])
    points[1] = CrossPoint(lines[0], lines[3])
    points[2] = CrossPoint(lines[1], lines[2])
    points[3] = CrossPoint(lines[1], lines[3])

    sp = SortPoint(points)

    width = int(np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))

    dstrect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(np.array(sp), dstrect)
    warpedimg = cv2.warpPerspective(src, transform, (width, height))

    return warpedimg


if __name__ == '__main__':
    src = cv2.imread("较好样张6.jpg")
    dst = imgcorr(src)
    cv2.imshow("Image", dst)
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", dst)

