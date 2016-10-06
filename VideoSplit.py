import cv2


def videotoframes(inputvideo, outputdirectory, imageformat, count):
    vidcap = cv2.VideoCapture(inputvideo)
    success, image = vidcap.read()
    while success:
        cv2.imwrite((outputdirectory + "/%d." + imageformat) % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    return count - 1
