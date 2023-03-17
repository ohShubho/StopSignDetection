import numpy as np
import cv2
import time

def mean_square_error(img1, img2):
    assert img1.shape == img2.shape, "Images must be the same shape."
    error = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    error = error/float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return error

def compare_images(img1, img2):
    return 1/mean_square_error(img1, img2)

def pyramid(image, scale=1.5, min_size=30, max_size=1000):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, w))
        if image.shape[0] < min_size or image.shape[1] < min_size:
            break
        if image.shape[0] > max_size or image.shape[1] > max_size:
            continue
        yield image

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y+window_size[1], x:x+window_size[0]])

def main():
    target_image = cv2.imread(args.image)
    target_image = cv2.resize(target_image, (500, 500))
    prototype_image = cv2.imread(args.prototype)

    max_sim = -1
    max_box = (0, 0, 0, 0)

    t0 = time.time()

    for p in pyramid(prototype_image, min_size=50, max_size=target_image.shape[0]):
        for (x, y, window) in sliding_window(target_image, step_size=2, window_size=p.shape):
            if window.shape[0] != p.shape[0] or window.shape[1] != p.shape[1]:
                continue

            temp_sim = compare_images(p, window)
            if temp_sim > max_sim:
                max_sim = temp_sim
                max_box = (x, y, p.shape[1], p.shape[0])

    t1 = time.time()

    print("Execution time: " + str(t1 - t0))
    print(max_sim)
    print(max_box)

    buff = 10
    (x, y, w, h) = max_box
    cv2.rectangle(target_image, (x-buff//2, y-buff//2), (x+w+buff//2, y+h+buff//2), (0, 255, 0), 2)

    cv2.imshow('image', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to the target image")
    parser.add_argument("-p", "--prototype", required=True, help="Path to the prototype object")
    args = parser.parse_args()
    main()