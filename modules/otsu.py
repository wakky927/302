import numpy as np
import cv2


def multi(img, c=3, l=None):
    """The threshold values are chosen to maximize the total sum of pairwise
    variances between the thresholded gray-level classes. See Notes and [1]_
    for more details.
    The input image must be grayscale, and the input classes must be 2, 3, 4,
    or 5.
    :param
        img: (N, M) ndarray
            Grayscale input image.
        c: int, optional
            Number of classes to be thresholded, i.e. the number of resulting
            regions.
        l: list, optional
            gray levels array
    :return
        th: (classes - 1) ndarray
            Array containing the threshold values for the desired classes.
        th_im: (N, M) ndarray
            Thresholded output image.
    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    """
    th = [0] * (c - 1)            # thresholds array
    th_img = img.copy()           # thresholded img
    L = 255                       # max gray-level
    sigma_max = 0                 # max(sigma)
    fi = np.zeros(L + 1)          # frequency: sum[i=0, 255] fi[i] = N (= width * height)
    pi = np.zeros(L + 1)          # probability: sum[i=0, 255] pi[i] = 1

    p = np.zeros((L + 1, L + 1))  # u-v interval zeroth-order moment
    s = np.zeros((L + 1, L + 1))  # u-v interval first-order moment
    h = np.zeros((L + 1, L + 1))  # modified between-class variance

    for i in range(L + 1):
        fi[i] = np.count_nonzero(img == i)
        pi[i] = fi[i] / img.size

    # calculation p, s, h
    # p(u, v) = p(1, v) - p(1, u-1)
    # s(u, v) = s(1, v) - s(1, u-1)
    # h(u, v) = s(u, v)^2 / p(u, v)

    p[0][0] = pi[0]
    s[0][0] = 1 * pi[0]

    for v in range(L):
        p[0][v+1] = p[0][v] + pi[v]
        s[0][v+1] = s[0][v] + (v + 1) * pi[v]

    for u in range(L):
        for v in range(L):
            if u > v:
                p[u][v] = 0.0
                s[u][v] = 0.0
                h[u][v] = 0.0

            else:
                p[u+1][v] = p[0][v] - p[0][u]
                s[u+1][v] = s[0][v] - s[0][u]

                if p[u][v] == 0:
                    h[u][v] = 0.0

                else:
                    h[u][v] = s[u][v]**2 / p[u][v]

    # calculation sigma, searching arg max: thresholds, and re-formatting image
    if l is None:
        if c == 3:
            l = [0, 128, 255]

        elif c == 4:
            l = [0, 85, 170, 255]

        elif c == 5:
            l = [0, 64, 128, 192, 255]

    if c == 2:
        for t0 in range(L - c):
            sigma = h[0][t0] + h[t0 + 1][L - 1]

            if sigma_max < sigma:
                sigma_max = sigma
                th[0] = t0

        th_img[img < th[0]] = l[0]
        th_img[th[0] <= img] = l[1]

    elif c == 3:
        for t0 in range(L - c):
            for t1 in range(t0 + 1, L - c + 1):
                sigma = h[0][t0] + h[t0 + 1][t1] + h[t1 + 1][L - 1]

                if sigma_max < sigma:
                    sigma_max = sigma
                    th[0] = t0
                    th[1] = t1

        th_img[img < th[0]] = l[0]
        th_img[(th[0] <= img) & (img < th[1])] = l[1]
        th_img[th[1] <= img] = l[2]

    elif c == 4:
        for t0 in range(L - c):
            for t1 in range(t0 + 1, L - c + 1):
                for t2 in range(t1 + 1, L - c + 2):
                    sigma = h[0][t0] + h[t0 + 1][t1] + h[t1 + 1][t2] + h[t2 + 1][L - 1]

                    if sigma_max < sigma:
                        sigma_max = sigma
                        th[0] = t0
                        th[1] = t1
                        th[2] = t2

        th_img[img < th[0]] = l[0]
        th_img[(th[0] <= img) & (img < th[1])] = l[1]
        th_img[(th[1] <= img) & (img < th[2])] = l[2]
        th_img[th[2] <= img] = l[3]

    elif c == 5:
        for t0 in range(L - c):
            for t1 in range(t0 + 1, L - c + 1):
                for t2 in range(t1 + 1, L - c + 2):
                    for t3 in range(t2 + 1, L - c + 3):
                        sigma = h[0][t0] + h[t0 + 1][t1] + h[t1 + 1][t2] \
                                + h[t2 + 1][t3] + h[t3 + 1][L - 1]

                        if sigma_max < sigma:
                            sigma_max = sigma
                            th[0] = t0
                            th[1] = t1
                            th[2] = t2
                            th[3] = t3

        th_img[img < th[0]] = l[0]
        th_img[(th[0] <= img) & (img < th[1])] = l[1]
        th_img[(th[1] <= img) & (img < th[2])] = l[2]
        th_img[(th[2] <= img) & (img < th[3])] = l[3]
        th_img[th[3] <= img] = l[4]

    else:
        return None

    return th, th_img


def emphasize(img, c=1, th=None, l=None):
    if th is None:
        th = [0, 128]
    em_img = img.copy()

    if l is None:
        if c == 1:
            l = [0, 255]

        elif c == 2:
            l = [0, 128, 255]

        elif c == 3:
            l = [0, 85, 170, 255]

        elif c == 4:
            l = [0, 64, 128, 192, 255]

    if c == 1:
        em_img[img < th[0]] = l[0]
        em_img[(th[0] <= img) & (img < th[1])] = l[1]
        em_img[th[1] <= img] = l[0]

    elif c == 2:
        em_img[img < th[0]] = l[0]
        em_img[(th[0] <= img) & (img < th[1])] = l[1]
        em_img[(th[1] <= img) & (img < th[2])] = l[2]
        em_img[th[2] <= img] = l[0]

    elif c == 3:
        em_img[img < th[0]] = l[0]
        em_img[(th[0] <= img) & (img < th[1])] = l[1]
        em_img[(th[1] <= img) & (img < th[2])] = l[0]
        em_img[(th[2] <= img) & (img < th[3])] = l[0]
        em_img[th[3] <= img] = l[0]

    return em_img


if __name__ == '__main__':
    im_path = "../../data/in/_00000000.bmp"
    img = cv2.imread(im_path, 0)

    # th, img = multi(img=img, c=4, l=[0, 255, 0, 0])
    # th, img = multi(img=img, c=5)
    # img = emphasize(img=img, c=2, th=[20, 70])
    img = emphasize(img=img, c=3, th=[14, 41, 80, 139])
    cv2.imshow('thresholded img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("../../data/out/threshold.bmp", im)

    print(0)
