from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_harris_response(im, k=0.15, sigma=1.2):
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    return Wdet - k * (Wtr ** 2)
    # return Wxx/Wyy

def get_harris_points(harrisim, threshold=0.01):
    # find top corner candidates above a threshold
    harrisim_t = (harrisim > harrisim.max() * threshold)
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    # sort candidates
    index = np.argsort(candidate_values)
    # save the points
    filtered_coords = []
    for i in index:
        filtered_coords.append(coords[i])
    return filtered_coords

def ResizeImage(im, new_width):
    param = im.size
    wpercent = (new_width / float(param[0]))
    new_height = int((float(param[1]) * float(wpercent)))
    resize_img = im.resize((new_width, new_height))
    return resize_img

def Assignment4Part1(img):
    resized_img = ResizeImage(img, 200)
    rotated_img = im.rotate(30)

    img = np.array(img)
    harrisim_orig = compute_harris_response(img)
    filtered_coords_orig = get_harris_points(harrisim_orig)

    resized_img = np.array(resized_img)
    harrisim_resized = compute_harris_response(resized_img)
    filtered_coords_resized = get_harris_points(harrisim_resized)

    rotated_img = np.array(rotated_img)
    harrisim_rotated = compute_harris_response(rotated_img)
    filtered_coords_rotated = get_harris_points(harrisim_rotated)

    plt.figure()
    plt.gray()

    ax = []
    ax.append(plt.subplot2grid((3, 2), (0, 0)))
    ax.append(plt.subplot2grid((3, 2), (0, 1)))
    ax.append(plt.subplot2grid((3, 2), (1, 0)))
    ax.append(plt.subplot2grid((3, 2), (1, 1)))
    ax.append(plt.subplot2grid((3, 2), (2, 0)))
    ax.append(plt.subplot2grid((3, 2), (2, 1)))

    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(img)
    ax[1].plot([p[1] for p in filtered_coords_orig], [p[0] for p in filtered_coords_orig], 'r*')
    ax[1].axis('off')

    ax[2].imshow(resized_img)
    ax[2].axis('off')
    ax[3].imshow(resized_img)
    ax[3].plot([p[1] for p in filtered_coords_resized], [p[0] for p in filtered_coords_resized], 'r*')
    ax[3].axis('off')

    ax[4].imshow(rotated_img)
    ax[4].axis('off')
    ax[5].imshow(rotated_img)
    ax[5].plot([p[1] for p in filtered_coords_rotated], [p[0] for p in filtered_coords_rotated], 'r*')
    ax[5].axis('off')
    plt.show()

def Assignment4Part2(im):
    im = np.array(im)
    harrisim = compute_harris_response(im)
    boundary = 28
    edges = []
    corners = []
    flat = []
    for i in range(len(harrisim)):
        for j in range(len(harrisim[i])):
            if (abs(harrisim[i][j]) < boundary):
                flat.append(harrisim[i][j])
            elif (harrisim[i][j] < 0):
                edges.append(harrisim[i][j])
            else:
                corners.append(harrisim[i][j])

    plt.figure()
    plt.gray()
    ax = []
    ax.append(plt.subplot2grid((3, 2), (0, 0), rowspan=3))
    ax.append(plt.subplot2grid((3, 2), (0, 1)))
    ax.append(plt.subplot2grid((3, 2), (1, 1)))
    ax.append(plt.subplot2grid((3, 2), (2, 1)))
    ax[0].imshow(im)
    ax[0].axis('off')
    ax[1].plot(corners, 'b-')
    ax[2].plot(edges, 'b-')
    ax[3].plot(flat, 'b-')
    plt.show()


if __name__ == '__main__':
    im = Image.open('./miet.jpeg').convert('L')
    Assignment4Part1(im)
    Assignment4Part2(im)