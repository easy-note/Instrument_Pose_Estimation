import numpy as np


def eval_gaussian(mu, sigma=20., h=480, w=640):

    mu_np = np.array(mu, dtype=np.float32).reshape(1, 2)
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)

    grid = np.array(np.meshgrid(y, x)).reshape(2, -1)
    diff = grid - mu_np.T
    norm_squared = np.sum(diff ** 2, axis=0)
    tmp = norm_squared / (sigma ** 2)
    proba = np.exp(-tmp).reshape(w, h)
    return proba.T

def eval_gt(mu, h=480, w=640):

    mu_np = np.array(mu, dtype=np.float32).reshape(1, 2) # [[335.62906 355.21484]]
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)

    grid = np.array(np.meshgrid(y, x)).reshape(2, -1)
    
    diff = grid - mu_np.T
    
    norm_squared = np.sum(diff ** 2, axis=0)
    # tmp = norm_squared / (sigma ** 2)
    # proba = np.exp(-tmp).reshape(w, h)
    # return proba.T


def __gaussian(norm_squared, sigma=20.):
    return np.exp(-norm_squared / (sigma ** 2))


def __circular_mask(grid, center, radius):

    dist_from_center = np.sqrt((grid[0, :] - center[0]) ** 2 + (grid[1, :] - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def eval_line(pt1, pt2, m, b, h=576, w=720, sigma=20):
    # (x, y)
    x = np.linspace(0, w - 1, w) # 구간 시작점, 구간 끝점, 구간 내 숫자 개수
    y = np.linspace(0, h - 1, h)
    grid = np.array(np.meshgrid(y, x)).reshape(2, -1) # (2, 414720)

    distance = np.abs(m * grid[1, :] + -1 * grid[0, :] + b) / np.sqrt(m ** 2 + 1) # (414720)
    
    heatmap = __gaussian(distance ** 2, sigma=sigma).reshape(w, h)
    
    # center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
    center = ((pt1[1] + pt2[1]) / 2, (pt1[0] + pt2[0]) / 2) # center int()
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    mask = __circular_mask(grid, center, dist//2 + sigma//2)
    
    heatmap *= mask.reshape(w, h)

    # from visualization import np_visual
    # img_path = '/raid/datasets/public/EndoVisPose/extract/labelled_train/Dataset1/img_000060_raw_train1.jpg'    
    # save_dir = '/SSL-2D-Pose/utils/save'
    # save_f = 'img_000060-heatmap.jpg'

    # target = heatmap.reshape(w,h).T
    # print(target)
    # print(target.shape)
    # np_visual(target, img_path, save_dir, save_f)

    # exit(0)
    

    return heatmap.T


if __name__ == "__main__":
    pass
