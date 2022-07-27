import numpy as np
from scipy.ndimage.filters import maximum_filter


def nms(heatmap, num_parts):

    NMS_THRESH = [.2]*num_parts #[.2, .1, .1, .1, .1] # hyper-parameter 
    # 각 joint 마다 confidence 의 threshold 

    for i in range(num_parts):
        NMS_THRESH[i] = max(np.average(heatmap[:, :, i]) * 4., NMS_THRESH[i])
        NMS_THRESH[i] = min(NMS_THRESH[i], .3)

    window_size = 20
    peaks = [[], [], [], [], []]  # use only for data augmentation

    filtered = maximum_filter(heatmap, footprint=np.ones((window_size, window_size, 1))) # maximum_filter : window 영역에서 maximum valule 를 뽑아 i,j의 값은 maximum value로 대체
    suppressed = heatmap * np.equal(heatmap, filtered) # Filter 후의 값으로 heatmap 대체
    suppressed = suppressed >= NMS_THRESH # 너무 작은 confidence는 제거

    # peak 좌표 (y, x) 를 뽑는 코드
    for ch in range(heatmap.shape[-1]):
        p = np.where(suppressed[:, :, ch] != 0)
        peaks[ch] += list(zip(p[1], p[0]))

    return peaks, suppressed

