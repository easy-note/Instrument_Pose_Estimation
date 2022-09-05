import numpy as np
from scipy.ndimage.filters import maximum_filter


def nms(heatmap, num_parts, window_size):

    NMS_THRESH = [0] * 5 #[.05, .05, .05, .05, .05] # hyper-parameter 
    # 각 joint 마다 confidence 의 threshold 

    for i in range(num_parts): #5
        NMS_THRESH[i] = max(np.average(heatmap[:, :, i]) * 4., NMS_THRESH[i])
        # NMS_THRESH[i] = max(np.average(heatmap[:, :, i]), NMS_THRESH[i])
        NMS_THRESH[i] = min(NMS_THRESH[i], .3)
    
    # NMS_THRESH = [0.02236322872340679, 0.023008402436971664, 0.023070527240633965, 0.025952735915780067, 0.02698430046439171]

    peaks = [[], [], [], [], []]  # use only for data augmentation
    
    # maximum_filter : window 영역에서 maximum valule 를 뽑아 i,j의 값은 maximum value로 대체
    filtered = maximum_filter(heatmap, footprint=np.ones((window_size, window_size, 1))) 

    suppressed = heatmap * np.equal(heatmap, filtered) # Filter 후의 값으로 heatmap 대체 -> max 값 주변을 다 0으로 치환
    suppressed = suppressed >= NMS_THRESH # 너무 작은 confidence는 제거

    # peak 좌표 (y, x) 를 뽑는 코드
    for ch in range(heatmap.shape[-1]): # 5
        p = np.where(suppressed[:, :, ch] != 0)        
        peaks[ch] += list(zip(p[1], p[0]))
    '''
    peaks 
    [[(140, 61), (172, 124), (269, 210)], [(141, 60), (172, 124), (269, 210)], [(175, 54), (135, 139)], [(208, 61), (92, 156)], [(10, 13), (308, 69), (13, 190)]]
    '''
    
    return peaks, suppressed

