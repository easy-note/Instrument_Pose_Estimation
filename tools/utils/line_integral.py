import numpy as np

def get_line(point1, point2):
    # (y, x)

    m = float(point1[0] - point2[0]) / float(point1[1] - point2[1] + 1e-9) # 기울기
    b = point2[0] - m * point2[1] # y 절편 
    return m, b


# x 좌표로 point들을 봅기 때문에 선이 세로로 있는 경우 point가 많이 뽑히지 않음! 
def get_points(point1, point2):
    np.seterr(divide='ignore', invalid='ignore') # Code에서 0이나 NaN으로 나누기를 하지 않는다고 설정
    # (y, x)
    m, b = get_line(point1, point2)
    number_of_samples_x = abs(point2[1] - point1[1]) + 1 # joint 사이의 간격이 얼마나 있는지 뽑음 (x 축 기준)
    number_of_samples_y = abs(point2[0] - point1[0]) + 1 # joint 사이의 간격이 얼마나 있는지 뽑음 (y 축 기준)
    # if number_of_samples_x > number_of_samples_y:
    number_of_samples = number_of_samples_x

    print('number_of_samples ', number_of_samples)
    print('min(point1[1], point2[1]) ', min(point1[1], point2[1]))
    print('max(point1[1], point2[1])', max(point1[1], point2[1]))
    x_s = np.linspace(start=min(point1[1], point2[1]), stop=max(point1[1], point2[1]), num=number_of_samples)
    y_s = m * x_s + b

    points = [y_s.astype(np.int32), x_s.astype(np.int32)]
    
    # else:
    #     number_of_samples = number_of_samples_y
    #     y_s = np.linspace(start=min(point1[0], point2[0]), stop=max(point1[0], point2[0]), num=number_of_samples)
    #     x_s = (y_s - b) / (m + 1e-9)

    # points = [y_s.astype(np.int32), x_s.astype(np.int32)]
    
    # print('x_s ', x_s)
    # print('y_s ', y_s)
    # print('points ', points)
    return points, number_of_samples

def compute_integral(pt1, pt2, connectivity):
    # (y, x)

    # get the points on the connecting line
    points, num_points = get_points(pt1, pt2)
    '''
    # integral
    try:
        score = connectivity[points].sum()
    except IndexError:  # basically no connectivity
        score = -200
    '''
    print('points, num_points', points, num_points)
    score = connectivity[points].sum()
    return score

if __name__ == "__main__":
    pass