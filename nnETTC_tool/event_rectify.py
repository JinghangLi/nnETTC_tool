import numpy as np
import cv2

import csv


def conf_to_K(conf):
    K = np.eye(3)
    K[[0, 1, 0, 1], [0, 1, 2, 2]] = conf
    return K

def generate_rectify_map(width, height, camera_matrix, dist_coeffs, R_rect0=None, P_rect0=None, criteria=None):
    """
    使用 cv2.undistortPointsIter 生成图像的 rectify_map。
    
    参数:
        - width (int): 图像的宽度。
        - height (int): 图像的高度。
        - camera_matrix (np.ndarray): 相机内参矩阵 (3x3)，描述了相机的焦距、主点等参数。
        - dist_coeffs (np.ndarray): 畸变系数，长度为 4、5 或更多，描述相机的畸变模型。
        - R_rect0 (np.ndarray): 矫正矩阵，用于将图像坐标系对齐，通过立体标定得到。
        - P_rect0 (np.ndarray): 投影矩阵，用于将图像坐标投影到标准化平面，通过立体标定得到。
        - criteria (tuple, optional): 迭代停止条件，默认为 `(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1e-6)`。

    返回:
        - rectify_map (np.ndarray): 去畸变后的像素坐标映射，形状为 (height, width, 2)。
    """
    
    def _save_rectify_map_to_csv(rectify_map, filename):
        """
        将 3D rectify_map 保存为 CSV 文件。
        参数:
            - rectify_map (np.ndarray): 形状为 (height, width, 2) 的去畸变映射。
            - filename (str): 保存的 CSV 文件名。
        """
        height, width, _ = rectify_map.shape
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入 CSV 的标题行
            writer.writerow(['x_original', 'y_original', 'x_rectified', 'y_rectified'])
            # 遍历 rectify_map 并写入每个像素的映射关系
            for y in range(height):
                for x in range(width):
                    rectified_coords = rectify_map[y, x]
                    # print(f"({x}, {y}) -> ({rectified_coords[0]}, {rectified_coords[1]})")
                    writer.writerow([x, y, rectified_coords[0], rectified_coords[1]])
                    
                    
    # 默认迭代停止条件
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1e-6)
    # 生成所有像素的网格坐标
    y, x = np.indices((height, width), dtype=np.float32)  # 生成坐标网格
    grid = np.stack((x, y), axis=-1).reshape(-1, 1, 2)  # 将x, y组合成一个 (480, 640, 2) 的矩阵
    # 使用 undistortPointsIter 计算去畸变后的坐标
    if R_rect0 is None or P_rect0 is None:
        undistorted_points = cv2.undistortPointsIter(grid, camera_matrix, dist_coeffs, criteria=criteria)
        # print message in red color
        print("\033[91m" + "NO R_rect0 or P_rect0 provided for rectification." + "\033[0m")
    else:
        undistorted_points = cv2.undistortPointsIter(
            grid, camera_matrix, dist_coeffs, R=R_rect0, P=P_rect0, criteria=criteria
        )
        
    # 将结果转换为图像大小 (height, width, 2)
    rectify_map = undistorted_points.reshape(height, width, 2)
    
    # _save_rectify_map_to_csv(undistorted_points, "undistorted_points.csv")

    return rectify_map
