#处理骨骼数据
import os

import numpy as np
from RGB.tool import load_config
import pandas as pd
from tqdm.auto import tqdm
def get_row_data(path:str)->np.array:
    """
    :param path: .npy文件保存地址
    """
    return np.load(path)
def str_to_float(list):
    """
    :param list: 需要转换的列表格式
    :return: 转换形式的list
    """
    #将列表中的str转换为float
    new_list = []
    for i in range(len(list)):
        new_list.append(float(list[i]))
    return new_list
def pick_key_ponit_data(skeleton_frame:pd.DataFrame):
    """
    :param skeleton_frame: 输入骨骼数据的pandas框架
    :return 返回以(x,y,z,Qw,Qx,Qy,Qz)为主的每一帧的骨骼坐标
    """
    X = str_to_float(skeleton_frame[3][1:].tolist())
    Y = str_to_float(skeleton_frame[4][1:].tolist())
    Z = str_to_float(skeleton_frame[5][1:].tolist())
    Qw = str_to_float(skeleton_frame[6][1:].tolist())
    Qx = str_to_float(skeleton_frame[7][1:].tolist())
    Qy = str_to_float(skeleton_frame[8][1:].tolist())
    Qz = str_to_float(skeleton_frame[9][1:].tolist())
    all = np.stack((X, Y, Z, Qw, Qx, Qy, Qz), axis=1)

    return all

def reduce_keypoints(input_data:np.array) -> np.array:
    """
        :param input_data: 输入.npy格式的骨骼数据，以去除关键点, 输入的维度应该是(video_number, frames, keypints_numbers, channels)
    """
    start = load_config('reduce_keypoints')['start']
    end = load_config('reduce_keypoints')['end']

    return np.concatenate((input_data[:,:,:start,:],input_data[:,:,end:,:]),axis=2)

def scale_center(input_data:np.array) -> np.array:
    """
    :param input_data: 输入.npy格式的骨骼数据,用于将所有channel的数据归一化，输入的维度应该是(video_number, frames, keypint_numbers, channels)
    """
    seq_list = []
    for singel_video in input_data:
        pose_list = []
        for frame in singel_video:
            bbox = np.array([[np.min(frame[:,0]),np.max(frame[:,0])],
                             [np.min(frame[:,1]),np.max(frame[:,1])],
                             [np.min(frame[:,2]),np.max(frame[:,2])],
                             [np.min(frame[:,3]),np.max(frame[:,3])],
                             [np.min(frame[:,4]),np.max(frame[:,4])],
                             [np.min(frame[:,5]),np.max(frame[:,5])],
                             [np.min(frame[:,6]),np.max(frame[:,6])]])
            max_dim = bbox[:,1] - bbox[:,0]
            frame[:,:] = (frame[:,:] - bbox[:,0]) / max_dim
            pose_list.append(frame)
        seq = np.stack(pose_list)#将所有的帧stack，组成一个视频
        seq_list.append(seq)
    result = np.stack(seq_list)
    return result


def label_to_onehot(input_data:np.array)->np.array:
    """
    :param input_data: 输入label的.npy文件，将输入的动作类型转换为onehot编码
    """
    onehot = np.eye(input_data.shape[0],dtype=np.uint8)
    onehot_list = []
    for label in input_data:
        onehot_label = onehot[label]
        onehot_list.append(onehot_label)
    all_one_hot_label = np.stack(onehot_list)
    return all_one_hot_label

def skeleton_data_format(root_path_skeleton:str,zero_index:list):
    skeleton_data_list = np.load(root_path_skeleton)
    all_skeleton = []
    stage_skeleton = []
    count = 0
    for i in range(len(skeleton_data_list)):
        if i in zero_index:
            continue
        else:
            stage_skeleton.append(skeleton_data_list[i])
            count += 1
            if count % 32 == 0:
                all_skeleton.append(np.stack(stage_skeleton))
                stage_skeleton = []
    return all_skeleton




if __name__ == '__main__':
    #获取根目录
    root_labels = 'E:/mdxDataset'
    file_path = os.listdir(root_labels)
    for path in tqdm(file_path, desc="All dataset processing ..."):
        every_label_file = f'{root_labels}/{path}/{path}/Labels.txt'
        labels_frame = pd.read_csv(every_label_file, sep=" ", header=None)
        zero_index = [index for index, x in enumerate(labels_frame[1].tolist()) if x == 0]
        skeleton_format = np.stack(skeleton_data_format(root_path_skeleton=f'E:/mdx/SelfAttention_ActionRecognition/Stacked_skeleton_data/{path}/{path}.npy', zero_index=zero_index))
        # np.save(f'E:/mdx/SelfAttention_ActionRecognition/Input_dataset/{path}/{path}_skeleton.npy', skeleton_format)
        print(f'{np.stack(skeleton_format).shape} has been saved in Input_dataset/{path}/{path}_skeleton.npy')








