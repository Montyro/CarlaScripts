import argparse
import os
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
import math


# Generate homogenous matrix from euler angles (in radians) and translation
def homogeneousMatrixFromPose(roll, pitch, yaw, x,y,z):
    x11 = np.cos(yaw)*np.cos(pitch)
    x12 = np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll)
    x13 = np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)
    x14 = x
    x21 = np.sin(yaw)*np.cos(pitch)
    x22 = np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll)
    x23 = np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)
    x24 = y
    x31 = -np.sin(pitch)
    x32 = np.cos(pitch)*np.sin(roll)
    x33 = np.cos(pitch)*np.cos(roll)
    x34 = z

    return np.array([[x11,x12,x13,x14],[x21,x22,x23,x24],[x31,x32,x33,x34],[0,0,0,1]])

def hMtoText(hM):
    return " ".join([str(x) for x in hM.flatten()])

#Change euler angles from degrees to radians and fix the range (from -180:180 to 0:360)
def FixEulerAngles(angle):
    if(angle<0):
        return np.radians(360 + angle)
    else:
        return np.radians(angle)

#From Euler angles to a quaternion
def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])

#Inverse of a quaternion
def quaternion_inverse(qx,qy,qz,qw):
    q = np.array([qx,qy,qz,qw])
    q_ = np.array([-qx,-qy,-qz,qw])
    qq_ = [qx**2+qy**2+qz**2+qw**2]
    return q_/qq_

#quaternion multiplication
def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)

#Rotation between euler angles (must transform to another coordinate system)
def EulerAnglesDifference(angle1,angle2):
    a1 = get_quaternion_from_euler(*angle1)
    a2 = get_quaternion_from_euler(*angle2)
    a1 = quaternion_inverse(*a1)

    r = quaternion_multiply(a2,a1)
    return euler_from_quaternion(*r)
    
#From a quaternion get euler angles
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians

def substract_positions(pose1,pose2):
    final_pose = pose1.copy()
    final_pose[:3] = EulerAnglesDifference(pose2[:3],pose1[:3])
    final_pose[3:] = pose1[3:] - pose2[3:]
    return final_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./ply_to_velodyne.py")

    parser.add_argument('--poses', '-p',
        type=str,
        default="00",
        required=True,
        help='Folder with files to change',
    )
    FLAGS,_ = parser.parse_known_args()

    with open("poses.txt") as f:
        poses = f.readlines()
    
    poses_hm = []
    #Poses are supposed to be relative to the first frame
    first_pose = np.array([ float(x) for x in poses[0].split(' ')])
    first_pose[:3] =np.array([FixEulerAngles(x) for x in first_pose[:3]])
    first_pose[-1],first_pose[-2] = first_pose[-2],first_pose[-1]
    first_pose[1],first_pose[2] = -first_pose[2],first_pose[1]

    poses_hm.append(hMtoText(homogeneousMatrixFromPose(*substract_positions(first_pose,first_pose))[:3,:4]))
    print(poses_hm)
    #Each row: roll pitch yaw x y z
    
    for pose in poses[1:]:
        pose = pose.split(' ')
        pose = np.array([float(x) for x in pose])
        pose[-1],pose[-2] = pose[-2],pose[-1]
        pose[1],pose[2] = -pose[2],pose[1]
        pose[:3] =  np.array([FixEulerAngles(x) for x in pose[:3]])
        
        hM = homogeneousMatrixFromPose(*(substract_positions(pose,first_pose)))
        hM = hMtoText(hM[:3,:4])
        poses_hm.append(hM)
           
    with open("poses_hm.txt","w") as f:
        f.write("\n".join(poses_hm))
            
        
       

