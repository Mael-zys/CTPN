import cv2
import numpy as np

def box_proj(FM, boxes):
    Project=[]
    for box in boxes:
        pts = np.float32(box)/16.0
        pts1 = np.float32([ [0,0],[0,14],[14,14],[14,0] ])
        M = cv2.getPerspectiveTransform(pts,pts1)
        dst = cv2.warpPerspective(FM,M,(14,14))
        Project.append(dst)
    Project=np.array(Project)
    Project=Project.transpose(0,3,1,2)
    return Project  #shape(batch,channel,14,14)


if __name__ == '__main__':
    FM = np.random.rand(512,512,20)
    boxes=[[[0,0],[0,500],[500,400],[400,100]]]
    R=box_proj(FM,boxes)
    
    print(R)
    print(R.shape)

# a=cv2.imread('test_all/img_1.jpg')

# # h,w = a.shape[:2]
# # print(a.shape)
# # print(h,w)
# pts = np.float32([ [25,370],[55,580],[580,1000],[700,0] ])

# # cv2.circle(a,(0,h-1),36,(0,290,0),-1)

# pts1 = np.float32([ [0,0],[0,140],[140,140],[140,0] ])
# M = cv2.getPerspectiveTransform(pts,pts1)

# dst = cv2.warpPerspective(a,M,(140,140))

# cv2.imwrite('test_all/0.png',dst)