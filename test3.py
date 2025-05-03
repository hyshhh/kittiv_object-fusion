import mayavi.mlab as mlab
from kitti_object import kitti_object, show_image_with_boxes, show_lidar_on_image, \
    show_lidar_with_boxes, show_lidar_topview_with_boxes, get_lidar_in_image_fov, \
    show_lidar_with_depth,fusion_2D,fusion3D,fusion333D,fusion3D1,fusion3D2,fusion3D3
from viz_util import draw_lidar
import cv2
from PIL import Image
import time

class visualization:
    # data_idx: determine data_idx
    def __init__(self, root_dir=r'/home/hys/OpenPCDet/kitti_object_vis-master/data/object', data_idx=45):
        dataset = kitti_object(root_dir=root_dir)

        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        print("There are {} objects.".format(len(objects)))
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        pc_velo = dataset.get_lidar(data_idx)[:, 0:3]  # 显示bev视图需要改动为[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        self.detects=dataset.get_detects(data_idx)
        self.detects3D=dataset.get_detects3D(data_idx)
        self.point=dataset.get_point(data_idx)
        self.point100=dataset.get_point100(data_idx)
        self.FPN=dataset.get_FPN(data_idx)
        seg=dataset.get_seg(data_idx)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        self.seg=seg
        # init the params
        self.objects = objects
        self.img = img
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.pc_velo = pc_velo
        self.calib = calib
    # 5. Lidar绘制3D bbox      #img_fov=True 
    def show_lidar_with_3d_boxes(self):
        show_lidar_with_boxes(self.pc_velo, self.objects, self.calib, True, self.img_width, self.img_height)
        mlab.show()
    def show_lidar_with_3d_boxes1(self):
        show_lidar_with_boxes(self.pc_velo, self.detects3D, self.calib, True, self.img_width, self.img_height)
        mlab.show()
    def show_lidar_with_3d_boxes2(self):
        show_lidar_with_boxes(self.pc_velo, self.FPN, self.calib, True, self.img_width, self.img_height)
        mlab.show()
    def show_lidar_with_3d_boxes3(self):
        show_lidar_with_boxes(self.pc_velo, self.point, self.calib, True, self.img_width, self.img_height)
        mlab.show()
    def show_lidar_with_3d_boxes4(self):
        show_lidar_with_boxes(self.pc_velo, self.point100, self.calib, True, self.img_width, self.img_height)
        mlab.show()              
    # 6. Lidar绘制FOV图
    def show_lidar_with_fov(self):
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(self.pc_velo, self.calib,
                                                                  0, 0, self.img_width, self.img_height, True)
        draw_lidar(imgfov_pc_velo)
        mlab.show()
    # 3. 图片上绘制3D bbox
    def show_image_with_3d_boxes(self):
        show_image_with_boxes(self.img, self.objects, self.calib, show3d=True)
        cv2.waitKey(0)
    # 4. 图片上绘制Lidar投影
    def show_image_with_lidar(self):
        show_lidar_on_image(self.pc_velo, self.img, self.calib, self.img_width, self.img_height)
        mlab.show()
        
    def show_lidar_with_3dview(self):
        draw_lidar(self.pc_velo)
        mlab.show()

    # 8. Lidar绘制BEV图
    def show_lidar_with_bev(self):
        from kitti_util import draw_top_image, lidar_to_top
        top_view = lidar_to_top(self.pc_velo)
        top_image = draw_top_image(top_view)
        cv2.imshow("top_image", top_image)
        cv2.waitKey(0)

    # 9. Lidar绘制BEV图+2D bbox
    def show_lidar_with_bev_2d_bbox(self):
        show_lidar_topview_with_boxes(self.pc_velo, self.objects, self.calib)
        mlab.show()
        

        
if __name__ == '__main__':
    kitti_vis = visualization()
    #kitti_vis.show_image_with_3d_boxes()   #3-1 3D检测框生成并投影到图像
    #kitti_vis.show_image_with_lidar()        #4 点云图像时空对齐
    kitti_vis.show_lidar_with_3d_boxes()      #3-2 3D检测框生存
    kitti_vis.show_lidar_with_3d_boxes1()      #3-2 3D检测框生存
    kitti_vis.show_lidar_with_3d_boxes2()      #3-2 3D检测框生存
    kitti_vis.show_lidar_with_3d_boxes3()      #3-2 3D检测框生存
    kitti_vis.show_lidar_with_3d_boxes4()      #3-2 3D检测框生存
    #kitti_vis.show_lidar_with_fov()
    print('...')

