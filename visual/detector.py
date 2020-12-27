from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import  MetadataCatalog
from tqdm import tqdm
import os
import cv2



class Object_dectector:
    """
    Detect objects in an image using models supported by detectron2
    """
    def __init__(self,cfg_path,threshold=0.5):
        """
        :param cfg_path:  cfg path, possible path can be found in model_zoo file
        :param threshold: ros_heads score threshold
        """
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
        self.predictor = DefaultPredictor(self.cfg)

    def detect(self,image_path,out_dir=None):
        """
        Detect objects in an image and save it
        :param image_path: input image path
        :param out_dir: the dir to save output images
        :return:
        """
        img = cv2.imread(image_path)
        outputs = self.predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        if out_dir:
            cv2.imwrite(out_dir,out.get_image()[:, :, ::-1])
        else:
            cv2.imwrite("img", out.get_image()[:, :, ::-1])


    def detect_imgs(self,folder_path,out_dir):
        """
        Detect objects in all images in a folder
        :param folder_path: must be in this format "../../../"
        :param out_dir: must be in this format "../../../
        Note out_dir should not be in folder_path
        :return:  None
        """
        if not os.path.exists(folder_path):
            raise ValueError("Folder path doesn't exist")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for file in tqdm(os.listdir(folder_path)):
            img = folder_path+file
            self.detect(img,out_dir+file)

# An example of using this class
if __name__ == "__main__":
    cfg_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    image_path = "../data/00003_rgb.png"
    detector = Object_dectector(cfg_path,0.25)
    #detector.detect(image_path,"img.png")
    detector.detect_imgs("../data/coco/","../coco_detect_out/")





