from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from tqdm import tqdm
import lmdb
import torch
import os
import cv2

class FeatureExtractor:

    def check_path(self,path):
        if not os.path.exists(path):
            raise ValueError("Path doesn't exists !!")

    def create_dir(self,out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def extract_feature(self,inputs,is_finetune):
        raise  NotImplementedError

    def save_feats_to_file(self, in_dir, out_dir,max_size = 1099511627776):
        self.check_path(in_dir)
        self.create_dir(out_dir)
        env = lmdb.open(out_dir, map_size=max_size)
        txn = env.begin(write=True)
        for file in tqdm(os.listdir(in_dir)):
            img = in_dir + file
            feats = self.extract_feature(img)
            txn.put(key=img,value=feats)
        txn.commit()
        env.close()



class ObjectFeatureExtractor(FeatureExtractor):

    def __init__(self, cfg_path, num_obj=37, threshold=0.01):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.TEST.DETECTIONS_PER_IMAGE = num_obj
        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)


    def extract_feature(self,inputs,output_instances=False):
        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        self.model.eval()
        with torch.no_grad():
            images = self.model.preprocess_image(inputs)
            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images,features,None)
            features_ = [features[f] for f in self.model.roi_heads.box_in_features]
            box_features = self.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
            box_features = self.model.roi_heads.box_head(box_features)
            predictions = self.model.roi_heads.box_predictor(box_features)
            pred_instances, pre_inds = self.model.roi_heads.box_predictor.inference(predictions,proposals)
            pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
            pred_instances = self.model._postprocess(pred_instances, inputs, images.image_sizes)
            feats = box_features[pre_inds]
            if output_instances:
                return feats,pred_instances
            return feats



class ImageFeatureExtractor(FeatureExtractor):
    pass


if __name__ == "__main__":
    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ofe = ObjectFeatureExtractor(cfg_path)
    img_path = "../data/test.jpg"
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2,0,1))
    inputs = [{"image":image,"height":height,"width":width}]
    ofe.extract_feature(inputs)
