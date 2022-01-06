import torch
from detectron2.detectron2.utils.logger import setup_logger
from detectron2.detectron2.modeling import build_model
import cv2
from detectron2.detectron2.config import get_cfg
import os

setup_logger()
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

cfg = get_cfg()
cfg.MODEL.DEVICE= 'cpu'
cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "detectron://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (pnumonia)
#Just run these lines if you have the trained model im memory
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
#build model
model = build_model(cfg)
model.eval()

inputs_list=[]
path = "../tmp_data/val"
for filename in os.listdir(path):
    if(filename != 'via_project.json' and filename != '.DS_Store'):
        im = cv2.imread(path + "/"+filename)
        height, width = im.shape[:2]
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        inputs_list.append([{"image": image, "height": height, "width": width}])
lenn = len(inputs_list)
print("all image num is ", lenn)
feats=[]
feats_list=[]

for i in range(lenn):
    inputs = inputs_list[i]
    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        # features of the proposed boxes
        print(i, len(box_features[pred_inds].tolist()))
        if(len(box_features[pred_inds].tolist())!=0):
            for i in box_features[pred_inds].tolist():
                feats_list.append(i)
        feats.append(box_features[pred_inds])
