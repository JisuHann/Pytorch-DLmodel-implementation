# detectron 적용하기
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.data import MetadataCatalog
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.utils.visualizer import Visualizer
from utils.utils import *
cfg = get_cfg()

def test_setup():
    cfg.merge_from_file("./detectron/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("book",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 100  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (person)
    cfg.MODEL.DEVICE = 'cpu'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = "./detectron/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("book",)
    global predictor
    predictor = DefaultPredictor(cfg)


# model predict image
def segmentation_predict(path):
    image = cv2.imread(path)
    outputs = predictor(image)  # noqa
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)  # noqa
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = Image.fromarray(v.get_image()[:, :, ::-1])
    img.show()  # show prediction result
    mask_array = outputs['instances'][outputs['instances'].pred_classes == 0].pred_masks.numpy()
    num_instances = mask_array.shape[0]
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []

    output = np.zeros_like(image)
    for i in range(num_instances):
        mask_array_instance.append(mask_array[:, :, i:(i + 1)])
        output = np.where(mask_array_instance[i] == True, i+1, output)
    im = Image.fromarray(output).convert('L')
    return
test_setup()