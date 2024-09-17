import os
from typing import List, Dict, Any, Optional, Union, Tuple
import glob
import time
import cv2
from tqdm.notebook import tqdm as tqdm  
import numpy as np
import torch
import multiprocessing as mp
import bisect
import atexit

from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from adet.config import get_cfg
import detectron2.data.transforms as T

from detectron2.modeling import build_model
from adet.utils.visualizer import TextVisualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog




class BatchPredictor:
    """
    Batch version of the DefaultPredictor to process multiple images at once.
    Compared to the DefaultPredictor, it processes a batch of images instead of a single image.

    This class takes a configuration object (cfg), loads a pre-trained model, 
    and applies the necessary preprocessing (resizing, format conversion) to the input images 
    before running batch inference. It's designed to handle multiple images efficiently.

    Attributes:
        cfg (CfgNode): The configuration object containing model and dataset details.
        model (torch.nn.Module): The model built from the configuration.
        metadata (Metadata): Metadata information from the dataset (if available).
        aug (T.Augmentation): The augmentation transform for resizing the input images.
        input_format (str): Format of the input image, either "RGB" or "BGR".
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
    
    def __call__(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Process a batch of images and run inference on them.
        Args:
            images (List[np.ndarray]): A list of images, where each image is a NumPy array of shape (H, W, C) and in BGR order (if using OpenCV).
        
        Returns:    
             List[Dict]: A list of prediction dictionaries for each image in the batch. Each dictionary contains the model's predictions for a single image.
        """ 

        assert isinstance(images, list), "Input must be a list of images"
        processed_images: List[torch.tensor] = []
        original_sizes: List[Tuple[int, int]] = []


        for original_img in images:
            print(original_img.shape)
            if self.input_format == 'RGB':
                original_img = original_img[:, :, ::-1]
            height, width = original_img.shape[:2]
            image = self.aug.get_transform(original_img).apply_image(original_img)
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            processed_images.append(image_tensor)
            original_sizes.append((height, width))
        
        batch_images = torch.stack(processed_images).to(self.cfg.MODEL.DEVICE)
        inputs_model = \
            [{'image': img, 'height': h, 'width': w} for img, (h, w) in zip(batch_images, original_sizes)]
        
        with torch.no_grad():
            predictions: List[dict] = self.model(inputs_model)
        return predictions


class SceneTextDetection:


    def __init__(
        self,
        model_weight: str,
        config_file:str =  './configs/R_50/mlt19_multihead/finetune.yaml'
    ):
        self.logger = setup_logger()
        self.cfg = self.setup_cfg(
            model_weight,
            config_file
        )
#         self.default_predictor = DefaultPredictor(self.cfg)
        self.default_predictor = "dasd"
        self.batch_predictor = BatchPredictor(self.cfg)
    

    def setup_cfg(
        self,
        model_weight: str,
        config_file: str
    ):
        opts = ['MODEL.WEIGHTS', model_weight]
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()
        return cfg
    

    def process_images(
        self,
        input_path: Union[str, List[str]],
    ) -> List[Dict[str, Any]]: 
        if isinstance(input_path, list):
            input_path_list = input_path
        elif os.path.isdir(input_path):
            input_path_list = [os.path.join(input_path, fname) for fname in os.listdir(input_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        elif os.path.isfile(input_path):
            input_path_list = [input_path]
        else:
            input_path_list = glob.glob(os.path.expanduser(input_path))

        assert input_path_list, "No input images found"
        # if output_path:
        #     os.makedirs(output_path, exist_ok=True)


        if len(input_path_list) == 1:
            # Single image case: use default predictor
            return self._process_single_image(input_path_list[0])
        else:
            # Multiple images: use AsyncPredictor
            return self._process_multiple_images(input_path_list)
        
    def _process_single_image(self, image_path: str) -> List[Dict[str, Any]]:


        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        start_time = time.time()
        predictions = self.default_predictor(img)

        self.logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions["instances"]), time.time() - start_time
            )
        )

        instances = predictions['instances'].to('cpu')

        result = [{
            'instances': instances
        }]

        return result

    def _process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        
#         images = [(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)) for path in image_paths]
        images = [ cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
        

        start_time = time.time()
        predictions = self.batch_predictor(images)

        results = []

        for i, pred in enumerate(predictions):
            instances = pred['instances'].to('cpu')
            path = image_paths[i] 

            self.logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(instances), (time.time() - start_time) / len(images)
                )
            )
            
            results.append([{
                'path': path,
                'instances': instances
            }])

        return results