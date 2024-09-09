
import os
from typing import List, Dict, Any, Optional, Union
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
from detectron2.utils.visualizer import ColorMode, Visualizer
from adet.config import get_cfg
from adet.utils.visualizer import TextVisualizer



class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5



class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.TRANSFORMER.ENABLED

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        if cfg.MODEL.BACKBONE.NAME == "build_vitaev2_backbone":
            self.predictor = ViTAEPredictor(cfg)


    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(image, self.metadata, instance_mode=self.instance_mode, cfg=self.cfg)
        else:
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output
    

class SceneTextDetection:


    def __init__(
        self,
        model_weight: str,
        config_file:str =  './configs/R_50/mlt19_multihead/finetune.yaml',
        parallel: bool= False,
        num_gpus: int = torch.cuda.device_count()
    ):
        self.logger = setup_logger()
        self.cfg = self.setup_cfg(
            model_weight,
            config_file
        )
        self.demo = VisualizationDemo(self.cfg, parallel=parallel)
        self.num_gpus = num_gpus if parallel else 1
        self.predictor = AsyncPredictor(self.cfg, num_gpus=self.num_gpus)

    

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
        output_path: Optional[str] = None
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
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        results = []

        if len(input_path_list) == 1:
            # Single image case: use default predictor
            return self._process_single_image(input_path_list[0], output_path)
        else:
            # Multiple images: use AsyncPredictor
            return self._process_multiple_images(input_path_list, output_path)
        
    def _process_single_image(self, image_path: str, output_path: Optional[str]) -> List[Dict[str, Any]]:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        start_time = time.time()
        predictions, visualized_output = self.demo.run_on_image(img)

        self.logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(predictions["instances"]), time.time() - start_time
            )
        )

        cv_image = cv2.cvtColor(np.array(visualized_output.get_image()), cv2.COLOR_RGB2BGR)
        instances = predictions['instances'].to('cpu')

        result = [{
            'img': cv_image,
            'instances': instances
        }]

        if output_path:
            input_basename = os.path.splitext(os.path.basename(image_path))[0]
            out_filename = os.path.join(output_path, f"{input_basename}.webp")
            cv2.imwrite(out_filename, cv_image, [cv2.IMWRITE_WEBP_QUALITY, 80])

        return result

    def _process_multiple_images(self, image_paths: List[str], output_path: Optional[str]) -> List[Dict[str, Any]]:
        results = []
        
        # Load all images first
        images = [(path, cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)) for path in image_paths]
        
        # Submit all tasks to the AsyncPredictor
        for _, img in images:
            self.predictor.put(img)

        # Process results as they become available
        for idx, (path, img) in enumerate(tqdm(images)):
            start_time = time.time()
            predictions = self.predictor.get()
            
            visualizer = TextVisualizer(img, self.demo.metadata, instance_mode=self.demo.instance_mode, cfg=self.cfg)
            
            if "instances" in predictions:
                instances = predictions["instances"].to(self.demo.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

            self.logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            cv_image = cv2.cvtColor(np.array(vis_output.get_image()), cv2.COLOR_RGB2BGR)
            instances = predictions['instances'].to('cpu')

            results.append({
                'img': cv_image,
                'instances': instances
            })

            if output_path:
                input_basename = os.path.splitext(os.path.basename(path))[0]
                out_filename = os.path.join(output_path, f"{input_basename}.webp")
                cv2.imwrite(out_filename, cv_image, [cv2.IMWRITE_WEBP_QUALITY, 80])

        return results



    
