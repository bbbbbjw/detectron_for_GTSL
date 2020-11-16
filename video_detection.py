# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import cv2 as cv
import tqdm
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList
from predictor import VisualizationDemo
import numpy as np 
import _pickle
# constants
WINDOW_NAME = "COCO detections"
import os


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    
    if args.confidence_threshold:
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--videos-input",
        help="dir of a lot of videos",
    )
    parser.add_argument("--video-input", help="Path to video file.")


    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--fps", default=5,type=int,help="set fps")
    parser.add_argument("--n", default=10,type=int,help="size of video-output")
    parser.add_argument(
    "--output",
    help="A file or directory to save output visualizations. "
)
    parser.add_argument("--pkl-output", help="Path to output pkl.")
    return parser
total_feat_out = []
def hook_fn_forward(module, input, output):
    global total_feat_out
    total_feat_out=output 




def deal_video(video_path,args):
    print(video_path)
    N = args.n
    pkl_outpath=args.pkl_output+"/VID/Anet_detection/"
    if not os.path.exists(pkl_outpath):
        os.makedirs(pkl_outpath)
    #读取视频
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second =int(video.get(cv2.CAP_PROP_FPS))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(video_path)
    #返回特征向量
    cap = cv.VideoCapture(video_path)
    frames_feature=[]
    fps_of_video = cap.get(5) 
    itertimes=-1
    group_of_frames=int(frames_per_second/args.fps)
    while cap.isOpened():
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            break
        itertimes+=1
        if itertimes%group_of_frames!=0:continue
        predictor.model.backbone.register_forward_hook(hook_fn_forward)
        frame = cv.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result=predictor(frame)
        model=predictor.model
        features=total_feat_out
        bbox_features = [features[f] for f in model.roi_heads.in_features]
        bbox_features = model.roi_heads.box_pooler(bbox_features, [result["instances"].pred_boxes])
        # 返回准确率前N的结果
        topn= min(N,len(result["instances"].pred_classes))
        frame_dict={}
        #x1,y1,x2,y2
        frame_dict["bboxes"] = result["instances"].pred_boxes[:topn].tensor
        #if wanna xywh use this
        #pre_bbox=frame_dict["bboxes"]
        #tranbbox=torch.zeros_like(frame_dict["bboxes"])
        #tranbbox[:,0]=(pre_bbox[:,0]+pre_bbox[:,2])/2
        #tranbbox[:,1]=(pre_bbox[:,1]+pre_bbox[:,3])/2
        #tranbbox[:,2]=pre_bbox[:,2]-pre_bbox[:,0]
        #tranbbox[:,3]=pre_bbox[:,3]-pre_bbox[:,1]
        #frame_dict["bboxes"]=tranbbox
        frame_dict["classes"] = result["instances"].pred_classes[:topn]
        frame_dict["RoIpoolingFeatures"] = bbox_features[:topn].mean(dim=2).mean(dim=2)
        frame_dict['time']=1.0*itertimes/fps_of_video
        frame_dict['scores'] = result["instances"].scores[:topn]
        if len(result["instances"].pred_classes)==0: continue
        if len(result["instances"].pred_classes)<N:
            print("WARNNING:N is large than predictions size")
            dis=N-len(result["instances"].pred_classes)
            for name,item in frame_dict.items():
                if name=='time':continue
                repeatshape=[]
                repeatshape.append(dis)
                for i in range(len(item[0].shape)):
                    repeatshape.append(1)
                b=torch.ones_like(item[0]).repeat(repeatshape)
                
                item=torch.cat((item,-b),0)
                frame_dict[name]=item
        frames_feature.append(frame_dict)
    video_name=video_path.split('/')[-1]
    save_path=pkl_outpath+video_name.split('.')[-2]+'.pkl'
    print(" output:"+save_path)
    file = open(save_path, 'wb')
    _pickle.dump(frames_feature, file)
    file.close()
    cap.release()
    #输出可视化结果
    if args.output:
        if os.path.isdir(args.output):
            output_fname = os.path.join(args.output, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".mkv"
        else:
            output_fname = args.output
        print(os.path)
        assert not os.path.isfile(output_fname), output_fname
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        assert os.path.isfile(video_path)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
    video.release()
    if args.output:
        output_file.release()
    else:
        cv2.destroyAllWindows()

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    demo = VisualizationDemo(cfg)
    if args.video_input:
        deal_video(args.video_input,args)
    if args.videos_input:
        videos_path=[]
        for root,dirs,files in os.walk(args.videos_input):
            for f in files:
                fpath=root
                fpath+="/"
                fpath+=f
                videos_path.append(fpath)
        for video_path  in videos_path:
            deal_video(video_path,args)