输出在VID/Anet_detection中，对于每个视频输出一个以video_id为名的pkl文件，储存了一个list，list元素是一个dict代表采样帧的信息，包括帧的时间，一个numpy向量记录了N个BBox的位置(x,y,h,w)，一个numpy向量记录N个BBox的RoI pooling特征，一个numpy向量记录了N个BBox预测置信度，一个numpy向量记录N个BBox预测的类别。

运行方式：
!python video_detection.py  --output video-out.mkv --pkl-output pkldata --video-input video-clip.mp4 --config-file detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml --confidence-threshold 0.6  \
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
不需要输出可视化结果时请删去—output,更多可选参数输入 -h获取
需要批量转化，请使用 –videos-input：
!python video_detection.py  --output video-out.mkv --pkl-output pkldata --videos-input videos_dir(视频所在文件夹) --config-file detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml --confidence-threshold 0.6  \
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
数据下载：
在GTSL/data中python video_download.py 输出在/video_data中
