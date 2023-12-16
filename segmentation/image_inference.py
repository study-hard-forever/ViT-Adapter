# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp

'''
命令：
CUDA_VISIBLE_DEVICES=0 python image_inference.py \
  configs/ac/mask2former_beitv2_adapter_large_896_80k_ac_ms.py  \
  work_dirs/mask2former_beitv2_adapter_large_896_80k_ac_ms/iter_80000.pth  \
  data/VOCdevkit/VOC2007/test_jpg/1_bengbianA_2_bengbianA_srcTray_1_srcIndex_1_DL_result_0_0_3_BengBian.jpg \
  --palette AcDataset 
'''
def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('--out', type=str, default="results_iter_80000", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
    
    '''
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(args.img, result,
                            palette=get_palette(args.palette),
                            show=False, opacity=args.opacity)
    mmcv.mkdir_or_exist(args.out)
    out_path = osp.join(args.out, osp.basename(args.img))
    cv2.imwrite(out_path, img)
    print(f"Result is save at {out_path}")
    '''
    
    # 遍历文件夹
    test_jpg_path = r'data/VOCdevkit/VOC2007/test_jpg'
    import os
    imgs = os.listdir(test_jpg_path)
    for img_path in imgs:
        img_path = os.path.join(test_jpg_path,img_path)
        result = inference_segmentor(model, img_path)
        # print(result)
        # show the results
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(img_path, result,
                                palette=get_palette(args.palette),
                                show=False, opacity=args.opacity)
        mmcv.mkdir_or_exist(args.out)
        out_path = osp.join(args.out, osp.basename(img_path))
        cv2.imwrite(out_path, img)
        print(f"Result is save at {out_path}")

if __name__ == '__main__':
    main()