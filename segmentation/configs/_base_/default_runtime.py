# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),  # 也可以改为True，使迭代方式的计数方式按照epoch形式
        dict(type='TensorboardLoggerHook')  # 默认为关闭，现在打开就开启TensorBoard日志了
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

'''
可视化（未添加）：
监控训练时候的状态：
启动时添加地址：--work-dir work_dir/test_visual
示例如下：python tools/train.py configs/pspnet/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py --work-dir work_dir/test_visual
参考链接：https://mmsegmentation.readthedocs.io/zh-cn/main/user_guides/visualization.html#id2
注：此处来自文档mmsegmentation1.2.2,此处版本为0.20.2，略有不同，如0.20.2版本的checkpoint配置为checkpoint_config，而1.2.2文档中checkpoint配置包含在default_hooks中，为checkpoint
'''

'''
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
'''