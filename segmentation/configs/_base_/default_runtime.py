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
