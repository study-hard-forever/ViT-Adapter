# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings（默认）
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

# 上述也可以按照epoch形式进行迭代
# runner = dict(type='EpochBasedRunner', max_epochs=100)  # 按epoch的方式进行迭代
# checkpoint_config = dict(by_epoch=True, interval=1)  # 每多少次迭代保存一次模型
# evaluation = dict(interval=1, metric='mIoU')  # 每多少次迭代计算一次指标

'''
可视化（未添加）：
数据和结果的可视化：模型测试或验证期间的可视化数据样本
MMSegmentation 提供了 SegVisualizationHook ，它是一个可以用于可视化 ground truth 和在模型测试和验证期间的预测分割结果的钩子 。 它的配置在 default_hooks 中
参考链接：https://mmsegmentation.readthedocs.io/zh-cn/main/user_guides/visualization.html#id2
注：此处来自文档mmsegmentation1.2.2,此处版本为0.20.2，略有不同，如上述checkpoint配置为checkpoint_config而下面的都包含在default_hooks中，为checkpoint
'''
'''
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
'''