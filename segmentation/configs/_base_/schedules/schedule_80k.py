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