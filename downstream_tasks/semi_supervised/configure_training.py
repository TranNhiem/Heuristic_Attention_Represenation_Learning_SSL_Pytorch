DATASET = 'ten_per'
METHOD = 'MASSL_600'
SCHEDULER = 'step'
WEIGHTS = '/code_spec/downstream_tasks/semi_supervised/MASSL_3MLP_512_600_.ckpt' if METHOD == 'MASSL_600' else '/code_spec/downstream_tasks/semi_supervised/MASSL_3MLP_512_600_.ckpt' if METHOD == 'MASSL_600ep' else '/code_spec/downstream_tasks/semi_supervised/Baseline_300epoch.ckpt'
