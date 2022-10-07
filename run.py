import os

setting_path = '/home/LAB/caohl/TransferExtend/train-setting'
outputs_path = '/home/LAB/caohl/TransferExtend/train-outputs'

setting_list = os.listdir(setting_path)

for js in setting_list:
    a = js.split('.')
    print('running:' + a[0])
    cmd = 'python train.py' + \
        ' ' + os.path.join(setting_path, js) + \
        ' ' + '>' + ' ' + os.path.join(outputs_path, a[0] + '.out')
    os.system(cmd)