# TransferExtend

## 协作规范

唯一分支：`main`

本地开发后提交 pr 到 `main` 分支即可
注意及时 `rebase` 跟踪进度

## 文件结构

├── localtools  //存放数据预处理程序和其他调试用工具
│   └──
├── model.py    //模型文件
├── setting.py  //训练配置文件
├── train.py    //训练文件
└── README.md   //说明文件

## 模型结构

waiting for update

## 当前任务

- [ ] dataset-fl 初步处理，排除不可用数据
- [ ] 完成 encoder-decoder 模型构建
- [ ] 测试 encoder-decoder 模型
- [ ] 完成 MLP 模型构建
- [ ] 合并 encoder-decoder 和 MLP 模型（multitask）
- [ ] 测试 multitask 模型