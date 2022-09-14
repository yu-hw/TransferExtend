# TransferExtend

## 协作规范

唯一分支：`main`

本地开发后提交 pr 到 `main` 分支即可
注意及时 `rebase` 跟踪进度

## 文件结构

```
├── logs        // 日志
│   └── log
├── model.py    // 模型文件
├── setting.py  // 训练配置程序
├── train.py    // 训练程序
├── vocab.py    // 词表构建程序
├── utils.py    // 一些共用函数
├── log.py      // 日志记录程序
└── README.md   // 说明文件
```

## 模型结构

waiting for update

## 当前任务

- [x] dataset-fl 初步处理，排除不可用数据
- [x] 完成 encoder-decoder 模型构建
- [ ] 测试 encoder-decoder 模型
- [x] 完成 MLP 模型构建
- [x] 合并 encoder-decoder 和 MLP 模型（multitask）
- [ ] 测试 multitask 模型

## 当前问题

- [x] decoder force_teaching
- [x] mlp 是否需要将 src 作为输入，如果需要，如何控制长度
- [ ] decoder initial hidden state 获取问题
- [ ] mlp initial input 获取问题
- [ ] loss 计算方式

## 资料补充

[Sequence to Sequence (seq2seq) and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#attention_bahdanau_luong)

[NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)

[Bahdanau 注意力](http://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html)