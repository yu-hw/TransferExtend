# TransferExtend

## 协作规范

唯一分支：`main`

本地开发后提交 pr 到 `main` 分支即可
注意及时 `rebase` 跟踪进度

## 模型结构

![训练模型](https://github.com/Chlience/TransferExtend/blob/main/pic/train.drawio.svg)

## 文件结构

```
.
├── data
├── localtools
│   ├── data_process.out
│   ├── data_process.py
│   ├── make-setting.py
│   ├── run.py
│   └── run.sh
├── module
│   ├── dataload.py
│   ├── dataprocess.py
│   ├── iterator.py
│   ├── loss.py
│   ├── model.py
│   ├── optimizer.py
│   ├── statistics.py
│   ├── utils.py
│   └── vocab.py
├── pic
│   └── train.drawio.svg
├── README.md
├── train.py
```

## 当前任务

- [x] dataset-fl 初步处理，排除不可用数据
- [x] 完成 encoder-decoder 模型构建
- [x] 测试 encoder-decoder 模型
- [x] 完成 MLP 模型构建
- [x] 合并 encoder-decoder 和 MLP 模型（multitask）
- [x] 测试 multitask 模型
- [ ] diff origin
  - [ ] 查看 diff 后如何截断数据

## 当前问题

- [x] decoder force_teaching
- [x] mlp 是否需要将 src 作为输入，如果需要，如何控制长度
- [x] decoder initial hidden state 获取问题
- [x] mlp initial input 获取问题
- [x] loss 计算方式
- [x] 多训练配置

## 资料补充

[Sequence to Sequence (seq2seq) and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#attention_bahdanau_luong)

[NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)

[Bahdanau 注意力](http://zh.d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html)