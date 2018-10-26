# MSCeleb1M-C1-Rewrite

Recognize celebrities from their face images! This code rebuilds Tensorflow graph and imports models provided in [wuyuebupt/MSCeleb1MTensorflowModel](https://github.com/wuyuebupt/MSCeleb1MTensorflowModel). For more information, see original repo.

Please download the pre-trained model from [wuyuebupt/MSCeleb1MTensorflowModel](https://github.com/wuyuebupt/MSCeleb1MTensorflowModel) and extract it. Set `--meta_graph` and `--model` arguments to the paths to MetaGraph dump and model file.

```
python3 identify.py \
    --meta_graph=model/graph-0707-065819.meta \
    --model=model/model-990960 \
    --input=FaceImageCroppedWithAlignment.tsv
```
