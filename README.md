
# HarDNeXt-Pytorch

[HarDNeXt: A Stage Receptive Field and Connectivity Aware Convolution Neural Network](https://etd.lib.nctu.edu.tw/cgi-bin/gs32/hugsweb.cgi?o=dnthucdr&s=id=%22G021080626010%22.&searchmode=basic)

<!-- ## Developing Log
<details><summary> <b>Expand</b> </summary>


* `2020-10-15` - Add [HarDNet_residual](https://), [HarDNet_refine1](https://), [HarDNet_no_bot](https://) performance result
* `2020-10-08` - Add speed testing code
* `2020-10-07` - Add  [ShuffleNetV2](https://arxiv.org/abs/1807.11164), [ResNet50, 101](https://arxiv.org/abs/1512.03385)
* `2020-10-06` - Add [HarDNet_residual](https://), [HarDNet_refine1](https://), [HarDNet_no_bot](https://)
* `2020-10-05` - Add [torchstat](https://github.com/Swall0w/torchstat) function
* `2020-09-26` - mirror folk from [Pytorch_HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)
</details> -->

### Image Classification (ImgaeNet 2012)
![](https://i.imgur.com/gjAEC5v.jpg)

---

#### Performance on ImageNet2012 Validation Set 
|   Model Name    |  MAC(B)  | FPS (2080 Ti)(FP32) |    EPI    | Top-1 Acc (FP32) |
|:---------------:|:--------:|:-------------------:|:---------:|:----------------:|
|    ResNet-34    |   3.67   |         347         |   0.435   |       73.3       |
|  DenseNet-121   |   2.88   |         97          |   1.08    |      74.65       |
|   HarDNet-39    |   2.12   |         245         |   0.487   |       74.4       |
| **HarDNeXt-28** | **2.07** |       **359**       | **0.354** |    **74.09**     |
| **HarDNeXt-32** | **2.11** |       **324**       | **0.397** |     **74.5**     |
| **HarDNeXt-39** | **2.84** |       **299**       | **0.466** |    **75.36**     |


|   Model Name    |  MAC(B)  | FPS (2080 Ti)(FP32) |    EPI    | Top-1 Acc (FP32) |
|:---------------:|:--------:|:-------------------:|:---------:|:----------------:|
|    ResNet-50    |   4.12   |         257         |   0.643   |      76.15       |
|  DenseNet-169   |   3.42   |         71          |   1.571   |        76        |
|   HarDNet-68    |   4.26   |         149         |   0.836   |       76.5       |
| **HarDNeXt-50** | **3.51** |       **215**       | **0.619** |    **76.32**     |

|   Model Name    |  MAC(B)  | FPS (2080 Ti)(FP32) |   EPI    | Top-1 Acc (FP32) |
|:---------------:|:--------:|:-------------------:|:--------:|:----------------:|
|   ResNet-101    |   7.84   |         141         |  1.207   |       77.3       |
|  DenseNet-201   |   4.37   |         58          |   1.92   |       77.2       |
|   ResNeXt-50    |   4.27   |         166         |   0.92   |      77.62       |
| **HarDNeXt-56** | **6.32** |       **182**       | **0.89** |     **77.3*

### Training
```
python ./src/main.py --model_name hardnext --arch 39 /imagenet/data/path
```
### Model Summary

```
python ./src/main.py --model_name hardnext --arch 39  --h 224 --w 224
```
### Model GPU Speed Test

```
python ./src/model_speed_test.py --model_name hardnext --arch 39  --h 224 --w 224 --gpu 0
```
