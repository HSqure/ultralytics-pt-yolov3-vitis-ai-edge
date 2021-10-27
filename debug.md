# DEBUG日志
- **主要脚本程序：**
`finetune.sh`

- **基本流程：**

	```mermaid
  graph LR
      B([float模型])-->C[量化]-->D([int8模型])-->A[剪枝]-->E([剪枝后的int8模型])-->F[导出模型权重]-->G[读取模型权重]
  
  ```
    **第一步**：

    量化裁减网络并导出模型参数。
    ```shell
	  python quant_fast_finetune.py --quant_mode calib --fast_finetune
    ```
    核心代码：
    ```python
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device)
        quant_model = quantizer.quant_model
  
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, register_buffers))
        quantizer.export_quant_config()
    ```
    **第二步**：

    读取参数并导出`.xmodel`
    ```shell
    python quant_fast_finetune.py  --quant_mode test --subset_len 1 --batch_size=8 --fast_finetune --deploy
    ```
    核心代码：
    ```python
        quantizer.load_ft_param()
        quant_model = quantizer.quant_model
        quantizer.export_xmodel(deploy_check=False)
    ```
  
    常用`Graph`结构可视化:
    ```python
    for node in graph._dev_graph.nodes:
    	print(node.name, node.op.type)p
    
    ```
  



---



## BUG#1：Deepcopy问题



### 1.BUG Log:

```shell
...
...
...
[VAIQ_NOTE]: =>Loading quant model parameters.(quantize_result/param.pth)
Traceback (most recent call last):
  File "quant_fast_finetune.py", line 510, in <module>
    file_path=file_path)
  File "quant_fast_finetune.py", line 457, in quantization
    quantizer.load_ft_param()
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/apis.py", line 86, in load_ft_param
    self.processor.advanced_quant_setup()
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/base.py", line 142, in advanced_quant_setup
    self.adaquant = AdvancedQuantProcessor(self.quantizer)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/adaquant.py", line 96, in __init__
    self._float_model = copy.deepcopy(quantizer.quant_model)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/copy.py", line 180, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/copy.py", line 280, in _reconstruct
    state = deepcopy(state, memo)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/copy.py", line 150, in deepcopy
    y = copier(x, memo)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/copy.py", line 240, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/copy.py", line 161, in deepcopy
    y = copier(memo)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/torch/tensor.py", line 44, in __deepcopy__
    raise RuntimeError("Only Tensors created explicitly by the user "
RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment

```
### 2.定位
第二步的读取参数的函数`quantizer.load_ft_param()`中:

**解决：**

`deepopy`改成直接赋值。



---



## BUG#2：xmodel导出错误问题



### 1.BUG Log:

```shell
(vitis-ai-pytorch) Vitis-AI /workspace/models/AI-Model-Zoo/VAI-1.4-Model-Zoo-Code/Pytorch/yolov3-6 > python quant_fast_finetune.py  --quant_mode test --subset_len 1 --batch_size=8 --fast_finetune --deploy
...
...
...
[VAIQ_ERROR]: Failed convert graph 'Model' to xmodel.
Traceback (most recent call last):
  File "quant_fast_finetune.py", line 510, in <module>
    file_path=file_path)
  File "quant_fast_finetune.py", line 486, in quantization
    quantizer.export_xmodel(deploy_check=False)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/apis.py", line 98, in export_xmodel
    self.processor.export_xmodel(output_dir, deploy_check)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/base.py", line 165, in export_xmodel
    dump_xmodel(output_dir, deploy_check, self._lstm_app)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/base.py", line 212, in dump_xmodel
    raise e
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/base.py", line 208, in dump_xmodel
    output_file_name=os.path.join(output_dir, depoly_info.dev_graph.name))
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/nndct_shared/compile/xir_compiler.py", line 139, in do_compile
    raise AddXopError(unknown_op_types) 
nndct_shared.utils.exception.AddXopError: Please support ops({'shape(Model::Model/Detect/1920)',
'shape(Model::Model/Detect/1872)', 'cast(Model::Model/Upsample/1721)', 'shape(Model::Model/Detect/1896)',
'floor(Model::Model/Upsample/1746)', 'tensor(Model::Model/Upsample/1733)', 'tensor(Model::Model/Upsample/1551)',
'shape(Model::Model/Detect/1900)', 'shape(Model::Model/Upsample/1732)', 'tensor(Model::Model/Upsample/1568)',
'cast(Model::Model/Upsample/1573)', 'shape(Model::Model/Upsample/1715)', 'cast(Model::Model/Upsample/1580)',
'cast(Model::Model/Upsample/1745)', 'shape(Model::Model/Detect/1922)', 'cast(Model::Model/Upsample/1563)',
'int(Model::Model/Upsample/1730)', 'shape(Model::Model/Detect/1898)', 'tensor(Model::Model/Upsample/1716)',
'shape(Model::Model/Upsample/1550)', 'int(Model::Model/Upsample/1582)', 'cast(Model::Model/Upsample/1728)',
'floor(Model::Model/Upsample/1564)', 'floor(Model::Model/Upsample/1729)', 'int(Model::Model/Upsample/1565)',
'floor(Model::Model/Upsample/1581)', 'shape(Model::Model/Upsample/1567)', 'shape(Model::Model/Detect/1924)',
'shape(Model::Model/Detect/1874)', 'shape(Model::Model/Detect/1876)', 'cast(Model::Model/Upsample/1738)',
'int(Model::Model/Upsample/1747)', 'cast(Model::Model/Upsample/1556)'}) in xGraph.

```



### 2.定位

- 裁减后的模型经过某处理步骤X后进入`.xmodel`转换函数`xir_compiler.py: XirCompiler.do_compile()`，因为包含不可转换的算子，所以报错。

裁减后的模型处理前后如下：

**处理前：**

- 位于源代码: `pytorch_nndct.qproc.utils.py (line 263): get_deploy_graph_list(quant_model, nndct_graph):`

Uncommon如下Debug代码：
```python
  for node in g_optmizer._dev_graph.nodes:
      print(f"{node.name}, {node.op.type}, {node.out_tensors[0].layout}")
```

```c++
[VAIQ_NOTE]: =>Converting to xmodel ...
Model::input_0, input, Layout.NCHW
Model::1557, const, None
Model::1574, const, None
Model::1722, const, None
Model::1739, const, None
Model::Model/Conv/Conv2d[conv]/input.2, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.4, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.5, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.7, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.8, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.10, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.11, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/512, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/input.13, elemwise_add, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.14, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.16, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.17, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.19, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.20, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/568, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/input.22, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.23, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.25, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.26, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/606, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/input.28, elemwise_add, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.29, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.31, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.32, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.34, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.35, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/662, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/input.37, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.38, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.40, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.41, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/700, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/input.43, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.44, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.46, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.47, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/738, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/input.49, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.50, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.52, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.53, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/776, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/input.55, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/Conv2d[conv]/input.56, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/LeakyReLU[act]/input.58, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/Conv2d[conv]/input.59, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/LeakyReLU[act]/814, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/input.61, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/Conv2d[conv]/input.62, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/LeakyReLU[act]/input.64, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/Conv2d[conv]/input.65, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/LeakyReLU[act]/852, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/input.67, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/Conv2d[conv]/input.68, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/LeakyReLU[act]/input.70, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/Conv2d[conv]/input.71, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/LeakyReLU[act]/890, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/input.73, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/Conv2d[conv]/input.74, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/LeakyReLU[act]/input.76, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/Conv2d[conv]/input.77, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/LeakyReLU[act]/928, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/input.79, elemwise_add, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.80, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.82, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.83, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.85, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.86, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/984, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/input.88, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.89, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.91, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.92, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/1022, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/input.94, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.95, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.97, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.98, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/1060, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/input.100, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.101, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.103, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.104, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/1098, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/input.106, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/Conv2d[conv]/input.107, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv1]/LeakyReLU[act]/input.109, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/Conv2d[conv]/input.110, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/Conv[cv2]/LeakyReLU[act]/1136, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[4]/input.112, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/Conv2d[conv]/input.113, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv1]/LeakyReLU[act]/input.115, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/Conv2d[conv]/input.116, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/Conv[cv2]/LeakyReLU[act]/1174, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[5]/input.118, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/Conv2d[conv]/input.119, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv1]/LeakyReLU[act]/input.121, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/Conv2d[conv]/input.122, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/Conv[cv2]/LeakyReLU[act]/1212, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[6]/input./Bottleneck/Conv[cv1]/Co124, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/Conv2d[conv]/input.125, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv1]/LeakyReLU[act]/input.127, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/Conv2d[conv]/input.128, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/Conv[cv2]/LeakyReLU[act]/1250, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[7]/input.130, elemwise_add, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.131, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.133, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.134, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.136, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.137, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/1306, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/input.139, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.140, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.142, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.143, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/1344, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/input.145, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.146, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.148, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.149, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/1382, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[2]/input.151, elemwise_add, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.152, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.154, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.155, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/1420, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[3]/input.157, elemwise_add, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.158, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.160, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.161, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.163, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.164, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.166, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.167, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.169, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.170, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.172, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.173, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.175, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.177, leaky_relu, Layout.NCHW
Model::Model/Upsample/1550, shape, Layout.NCHW
Model::Model/Upsample/1551, tensor, Layout.NCHW
Model::Model/Upsample/1556, cast, Layout.NCHW
Model::Model/Upsample/1558, mul, Layout.NCHW
Model::Model/Upsample/1563, cast, Layout.NCHW
Model::Model/Upsample/1564, floor, Layout.NCHW
Model::Model/Upsample/1565, int, Layout.NCHW
Model::Model/Upsample/1567, shape, Layout.NCHW
Model::Model/Upsample/1568, tensor, Layout.NCHW
Model::Model/Upsample/1573, cast, Layout.NCHW
Model::Model/Upsample/1575, mul, Layout.NCHW
Model::Model/Upsample/1580, cast, Layout.NCHW
Model::Model/Upsample/1581, floor, Layout.NCHW
Model::Model/Upsample/1582, int, Layout.NCHW
Model::Model/Upsample/1584, resize, Layout.NCHW
Model::Model/Concat/input.178, concat, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.179, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.181, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.182, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.184, leaky_relu, Layout.NCHW
Model::Modelnv2d[conv]/input.185, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.187, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.188, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.190, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.191, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.193, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.194, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.218, leaky_relu, Layout.NCHW
Model::Model/Conv/Conv2d[conv]/input.196, conv2d, Layout.NCHW
Model::Model/Conv/LeakyReLU[act]/input.198, leaky_relu, Layout.NCHW
Model::Model/Upsample/1715, shape, Layout.NCHW
Model::Model/Upsample/1716, tensor, Layout.NCHW
Model::Model/Upsample/1721, cast, Layout.NCHW
Model::Model/Upsample/1723, mul, Layout.NCHW
Model::Model/Upsample/1728, cast, Layout.NCHW
Model::Model/Upsample/1729, floor, Layout.NCHW
Model::Model/Upsample/1730, int, Layout.NCHW
Model::Model/Upsample/1732, shape, Layout.NCHW
Model::Model/Upsample/1733, tensor, Layout.NCHW
Model::Model/Upsample/1738, cast, Layout.NCHW
Model::Model/Upsample/1740, mul, Layout.NCHW
Model::Model/Upsample/1745, cast, Layout.NCHW
Model::Model/Upsample/1746, floor, Layout.NCHW
Model::Model/Upsample/1747, int, Layout.NCHW
Model::Model/Upsample/1749, resize, Layout.NCHW
Model::Model/Concat/input.199, concat, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/Conv2d[conv]/input.200, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv1]/LeakyReLU[act]/input.202, leaky_relu, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/Conv2d[conv]/input.203, conv2d, Layout.NCHW
Model::Model/Bottleneck/Conv[cv2]/LeakyReLU[act]/input.205, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.206, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.208, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.209, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.211, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.212, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.214, leaky_relu, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.215, conv2d, Layout.NCHW
Model::Model/Sequential/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/input.217, leaky_relu, Layout.NCHW
Model::Model/Detect/Conv2d[m]/ModuleList[0]/1870, conv2d, Layout.NCHW
Model::Model/Detect/1872, shape, Layout.NCHW
Model::Model/Detect/1874, shape, Layout.NCHW
Model::Model/Detect/1876, shape, Layout.NCHW
Model::Model/Detect/1880, reshape, Layout.NCHW
Model::Model/Detect/1882, permute, Layout.NCHW
Model::Model/Detect/Conv2d[m]/ModuleList[1]/1894, conv2d, Layout.NCHW
Model::Model/Detect/1896, shape, Layout.NCHW
Model::Model/Detect/1898, shape, Layout.NCHW
Model::Model/Detect/1900, shape, Layout.NCHW
Model::Model/Detect/1904, reshape, Layout.NCHW
Model::Model/Detect/1906, permute, Layout.NCHW
Model::Model/Detect/Conv2d[m]/ModuleList[2]/1918, conv2d, Layout.NCHW
Model::Model/Detect/1920, shape, Layout.NCHW
Model::Model/Detect/1922, shape, Layout.NCHW
Model::Model/Detect/1924, shape, Layout.NCHW
Model::Model/Detect/1928, reshape, Layout.NCHW
Model::Model/Detect/1930, permute, Layout.NCHW
```
**处理后：**
- 位于源代码: `pytorch_shared.compile.xir_compiler.py (line 131): do_compile(...):`

添加如下Debug代码：
```python
    for node in compile_graph.nodes:
    	print(f'Node type: {node.op.type} | Node name: {node.name}') 
```
```
NODE TYPE: input
NODE TYPE: const
NODE TYPE: const
NODE TYPE: const
NODE TYPE: const
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: shape
NODE TYPE: tensor
NODE TYPE: cast
NODE TYPE: mul
NODE TYPE: cast
NODE TYPE: floor
NODE TYPE: int
NODE TYPE: shape
NODE TYPE: tensor
NODE TYPE: cast
NODE TYPE: mul
NODE TYPE: cast
NODE TYPE: floor
NODE TYPE: int
NODE TYPE: resize
NODE TYPE: concat
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: shape
NODE TYPE: tensor
NODE TYPE: cast
NODE TYPE: mul
NODE TYPE: cast
NODE TYPE: floor
NODE TYPE: int
NODE TYPE: shape
NODE TYPE: tensor
NODE TYPE: cast
NODE TYPE: mul
NODE TYPE: cast
NODE TYPE: floor
NODE TYPE: int
NODE TYPE: resize
NODE TYPE: concat
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: shape
NODE TYPE: shape
NODE TYPE: shape
NODE TYPE: reshape
NODE TYPE: permute
NODE TYPE: conv2d
NODE TYPE: shape
NODE TYPE: shape
NODE TYPE: shape
NODE TYPE: reshape
NODE TYPE: permute
NODE TYPE: conv2d
NODE TYPE: shape
NODE TYPE: shape
NODE TYPE: shape
NODE TYPE: reshape
NODE TYPE: permute
```
**(对照组) 可正常导出的量化模型处理后：**
```
NODE TYPE: input
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: elemwise_add
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: resize
NODE TYPE: concat
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: resize
NODE TYPE: concat
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: leaky_relu
NODE TYPE: conv2d
NODE TYPE: reshape
NODE TYPE: permute
NODE TYPE: conv2d
NODE TYPE: reshape
NODE TYPE: permute
NODE TYPE: conv2d
NODE TYPE: reshape
NODE TYPE: permute
```



### 3.排查方向

- 检查第二步中，模型从读取到`.xmodel`转换前经过了什么处理，以及是否经过了常规流程（对照组）中应该经过的处理。
- 已查出核心处理部分,位于源代码: `pytorch_nndct/qproc/utils.py (line 258): `：
```python
def get_deploy_graph_list(quant_model, nndct_graph):
  g_optmizer = DevGraphOptimizer(nndct_graph)
  g_optmizer.infer_tensor_layout()
  g_optmizer.strip_redundant_ops()
  
  # for node in g_optmizer._dev_graph.nodes:
  #   print(f"{node.name}, {node.op.type}, {node.out_tensors[0].layout}")
    
  # sync model data with dev graph
  connect_module_with_graph(quant_model, g_optmizer.frozen_graph, recover_param=False)
  update_nndct_blob_data(quant_model, g_optmizer.frozen_graph)
  connect_module_with_graph(quant_model, nndct_graph, recover_param=False)
  
  g_optmizer.constant_folding() # <--------------- [prob 1]
  if NndctOption.nndct_parse_debug.value >= 3:
    NndctDebugLogger.write(f"\nfrozen dev graph:\n{g_optmizer.frozen_graph}")
  
  deploy_graphs = g_optmizer.partition_by_quant_part() 
  
  return deploy_graphs
```
- 找到处理部分在该函数的如下部分 `[prob 1]`：
```python
  g_optmizer.constant_folding()
```
在经过finetune之后的模型无法正常被该函数处理（处理前后无任何变化），而正常模型经过处理后，部分`op`被清除，可以正常导出`.xmodel`。
- 继续查询该函数的定义,位于`nndct_shared/compile/deploy_optimizer.py (line 68)`:
```python
  def constant_folding(self):
    folding_nodes = set()
    for node in self._dev_graph.nodes:
      if node.in_quant_part is False: # <--------------- [prob 2]
          continue
      if hasattr(node.op, "AttrName"):
        for attr_name in node.op.attrs.keys():
          attr_val = node.node_attr(attr_name)
          if isinstance(attr_val, list):
            for i, val in enumerate(attr_val):
              attr_val[i] = self._materialize(node, val, folding_nodes)
          else:
            attr_val = self._materialize(node, attr_val, folding_nodes)
          if node.op.attrs[attr_name].type == list:
            attr_val = [attr_val]
          node.set_node_attr(attr_name, attr_val)
          
    if folding_nodes:   # <--------------- [prob 1]
      for node_name in folding_nodes:
        self._dev_graph.remove_node_forcely(self._dev_graph.node(node_name))
      
      self._dev_graph.reconnect_nodes()
```
其中`[prob 1]`在如下部分进行`op`清除处理:
```python
    if folding_nodes:   
      for node_name in folding_nodes:
        self._dev_graph.remove_node_forcely(self._dev_graph.node(node_name))
```
- 已经找到原因，位于其中`[prob 2]`,经过finetune之后的模型由于是通过`quantizer.load_ft_param()`读入，所以`node.in_quant_part`标志是`False`，无法通过校验，所以没法经过`op`移除处理。
```python
      if node.in_quant_part is False: # <--------------- [prob 2]
          continue
```



### 4.解决方案



#### 4.1. 操作顺序修改 ❌

将流程修改为：

```mermaid
graph LR
    B([float模型])-->A[剪枝]-->E([剪枝后的float模型])-->F[导出模型权重]-->G[读取模型权重]-->C[量化]-->D([剪枝后的int8模型])
```

**结果：** ❌

错误日志:

```shell
...
...
[VAIQ_NOTE]: =>Fast finetuning module parameters for better quantization accuracy...
  0%|                                                                                                                                             | 0/75 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "quant_fast_finetune.py", line 511, in <module>
    file_path=file_path)
  File "quant_fast_finetune.py", line 454, in quantization
    quantizer.fast_finetune(evaluate, (quant_model, ft_loader, register_buffers))
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/apis.py", line 82, in fast_finetune
    self.processor.finetune(run_fn, run_args)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/base.py", line 151, in finetune
    self.adaquant.finetune(run_fn, run_args)
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/pytorch_nndct/qproc/adaquant.py", line 470, in finetune
    del self.cached_output[pn_node.module]
KeyError: deephi_Input()
```

**原因分析：**

prune剪枝工具应该只允许量化过的模型作为输入。该方案确认无效。



---



#### 4.2. 强行抹除标志 ✔️

找出重新读取裁减后的模型之后没有带上`node.in_quant_part`标志的原因，强行把裁减后的模型的`node.in_quant_part`标志置`True`。

位于`quantizer_source_code/pytorch_nndct/qproc/adaquant.py (line 99)`:
```python
class AdvancedQuantProcessor(torch.nn.Module):
  r""" 
  This class re-implements the Adaquant technique proposed in the following paper.
  "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming, 
  arXiv:2006.10518, 2020."
  """
  def __init__(self, quantizer):
    super().__init__()
    self._quantizer = quantizer
    quantizer.load_param()
    self._graph = quantizer.graph
    self._quant_model = quantizer.quant_model
    self._float_model = copy.deepcopy(self._quant_model) # deepcopy有bug
    for mod in self._float_model.modules():
      if hasattr(mod, "node"):
        mod.node.in_quant_part = False # <--------- [prob1]
      
      if hasattr(mod, "quantizer"):
        mod.quantizer = None
        
    self._data_loader = None
    self._num_batches_tracked = 0
    self._cached_outputs = defaultdict(list)
    self._cached_output = defaultdict(list)
    self._handlers = []
    self._net_input_nodes = [node for node in self._graph.nodes if node.op.type == NNDCT_OP.INPUT]
    self._float_weights = defaultdict(list)
```
`[prob1]`处在重新读取模型权重时，模型被认为是float模型（虽然实际上已经是被量化过的int8了），此时`node.in_quant_part`标志被置`False`:

```python
      if hasattr(mod, "node"):
        mod.node.in_quant_part = False
```
如果将其改为`True`则可正常导出`.xmodel`。
```python
      if hasattr(mod, "node"):
        mod.node.in_quant_part = True
```

**结果：** ✔️

**解决问题：** `.xmodel`导出问题

**未解决问题：** 剪枝prune过程中所依赖的loss一直为0，导致导致prune无效。



---



#### 4.3. 模型参数修正 

为了解决剪枝prune过程中所依赖的loss一直为0，导致prune无效的问题，需要将读入的裁减后的模型参数补上原有的模型参数后重新走一边量化流程，使其带上 `node.in_quant_part`标志。

**注意点:**	

- 量化后的模型`quant_model`需要把原模型`hyp`和`register_buffers`参数带上以便能正常运行以计算loss:

    位于`./quant_fast_finetune.py (line 117) :`

    ```python
    compute_loss = ComputeLoss(model) if hasattr(model, 'hyp') else None
    ```

    如果没有`loss`参数，可使用`getattr()`和`setattr()`将原`float`模型中的`loss`参数复制过去:

    ```python
    setattr(quant_model,'hyp',getattr(model,'hyp'))
    ```

错误日志:

```shell
Traceback (most recent call last):
  File "quant_fast_finetune.py", line 511, in <module>
    file_path=file_path)
  File "quant_fast_finetune.py", line 442, in quantization
    register_buffers=register_buffers)
  File "quant_fast_finetune.py", line 117, in test
    compute_loss = ComputeLoss(model) if hasattr(model, 'hyp') else None
  File "/workspace/models/AI-Model-Zoo/VAI-1.4-Model-Zoo-Code/Pytorch/yolov3-6/utils/new/loss.py", line 107, in __init__
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
  File "/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 576, in __getattr__
    type(self).__name__, name))
AttributeError: 'Model' object has no attribute 'model'

```

原因是`quant_model`并没有`loss`计算环节中所需的模块`model`的`output`部分。

float模型`model`部分：

```shell
 
Model(
    (model): Sequential(
                        ...
                        ...
                        ...
                        (28): Detect(
                                	(m): ModuleList(
                                            (0): Conv2d(256, 18, kernel_size=(1, 1), stride=(1, 1)) <---------- [output_0]
                                            (1): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1)) <---------- [output_1]
                                            (2): Conv2d(1024, 18, kernel_size=(1, 1), stride=(1, 1)) <---------- [output_2]
                                        )
                        	)
    	)
    ...
    ...
)
```



量化后的模型`quant_model`对应部分：

```shell

Model(
	...
    (module_276): deephi_Conv2d(256, 18, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1]) <---------- [output_0]
    (module_277): DeephiTensorModule('size')
    (module_278): DeephiTensorModule('size')
    (module_279): DeephiTensorModule('size')
    (module_280): DeephiTensorModule('view')
    (module_281): DeephiTensorModule('permute')
    (module_282): DeephiTensorModule('contiguous')
    (module_283): deephi_Conv2d(512, 18, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1]) <---------- [output_1]
    (module_284): DeephiTensorModule('size')
    (module_285): DeephiTensorModule('size')
    (module_286): DeephiTensorModule('size')
    (module_287): DeephiTensorModule('view')
    (module_288): DeephiTensorModule('permute')
    (module_289): DeephiTensorModule('contiguous')
    (module_290): deephi_Conv2d(1024, 18, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1]) <---------- [output_2]
    (module_291): DeephiTensorModule('size')
    (module_292): DeephiTensorModule('size')
    (module_293): DeephiTensorModule('size')
    (module_294): DeephiTensorModule('view')
    (module_295): DeephiTensorModule('permute')
    (module_296): DeephiTensorModule('contiguous')
)
```

**问题：**

量化后的模型`quant_model`和原模型`model`结构几乎完全不一样,难以使用如下函数来计算loss：

```python
 compute_loss = ComputeLoss(model) if hasattr(model, 'hyp') else None
```

**解决方法：**

实际上`compute_loss`函数内部仅仅需要取得模型的一些附带的几个`Hyperparameter`，如下：

```python
#主模型
model.hyp
model.gr
#模型output部分
model.model[-1].na
model.model[-1].nc
model.model[-1].nl
model.model[-1].anchors
model.model[-1].stride
```

