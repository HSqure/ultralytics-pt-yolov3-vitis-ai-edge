# ultralytics-pt-yolov3-vitis-ai-edge
This demo is only used for inference testing of `Vitis AI v1.4` and quantitative compilation of DPU. It is compatible with the training results of `ultramatics v9.5.0`  (it needs to use the model saving method of `Pytorch V1.4`)

### envirment
* `Yocto sdk 2020.1`
* `Vitis-AI V1.4 Docker-GPU`

### Attention
Training code using [ultralytis yolov3](https://github.com/ultralytics/yolov3) **BUT** the output part of the model weights in the code **NEED SOME MODIFICATION**.

**weights saving part**
```python
# Save model
if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
    ckpt = {'epoch': epoch,
            'best_fitness': best_fitness,
            'training_results': results_file.read_text(),
            'model': deepcopy(de_parallel(model)).half(),
            'ema': deepcopy(ema.ema).half(),
            'updates': ema.updates,
            'optimizer': optimizer.state_dict(),
            'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

    # Save last, best and delete
    torch.save(ckpt, last)
    torch.save(model, last_quant, _use_new_zipfile_serialization=False) # 用于量化的兼容版本

    if best_fitness == fi:
        torch.save(ckpt, best)
        torch.save(model, best_quant, _use_new_zipfile_serialization=False) # 用于量化的兼容版本
    if wandb_logger.wandb:
        if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
            wandb_logger.log_model(
                last.parent, opt, epoch, fi, best_model=best_fitness == fi)
    del ckpt

```
