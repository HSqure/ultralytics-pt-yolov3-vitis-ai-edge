import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18

from tqdm import tqdm

#----------------------------------------
import argparse
import time
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
from utils import *
from utils.parse_config import parse_data_cfg
from utils import torch_utils
from torch.utils.data import DataLoader
#----------------------------------------

ANCHORS = 9//3
#device = torch.device("cuda")
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="test_sample",
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="weights",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--subset_len',
    default=200,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=16,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
args, _ = parser.parse_known_args()

''' yolov3 val '''
def test(model, 
        data_cfg, 
        register_buffers,
        batch_size=16,
        subset_len=None,
        img_size=416,
        iou_thres=0.25,
        conf_thres=0.001,
        nms_thres=0.5,
        save_json=False):

    # if model is None:
    #     device = torch_utils.select_device()
    #     # Load weights
    #     model=torch.load(weights, map_location=device)
    # else:
    #     device = next(model.parameters()).device  # get model device

    # Configure run
    data_cfg_iner = parse_data_cfg(data_cfg)
    nc = int(data_cfg_iner['classes'])  # number of classes
    test_path = data_cfg_iner['valid']  # path to test images
    names = load_classes(data_cfg_iner['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size=img_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
    loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, fname='test_batch0.jpg')

        # Run model
        inf_out, train_out = model_with_post_precess(imgs, model, data_cfg, register_buffers)  # inference and training outputs

        # # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss_i, _ = compute_loss(train_out, targets, model)
            loss += loss_i.item()

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': coco91class[int(d[6])],
                        'bbox': [float3(x) for x in box[di]],
                        'score': float(d[4])
                    })

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    iou, bi = bbox_iou(pbox, tbox).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and bi not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(bi)

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return results
    return mp, mr, map, mf1, loss / len(dataloader)

def load_data(train=False,
              data_dir='',
              batch_size=16,
              subset_len=None,
              sample_method='random',
              distributed=False,
              **kwargs):

    #prepare data
    # random.seed(12345)
    traindir = data_dir + '/train'
    valdir = data_dir + '/val'
    # print('\n\n\n----------->'+os.getcwd()+'\n\n\n')

    train_sampler = None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
    size = 416
    resize = 256
    if train:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if subset_len:
            assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = torch.utils.data.Subset(
                dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **kwargs)
    else:
        dataset = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]))
        print(len(dataset))
        if subset_len:
            assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = torch.utils.data.Subset(
                dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return data_loader, train_sampler


"""Computes and stores the average and current value"""
class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions
            for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def model_with_post_precess(images, model, data_cfg, register_buffers):
    model = model.to(device)
    x=[]
    z=[]  # inference output
    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['valid']  # path to test images
    names = load_classes(data_cfg['names'])  # class names
    nl = ANCHORS
    grid = [torch.zeros(1)] * nl  # init grid
    stride = torch.tensor((8,16,32),dtype=float)  # strides computed during build
    anchor_grid = register_buffers['anchor_grid']

    for output in model(images):
        x.append(output) # update list
        
    # print(x)
    for i in range(nl):
        bs, _, ny, nx, no = x[i].shape
        if grid[i].shape[2:4] != x[i].shape[2:4]:
            grid[i] = _make_grid(nx, ny).to(x[i].device)
        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))

    return (torch.cat(z, 1), x)

''' read buffers '''
def model_info_read(model):
    for name, buf in model.named_buffers():
        if 'anchor_grid' in name:
            register_buffers={'anchor_grid':buf}
    return register_buffers

def evaluate(model, val_loader, data_cfg, register_buffers):
    model.eval()
    for iteraction, (images, labels) in tqdm(enumerate(val_loader), 
                                                total=len(val_loader)):
        images = images.to(device)
        # inference and get result 
        inf_out, train_out = model_with_post_precess(images, model, data_cfg, register_buffers)
        # out = model_with_post_precess(images, model, data_cfg)

def quantization(title='optimize',
                model_name='', 
                file_path=''): 

    data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    # Load weights
    # model.load_state_dict(torch.load(file_path))
    model = torch.load(file_path, map_location=device)
    # read buffers: anchor_grid
    register_buffers = model_info_read(model)

    # ========= visualization ========
    # ----------- 01.modules ----------
    # print('\n\n\n')
    # for idx, m in enumerate(model.modules()):
    #     print(idx,'->',m)
    # print('\n\n\n')
    # ----------- 02.named_children -----------
    # print('\n\n\n')
    # for name, module in model.named_children():
    #     print(name,': ',module)
    # print('\n\n\n')    
    # ----------- 03.named_modules -----------
    # print('\n\n\n')
    # for idx, m in enumerate(model.named_modules()):
    #     print(idx,'->',m)
    # print('\n\n\n')    
    # ----------------------------------------

    # ================================ Quantizer API ====================================
    # ===================================================================================
    input = torch.randn([batch_size, 3, 416, 416])
    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device)
        quant_model = quantizer.quant_model
    
    # # to get loss value after evaluation
    # loss_fn = torch.nn.CrossEntropyLoss().to(device)

    val_loader, _ = load_data(
        subset_len=subset_len,
        train=False,
        batch_size=batch_size,
        sample_method='random',
        data_dir=data_dir)

    evaluate(quant_model, 
             val_loader, 
             ata_cfg='data/pedestrian.data', 
             register_buffers=register_buffers)

    # # fast finetune model or load finetuned parameter before test
    # if finetune:
    #     ft_loader, _ = load_data(
    #         subset_len=1024,
    #         train=False,
    #         batch_size=batch_size,
    #         sample_method=None,
    #         data_dir=data_dir,
    #         model_name=model_name)
    #     if quant_mode == 'calib':
    #         quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
    #     elif quant_mode == 'test':
    #         quantizer.load_ft_param()
    
    # -------------------------------- yolov3 val ---------------------------------------
    # -----------------------------------------------------------------------------------
    if quant_mode == 'test':
        with torch.no_grad():
            mAP = test(model=quant_model,
                        data_cfg='data/pedestrian.data',
                        subset_len=subset_len,
                        register_buffers=register_buffers)


    # -----------------------------------------------------------------------------------

    # handle quantization result
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=False)
    # ===================================================================================

if __name__ == '__main__':

    model_name = 'yolov3-pedestrian'
    file_path = os.path.join(args.model_dir, model_name + '.pth')

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(
        title=title,
        model_name=model_name,
        file_path=file_path)

    print("-------- End of {} test ".format(model_name))

    