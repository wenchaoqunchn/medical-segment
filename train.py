import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import warnings
import segmentation_models_pytorch as smp
import cv2
import torchvision

warnings.filterwarnings("ignore")
path = ""
global restext
malignant = path
# %%
count_malignant = len(os.listdir(malignant))
# %%

# %%
# Get the list of all the images

malignant_cases = glob.glob(malignant + "/*")

# An empty list. We will insert the data into this list in (img_path, label) format
exclude = []
train_data_mask = []
train_data_img = []
# Go through all the malignant cases. The label for these cases will be 1
for img in malignant_cases:

    if img.endswith("_mask.png"):
        train_data_mask.append(img)
    elif img.endswith("_mask_1.png") or img.endswith("_mask_2.png"):
        exclude.append(img)
    else:
        train_data_img.append(img)

# %%
train_data_img = sorted(train_data_img)
train_data_mask = sorted(train_data_mask)
# %%
len(train_data_img), len(train_data_mask), len(exclude)
# %%
train_data_img[100]
# %%
train_data_mask[100]
# %%
images = []
masks = []
size_x = 64
size_y = 64

for every_img_path in train_data_img:
    img = cv2.imread(every_img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size_y, size_x))
    images.append(img)

for every_mask_path in train_data_mask:
    mask = cv2.imread(every_mask_path, 0)
    mask = cv2.resize(mask, (size_y, size_x))
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)
# %%
x = images / 255
y = masks / 255
# %%
import random
from skimage.io import imshow

########## Displaying random image from X_train and Y_train #########
random_num = random.randint(0, 210)
imshow(x[random_num])
plt.show()
imshow(y[random_num])
plt.show()

test_img = x[random_num]
test_img2 = y[random_num]
# print(test_img.min(), test_img.max())
# print(test_img.shape)
#
# print(test_img2.min(), test_img2.max())
# print(test_img2.shape)
# %%
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

mask_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

# %%
def adjust_data(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1.0
    mask[mask <= 0.5] = 0.0

    return (img, mask)

# %%
class MyDataset(Dataset):
    def __init__(
        self,
        images=images,
        masks=masks,
        adjust_data=adjust_data,
        image_transform=image_transform,
        mask_transform=mask_transform,
    ):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.adjust_data = adjust_data
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = image_path
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = mask_path
        #         mask =cv2.imread(mask_path, 0)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # image, mask = self.adjust_data(image, mask)

        if self.image_transform:
            image = self.image_transform(image).float()

        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

# %%
index = 200
data = MyDataset()[index]
data[0].shape, data[1].shape
# %%
plt.imshow(data[1].permute(1, 2, 0).squeeze(-1).numpy())
# %%
len(images)

# %%
def prepare_loaders(
    images=images,
    masks=masks,
    train_num=int(210 * 0.6),
    valid_num=int(210 * 0.8),
    bs=32,
):
    train_images = images[:170]
    valid_images = images[170:200]
    test_images = images[200:]

    train_masks = masks[:170]
    valid_masks = masks[170:200]
    test_masks = masks[200:]

    train_ds = MyDataset(images=train_images, masks=train_masks)
    valid_ds = MyDataset(images=valid_images, masks=valid_masks)
    test_ds = MyDataset(images=test_images, masks=test_masks)

    train_loader = DataLoader(train_ds, batch_size=bs, num_workers=0, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=bs, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=0, shuffle=True)

    print("DataLoader Completed")

    return train_loader, valid_loader, test_loader

# %%
train_loader, valid_loader, test_loader = prepare_loaders(
    images=images,
    masks=masks,
    train_num=int(210 * 0.65),
    valid_num=int(210 * 0.85),
    bs=32,
)
# %%
len(train_loader)
# %%
# data = next(iter(train_loader))
# data[0].shape, data[1].shape
# %%
device = torch.device("cpu")
device

# %%
class Block(nn.Module):
    def __init__(self, inputs=3, middles=64, outs=64):
        super().__init__()
        # self.device = device
        # self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(inputs, middles, 3, 1, 1)
        self.conv2 = nn.Conv2d(middles, outs, 3, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outs)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn(self.conv2(x)))
        # e1 = x
        # x = self.pool(x)

        return self.pool(x), x
        # self.pool(x): [bs, out, h*.5, w*.5]
        # x: [bs, out, h, w]

        # return x, e1
        # x: [bs, out, h*.5, w*.5]
        # e1: [bs, out, h, w]

# %%
# import torch.nn as nn
# Tencho's Model

class UNet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # self.device = device
        # self.dropout = nn.Dropout(dropout)

        self.en1 = Block(3, 64, 64)
        self.en2 = Block(64, 128, 128)
        self.en3 = Block(128, 256, 256)
        self.en4 = Block(256, 512, 512)
        self.en5 = Block(512, 1024, 512)

        self.upsample4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.de4 = Block(1024, 512, 256)

        self.upsample3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.de3 = Block(512, 256, 128)

        self.upsample2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.de2 = Block(256, 128, 64)

        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.de1 = Block(128, 64, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [bs, 3, 256, 256]

        x, e1 = self.en1(x)
        # x: [bs, 64, 128, 128]
        # e1: [bs, 64, 256, 256]

        x, e2 = self.en2(x)
        # x: [bs, 128, 64, 64]
        # e2: [bs, 128, 128, 128]

        x, e3 = self.en3(x)
        # x: [bs, 256, 32, 32]
        # e3: [bs, 256, 64, 64]

        x, e4 = self.en4(x)
        # x: [bs, 512, 16, 16]
        # e4: [bs, 512, 32, 32]

        _, x = self.en5(x)
        # x: [bs, 512, 16, 16]

        x = self.upsample4(x)
        # x: [bs, 512, 32, 32]
        x = torch.cat([x, e4], dim=1)
        # x: [bs, 1024, 32, 32]
        _, x = self.de4(x)
        # x: [bs, 256, 32, 32]

        x = self.upsample3(x)
        # x: [bs, 256, 64, 64]
        x = torch.cat([x, e3], dim=1)
        # x: [bs, 512, 64, 64]
        _, x = self.de3(x)
        # x: [bs, 128, 64, 64]

        x = self.upsample2(x)
        # x: [bs, 128, 128, 128]
        x = torch.cat([x, e2], dim=1)
        # x: [bs, 256, 128, 128]
        _, x = self.de2(x)
        # x: [bs, 64, 128, 128]

        x = self.upsample1(x)
        # x: [bs, 64, 256, 256]
        x = torch.cat([x, e1], dim=1)
        # x: [bs, 128, 256,256, 256
        _, x = self.de1(x)
        # x: [bs, 64, 256, 256]

        x = self.conv_last(x)
        # x: [bs, 1, 256, 256]

        # x = x.squeeze(1)
        return x

# %%
model = UNet().to(device)
# %%
# loss_fn = nn.BCELoss().to(device)
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
optimizer = torch.optim.Adam(
    model.parameters(),
)
# %%
# Scheduler
from torch.optim import lr_scheduler

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

# %%
def train_one_epoch(
    model=model,
    dataloader=train_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=None,
    device=device,
    epoch=1,
):
    model.train()
    train_loss, dataset_size = 0, 0

    bar = tqdm(dataloader, total=len(dataloader))
    tp_l, fp_l, fn_l, tn_l = [], [], [], []

    for data in bar:
        x = data[0].to(device)
        y_true = data[1].to(device)
        y_pred = model(x)

        loss = loss_fn(y_pred, y_true)

        pred_mask = (y_pred > 0.5).float()
        btp, bfp, bfn, btn = smp.metrics.get_stats(
            pred_mask.long(), y_true.long(), mode="binary"
        )

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # 실시간 train_epoch_loss
        # why? tqdm 간지를 위해
        bs = x.shape[0]
        dataset_size += bs
        train_loss += loss.item() * bs
        train_epoch_loss = train_loss / dataset_size

        tp_l.append(btp)
        fp_l.append(bfp)
        fn_l.append(bfn)
        tn_l.append(btn)

        tp = torch.cat(tp_l)
        fp = torch.cat(fp_l)
        fn = torch.cat(fn_l)
        tn = torch.cat(tn_l)

        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")

        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        bar.set_description(
            f"EP:{epoch} | TL:{train_epoch_loss:.3e} | ACC: {accuracy:.2f} | F1: {f1_score:.3f} "
        )

    metrics = dict()

    metrics["f1_score"] = f1_score.detach().cpu().item()
    metrics["accuracy"] = accuracy.detach().cpu().item()

    metrics["recall"] = recall.detach().cpu().item()
    metrics["precision"] = precision.detach().cpu().item()

    metrics["dataset_iou"] = dataset_iou.detach().cpu().item()
    metrics["per_iou"] = per_image_iou.detach().cpu().item()

    metrics["loss"] = train_epoch_loss

    return metrics

# %%
@torch.no_grad()
def valid_one_epoch(
    model=model, dataloader=valid_loader, loss_fn=loss_fn, device=device, epoch=0
):
    model.eval()
    valid_loss, dataset_size = 0, 0
    bar = tqdm(dataloader, total=len(dataloader))
    tp_l, fp_l, fn_l, tn_l = [], [], [], []

    with torch.no_grad():
        for data in bar:
            x = data[0].to(device)
            y_true = data[1].to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y_true)

            pred_mask = (y_pred > 0.5).float()
            btp, bfp, bfn, btn = smp.metrics.get_stats(
                pred_mask.long(), y_true.long(), mode="binary"
            )

            tp_l.append(btp)
            fp_l.append(bfp)
            fn_l.append(bfn)
            tn_l.append(btn)

            tp = torch.cat(tp_l)
            fp = torch.cat(fp_l)
            fn = torch.cat(fn_l)
            tn = torch.cat(tn_l)

            recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")

            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

            # per image IoU means that we first calculate IoU score for each image
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )

            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset
            # with "empty" images (images without target class) a large gap could be observed.
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            # 실시간 valid_epoch_loss
            bs = x.shape[0]
            dataset_size += bs
            valid_loss += loss.item() * bs
            valid_epoch_loss = valid_loss / dataset_size

            bar.set_description(
                f"EP:{epoch} | VL:{valid_epoch_loss:.3e} | ACC: {accuracy:.2f} | F1: {f1_score:.3f} "
            )

    metrics = dict()

    metrics["f1_score"] = f1_score.detach().cpu().item()
    metrics["accuracy"] = accuracy.detach().cpu().item()

    metrics["recall"] = recall.detach().cpu().item()
    metrics["precision"] = precision.detach().cpu().item()

    metrics["dataset_iou"] = dataset_iou.detach().cpu().item()
    metrics["per_iou"] = per_image_iou.detach().cpu().item()

    metrics["loss"] = valid_epoch_loss

    return metrics

# %%
import copy

def run_training(
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    device=device,
    n_epochs=100,
    early_stop=20,
    scheduler=None,
):
    global restext
    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    lowest_epoch, lowest_loss = np.inf, np.inf

    train_history, valid_history = [], []
    train_recalls, valid_recalls = [], []

    train_pres, valid_pres = [], []
    train_accs, valid_accs = [], []

    train_f1s, valid_f1s = [], []

    train_per_ious, valid_per_ious = [], []
    train_dataset_ious, valid_dataset_ious = [], []

    print_iter = 5

    best_score = 0
    best_model = "None"

    for epoch in range(0, n_epochs):
        gc.collect()

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch + 1,
        )

        valid_metrics = valid_one_epoch(
            model, dataloader=valid_loader, device=device, epoch=epoch + 1
        )

        # 줍줍 : Joob-Joob, which means 'get-get'
        train_history += [train_metrics["loss"]]
        valid_history += [valid_metrics["loss"]]

        train_recalls += [train_metrics["recall"]]
        valid_recalls += [valid_metrics["recall"]]

        train_pres += [train_metrics["precision"]]
        valid_pres += [valid_metrics["precision"]]

        train_accs += [train_metrics["accuracy"]]
        valid_accs += [valid_metrics["accuracy"]]

        train_f1s += [train_metrics["f1_score"]]
        valid_f1s += [valid_metrics["f1_score"]]

        train_per_ious += [train_metrics["per_iou"]]
        valid_per_ious += [valid_metrics["per_iou"]]

        train_dataset_ious += [train_metrics["dataset_iou"]]
        valid_dataset_ious += [valid_metrics["dataset_iou"]]

        print()
        if (epoch + 1) % print_iter == 0:
            print(
                f"Epoch:{epoch + 1}|TL:{train_metrics['loss']:.3e}|VL:{valid_metrics['loss']:.3e}|F1:{valid_metrics['f1_score']:.4f}|Dataset IOU:{valid_metrics['dataset_iou']:.4f}|Per Img IOU:{valid_metrics['per_iou']:.4f}|"
            )
            print()
            restext += f"Epoch:{epoch + 1}|TL:{train_metrics['loss']:.3e}|VL:{valid_metrics['loss']:.3e}|F1:{valid_metrics['f1_score']:.4f}|Dataset IOU:{valid_metrics['dataset_iou']:.4f}|Per Img IOU:{valid_metrics['per_iou']:.4f}|\n"
            # textWidget.setPlainText(restext)

        if best_score < valid_metrics["f1_score"]:
            print(
                f"Validation F1 Improved({best_score:.2f}) --> ({valid_metrics['f1_score']:.2f})"
            )
            restext += f"Validation F1 Improved({best_score:.2f}) --> ({valid_metrics['f1_score']:.2f})"
            # textWidget.setPlainText(restext)
            best_model = model
            best_score = valid_metrics["f1_score"]
            best_model = copy.deepcopy(model.state_dict())
            PATH2 = f"model_f1.bin"
            torch.save(model.state_dict(), PATH2)
            print(f"Better_F1_Model Saved")
            restext += f"Better_F1_Model Saved\n"
            # textWidget.setPlainText(restext)
            print()

        if valid_metrics["loss"] < lowest_loss:
            print(
                f"Validation Loss Improved({lowest_loss:.4e}) --> ({valid_metrics['loss']:.4e})"
            )
            restext += f"Validation Loss Improved({lowest_loss:.4e}) --> ({valid_metrics['loss']:.4e})"
            # textWidget.setPlainText(restext)
            lowest_loss = valid_metrics["loss"]
            lowest_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"model.bin"
            torch.save(model.state_dict(), PATH)
            print(f"Better Loss Model Saved")
            restext += f"Better Loss Model Saved\n"
            # textWidget.setPlainText(restext)
            print()
        else:
            if early_stop > 0 and lowest_epoch + early_stop < epoch + 1:
                print("There is no improvement")
                restext += "There is no improvement"
                # textWidget.setPlainText(restext)
                break

    print()
    restext += "\n"
    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    restext += "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
        time_elapsed // 3600,
        (time_elapsed % 3600) // 60,
        (time_elapsed % 3600) % 60,
    )
    # textWidget.setPlainText(restext)
    print("Best Loss: %.4e at %d th Epoch" % (lowest_loss, lowest_epoch))
    restext += "Best Loss: %.4e at %d th Epoch" % (lowest_loss, lowest_epoch)
    # textWidget.setPlainText(restext)

    # load best model weights
    # model.load_state_dict(best_model_wts)
    model.load_state_dict(torch.load("./model_f1.bin"))

    result = dict()
    result["Train Loss"] = train_history
    result["Valid Loss"] = valid_history

    result["Train Recall"] = train_recalls
    result["Valid Recall"] = valid_recalls

    result["Train Precision"] = train_pres
    result["Valid Precision"] = valid_pres

    result["Train Accuracy"] = train_accs
    result["Valid Accuracy"] = valid_accs

    result["Train F1 Score"] = train_f1s
    result["Valid F1 Score"] = valid_f1s

    result["Train per Image IOU"] = train_per_ious
    result["Valid per Image IOU"] = valid_per_ious

    result["Train Dataset IOU"] = train_dataset_ious
    result["Valid Dataset IOU"] = valid_dataset_ious

    return model, result

# %%
model, result = run_training(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    n_epochs=200,
)
"""
# %%
type(result)
# %%
import json

# Serializing json
json_object = json.dumps(result, indent=4)
# %%
with open("unet-breastCancer-40epoch.json", "w") as outfile:
    json.dump(json_object, outfile)
# %%
## Train/Valid Loss History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid Loss History", fontsize=20)
plt.plot(
    range(0, len(result['Train Loss'][plot_from:])),
    result['Train Loss'][plot_from:],
    label='Train Loss'
)

plt.plot(
    range(0, len(result['Valid Loss'][plot_from:])),
    result['Valid Loss'][plot_from:],
    label='Valid Loss'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
plt.show()
# %%
## Train/Valid Accuracy History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid Accuracy History", fontsize=20)
plt.plot(
    range(0, len(result['Train Accuracy'][plot_from:])),
    result['Train Accuracy'][plot_from:],
    label='Train Accuracy'
)

plt.plot(
    range(0, len(result['Valid Accuracy'][plot_from:])),
    result['Valid Accuracy'][plot_from:],
    label='Valid Accuracy'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
# %%
## Train/Valid Recall History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid Recall History", fontsize=20)
plt.plot(
    range(0, len(result['Train Recall'][plot_from:])),
    result['Train Recall'][plot_from:],
    label='Train Recall'
)

plt.plot(
    range(0, len(result['Valid Recall'][plot_from:])),
    result['Valid Recall'][plot_from:],
    label='Valid Recall'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
# %%
## Train/Valid Precision History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid Precision History", fontsize=20)
plt.plot(
    range(0, len(result['Train Precision'][plot_from:])),
    result['Train Precision'][plot_from:],
    label='Train Precision'
)

plt.plot(
    range(0, len(result['Valid Precision'][plot_from:])),
    result['Valid Precision'][plot_from:],
    label='Valid Precision'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
# %%
## Train/Valid F1 History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid F1 Score History", fontsize=20)
plt.plot(
    range(0, len(result['Train F1 Score'][plot_from:])),
    result['Train F1 Score'][plot_from:],
    label='Train F1 Score'
)

plt.plot(
    range(0, len(result['Valid F1 Score'][plot_from:])),
    result['Valid F1 Score'][plot_from:],
    label='Valid F1 Score'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
# %%
## Train/Valid Per Image IOU History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid per Image IOU History", fontsize=20)
plt.plot(
    range(0, len(result['Train per Image IOU'][plot_from:])),
    result['Train per Image IOU'][plot_from:],
    label='Train per Image IOU'
)

plt.plot(
    range(0, len(result['Valid per Image IOU'][plot_from:])),
    result['Valid per Image IOU'][plot_from:],
    label='Valid per Image IOU'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
# %%
## Train/Valid Dataset IOU History
plot_from = 0
plt.figure(figsize=(20, 10))
plt.title("Train/Valid Dataset IOU History", fontsize=20)
plt.plot(
    range(0, len(result['Train Dataset IOU'][plot_from:])),
    result['Train Dataset IOU'][plot_from:],
    label='Train Dataset IOU'
)

plt.plot(
    range(0, len(result['Valid Dataset IOU'][plot_from:])),
    result['Valid Dataset IOU'][plot_from:],
    label='Valid Dataset IOU'
)

plt.legend()
# plt.yscale('log')
plt.grid(True)
"""
