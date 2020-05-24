import yaml
from typing import Optional
import os
from os import listdir
from pathlib import Path
import fileinput
from warnings import filterwarnings

import torch
from torch import nn
from torch.quantization import convert
import torch.nn.utils.prune as prune
from sklearn.metrics import roc_curve, auc
import torchvision

from fastai.vision import (
    np,
    jitter,
    TfmPixel,
    uniform_int,
    get_transforms,
    ImageDataBunch,
    Callable,
    Union,
    vision,
    callbacks,
    ClassificationInterpretation,
    cnn_learner,
    models,
    error_rate,
    accuracy,
    ImageList,
)

from fastai.metrics import FBeta, Precision, Recall, AUROC


def asklr():
    """Ask lower bound of the learning rate for further training.

    The lower bound is determined from the graph displayed by LR finder. The
    upper bound defaults to 1e-6

    Returns
    -------
    double
        Can be in exponential form. E.g. 1e-5
    """
    return input("Enter minimum learning rate: ")


def _cutout(x, n_holes: uniform_int = 1, length: uniform_int = 40):
    """Transforms the input with random patches of gaussian noise.

    Usually used for data augmentation.

    Parameters
    ----------
    x : np.array
        The image matrix
    n_holes : uniform_int, optional
        Number of patches to put on image, by default 1
    length : uniform_int, optional
        The patch size in pixels, by default 40

    Returns
    -------
    np.array
        The transformed images containing the random patches.
    """
    h, w = x.shape[1:]
    for _ in range(n_holes):
        h_y = np.random.randint(0, h)
        h_x = np.random.randint(0, w)
        y1 = int(np.clip(h_y - length / 2, 0, h))
        y2 = int(np.clip(h_y + length / 2, 0, h))
        x1 = int(np.clip(h_x - length / 2, 0, w))
        x2 = int(np.clip(h_x + length / 2, 0, w))
        x[:, y1:y2, x1:x2].normal_(mean=0.5, std=0.5).clamp_(0, 1)
    return x


def custom_split(m: nn.Module):
    """Splits the architecture at the returned layers.

    Used to selectively freeze the architecture for use in transfer learning.

    Parameters
    ----------
    m : nn.Module
        Then PyTorch module containing the model.

    Returns
    -------
    nn.sequential
        Specific layers where the model is split into the subgroups.
    """
    return m[0][1][10], m[1]


class Detector:
    """Base class which implements training, optimization, and augmentations.

    For every classification task, a separate object is instantiated with contains
    separate model and data loader defined in the derived class.

    Returns
    -------
    Detector
        The detector object without full functionality like data loading.
    """

    image_list = []

    def __init__(self):
        pass

    def transforms(self, noise=True, blur=True, basic=True):
        """Applies transformations on the images for the purpose of data
        augmentation.

        The available transforms are blurring through jitter, noisy patches,
        and basic transformations with include rotation, flipping, brightness,
        contrast, etc.

        Parameters
        ----------
        noise : bool, optional
            Add noisy patches to the images, by default True
        blur : bool, optional
            Blur the images with 0.3 probability, by default True
        basic : bool, optional
            Do basic transforms like rotation and flipping, by default True

        Returns
        -------
        fastai.transforms
            The transform object to be used during data augmentation
        """
        xtra_tfms = []
        if noise:
            cutout = TfmPixel(_cutout, order=20)
            xtra_tfms.append(cutout(n_holes=(10, 30), length=(10, 20), p=1))
        if blur:
            xtra_tfms.append(jitter(magnitude=0.03, p=0.3))
        if basic:
            self.tfms = get_transforms(
                flip_vert=True, max_rotate=22, xtra_tfms=xtra_tfms
            )
        else:
            self.tfms = get_transforms(xtra_tfms=xtra_tfms)
        return self.tfms

    def create_custom_body(
        self,
        arch: Callable,
        pretrained: bool = True,
        cut: Optional[Union[int, Callable]] = None,
    ):
        """Creates custom architecture from the MobileNet v2 architecture
        obtained from PyTorch.

        The head of the pretrained model is removed and a custom head is added.
        The modules are fused to make it ready for quantization.

        Parameters
        ----------
        arch : Callable
            Dummy argument to keep fastai utility functions happy.
        pretrained : bool, optional
            Dummy argument, by default True
        cut : Optional[Union[int, Callable]], optional
            Dummy argument, by default None

        Returns
        -------
        nn.Sequential
            The model with its head removed
        """
        model = torchvision.models.quantization.mobilenet_v2(
            pretrained=True, progress=True, quantize=False
        )
        model.fuse_model()
        mchild = list(model.children())
        return nn.Sequential(mchild[-2], mchild[0], mchild[-1])

    def createmodel(self, quantize=True):
        """Creates the model and attaches with the dataloader.

        By default it sets up the model for quantization aware training.

        Parameters
        ----------
        quantize : bool, optional
            To quantize or not, by default True
        """
        print("Creating model..")

        vision.learner.create_body = self.create_custom_body

        self.learn = cnn_learner(
            self.data,
            models.mobilenet_v2,
            pretrained=True,
            metrics=[error_rate, FBeta(beta=1), Precision(), Recall(), AUROC()],
            split_on=custom_split,
            model_dir=self.model_dir,
        )

        if quantize:
            self.learn.model[0].qconfig = torch.quantization.default_qat_qconfig
            self.learn.model = torch.quantization.prepare_qat(
                self.learn.model, inplace=True
            )

    def findlr(self):
        """Finds the optimum learning rate and displays the graph

        Multiple batches are fed to the model and loss is plotted against the
        learning rates during the brief mock training.
        """
        self.learn.lr_find()
        self.learn.recorder.plot()

    def train(self, epochs=10, firstrun=False, min_lr=None, interpret=False):
        """Trains the model and saves the best model

        Use One cycle scheduler for learning rate. The model with least test
        loss is saved as recentbest.pth in model_dir.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs, by default 10
        firstrun : bool, optional
            If the frozen layers exist are unfreezed, by default False
        min_lr : double, optional
            The minimum learning rate for differential LRs, by default None
        interpret : bool, optional
            Show top losses after training, by default False
        """
        print("Training..")

        self.learn.fit_one_cycle(
            epochs,
            max_lr=slice(min_lr, 1e-3),
            callbacks=[
                callbacks.SaveModelCallback(self.learn, name="recentbest"),
                callbacks.ReduceLROnPlateauCallback(self.learn, patience=1),
            ],
        )

        self.learn.recorder.plot_losses()

        if interpret:
            self.interpretation = ClassificationInterpretation.from_learner(self.learn)
            print(self.interpretation.most_confused(min_val=2))

        if firstrun:
            self.learn.save("firstrun")
            self.learn.unfreeze()
            self.learn.fit_one_cycle(1)
            self.findlr()

    def getroc(self):
        """Compute ROC-AOC score

        Returns
        -------
        double
            ROC-AUC score
        """
        preds, y = self.learn.get_preds()
        probs = np.exp(preds[:, 1])
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y, probs, pos_label=1)
        # Compute ROC area
        self.roc_auc = auc(fpr, tpr)
        print("ROC area is {0}".format(self.roc_auc))
        return self.roc_auc

    def prunemodel(self, amount=0.5):
        """Prune the model weights

        Global unstructured pruning is done across all the weights in the model.

        Parameters
        ----------
        amount : float, optional
            The extent of pruning <= 1, by default 0.5
        """
        print("Pruning model..")

        self.prun_param = [
            (x, "weight") for x in self.learn.model.modules() if (hasattr(x, "weight"))
        ]
        prune.global_unstructured(
            self.prun_param, pruning_method=prune.L1Unstructured, amount=amount,
        )

    def getsparsity(self):
        """Returns the amount of sparsity in the model

        Finds the number of elements in whole model which are zero and returns
        the percentage based on total number of weights.

        Returns
        -------
        double
            Sparsity
        """
        zero = 0
        nan = 0
        for x in self.prun_param:
            zero += torch.sum(x[0].weight == 0)
        for x in self.prun_param:
            nan += x[0].weight.nelement()
        self.sparsity = 100.0 * float(zero) / float(nan)
        print("Global sparsity: {:.2f}%".format(self.sparsity))
        return self.sparsity

    def getaccuracy(self):
        """Returns the accuracy of the model on the test data

        Returns
        -------
        double
            Accuracy
        """
        self.accuracy = accuracy(*(self.learn.get_preds()))
        return self.accuracy

    def finishpruning(self):
        """Finalizes pruning by removing the parallel set of original unpruned
        weights.

        NB: Changes the model names different from original. Must be kept in
        mind while loading weights.
        """
        for x in self.prun_param:
            prune.remove(x[0], "weight")

    def loadmodel(self, path=None):
        """Loads the model weights from the file.

        If the model and file layer names don't match, no warning will be
        raised.

        Parameters
        ----------
        path : Path(), optional
            The location of the file, by default self.model_dir/"recentbest"
        """
        print("Loading model..")

        if not path:
            path = self.model_dir / "recentbest"
        self.learn.load(path, strict=False)

    def quantize(self):
        """Quantize the model and save as self.quantizedmodel.

        Weights are quantized to 8-bit precision and the model is saved as
        quantizemodel.pth in self.model_dir directory.

        Returns
        -------
        [type]
            [description]
        """
        print("Quantizing model..")

        self.learn.model.cpu()
        self.quantizedmodel = convert(self.learn.model, inplace=False)
        torch.save(self.quantizedmodel, self.model_dir / "quantizedmodel.pth")
        self.learn.model.to("cuda")
        return self.quantizedmodel

    def trace(self):
        """Convert the quantized model to TorchScript format

        Used during the app building process. The model is stored as mobile.pt
        and the operator list as mobile.yaml in self.model_dir.
        """
        print("Tracing model for mobile deployment..")

        traced_script_module = torch.jit.trace(
            self.quantizedmodel, torch.rand((1, 3, 224, 224))
        )
        traced_script_module.save(self.model_dir / "mobile.pt")

        model_pt = torch.jit.load(self.model_dir / "mobile.pt")
        ops = torch.jit.export_opnames(model_pt)
        with open(self.model_dir / "mobile.yaml", "w") as output:
            yaml.dump(ops, output)


class PestDetector(Detector):
    """Derived class for disease detection from the images.

    Implements data loading and defines Path variables.

    Parameters
    ----------
    Detector : Detector
        The base class which implements core functionality like training.
    """

    def __init__(self, root, model_dir=None):
        """Set the root and model_dir directory

        The arguments can be path object or just plain strings.

        Parameters
        ----------
        root : Path() or str
            The directory containing 'images' directory along with train, valid
            , test file lists.
        model_dir : Path() or str, optional
            The location to store the models, by default None
        """
        super().__init__()
        self.root, self.model_dir = Path(root), Path(model_dir)

    def getdata(self, bs=32, num_workers=16, noise=True, blur=True, basic=True):
        """Returns the dataloader to be used during training.

        The returned data is normalized and the image are resized to 224x224px.

        Parameters
        ----------
        bs : int, optional
            Batch size, by default 32
        num_workers : int, optional
            Num of process used for fetching data, by default 16
        noise : bool, optional
            Whether to add noisy patches as augmentation, by default True
        blur : bool, optional
            Whether to add blur augmentation, by default True
        basic : bool, optional
            Whether to do basic augmentation like rotation, flipping, etc.
            , by default True

        Returns
        -------
        dataloader
            Dataloader with random sampling enabled.
        """
        print("Going through the data..")

        filenames = ["test", "val", "train"]
        filenames = [self.root / (x + ".txt") for x in filenames]
        with open(self.root / "list.txt", "w") as fout:
            fin = fileinput.input(filenames)
            for line in fin:
                fout.write(line)
            fin.close()
        self.data = (
            (
                ImageList.from_csv(
                    path=self.root, folder="images", csv_name="list.txt", delimiter=" "
                )
            )
            .split_by_idx(list(range(22169)))
            .label_from_df()
            .transform(self.transforms(noise, blur, basic), size=224)
            .databunch(bs=bs, num_workers=num_workers)
        ).normalize()
        return self.data
