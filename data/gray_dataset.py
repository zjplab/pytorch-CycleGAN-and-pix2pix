import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random

class GrayDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.

    This dataset is required by pix2pix-based gray model ('--model gray')
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (Gray) and
        the nubmer of channels for output image is 3 (RGB). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=3, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot)
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        assert(opt.input_nc == 1 and opt.output_nc == 3 and opt.direction == 'AtoB')
        self.transform = get_transform(self.opt, convert=False)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the gray image
            B (tensor) - - the corrsponding RGB
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        A=Image.fromarray(color.rgb2gray(np.array(AB)), mode="L")
        B=AB
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (w,h))
        A_transform = get_transform_one_channel(self.opt, transform_params, grayscale=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # AB_path is a list of file paths 
        return len(self.AB_paths)

def get_transform_one_channel(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5, ))]
    return transforms.Compose(transform_list)