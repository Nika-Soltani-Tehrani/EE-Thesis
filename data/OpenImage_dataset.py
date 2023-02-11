from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms


class OpenImageDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = opt.dataroot  # get the image directory
        self.paths = sorted(make_dataset(self.dir_AB, float("inf")))  # get image paths
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        # split AB image into A and B
        w, h = img.size
        
        if self.opt.isTrain:
            transform = transforms.Compose([
                transforms.RandomCrop(256, pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # apply the same transform to both A and B
        img = transform(img)

        return {'data': img, 'path': path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
