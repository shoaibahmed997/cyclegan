import torch
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

from cyclegan import G_AB
img = './test.jpg'

transforms_ = transforms.Compose([
    transforms.Resize(int(256 * 1.12), Image.BICUBIC),
    transforms.RandomCrop((256, 256)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

G_AB.load_state_dict(torch.load("saved_models/horses2zebras/horse2zebra_pretrained.pth"))

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    # imgs = next(iter(val_dataloader))
    real_A = transforms_(img)
    G_AB.eval()
    
    # real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    # real_B = Variable(imgs["B"].type(Tensor))

    # Arange images along x-axis
    # real_A = make_grid(real_A, nrow=5, normalize=True)
    # real_B = make_grid(real_B, nrow=5, normalize=True)
    # fake_A = make_grid(fake_A, nrow=5, normalize=True)
    # fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    # image_grid = torch.cat((fake_B, real_B, fake_A), 1)
    save_image(fake_B, "images/horses2zebras/inference.jpg")

sample_images()