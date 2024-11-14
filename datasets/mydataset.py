from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image, ImageFile

# While images may be partially missing, they can be used to continue processing without interrupting the entire process.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')
        assert len(self) > 0

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



def build_dataset(args):
    total_dataset = ImageDataset(args.data_root, args.img_size)
    if args.valid_frac > 0:
        train_size = int((1 - args.valid_frac) * len(total_dataset))
        valid_size = len(total_dataset) - train_size
        train_set, valid_set = random_split(total_dataset, [train_size, valid_size],
                                              # generator=torch.Generator().manual_seed(random_split_seed)
                                              )
        print(
            f'training with dataset of {len(train_set)} samples and validating with randomly splitted {len(valid_set)} samples')
    else:
        valid_set = train_set = total_dataset
        print(f'training with shared training and valid dataset of {len(valid_set)} samples')

    return train_set, valid_set