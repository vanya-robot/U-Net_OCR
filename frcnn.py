from PIL import Image
import torchvision.transforms as transforms
import torch


def get_transform(train):
    transform = [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)]
    if train:
        transform.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform)


def frcnn_predict(directory, file_name, model):
    img = Image.open(directory + file_name).convert("RGB")
    img = get_transform(train=False)(img)
    # img = img.cuda() if GPU is used
    model.eval()
    prediction = model((img,))
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    coords = prediction[0]['boxes'][0]
    file, extention = file_name.split('.')[0], file_name.split('.')[1]
    img.crop((coords[0].item(), coords[1].item(), coords[2].item(), coords[3].item())).save(directory + file + '_crop'
                                                                                                               '.png')
    return directory + file + '_crop.png'
