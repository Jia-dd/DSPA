import os
import sys
import torch
import json
import time
import argparse
from torchvision import transforms, datasets
from tqdm import tqdm
from model.modified_resnet50 import resnet50
from dassl.utils import load_checkpoint


def main(args):
    print("*************** Arguments ***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print()

    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # transform
    if args.image_transform:
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])
    print("data_transform:")
    print(data_transform)

    # test dataset
    data_root = args.data_root
    test_set = args.test_set
    image_path = os.path.join(data_root, test_set)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    print("test image path: {}".format(image_path))

    # hyperparameter
    batch_size = args.batch_size
    print("batch_size: {}".format(batch_size))

    # number of workers
    numworker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process.'.format(numworker))

    # Test data loading
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=data_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=numworker)
    print("using {} images for test.".format(test_num))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    json_file = open(json_path, "r")
    classes = json.load(json_file)

    # network
    classification_model = resnet50(args.embedding_dim, num_classes=args.n_cls)
    classification_model = classification_model.to(device)

    # load model weights
    weights_path = args.weights_path
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    print(f"weights_path: {weights_path}")
    checkpoint = load_checkpoint(weights_path)
    state_dict = checkpoint["state_dict"]
    classification_model.load_state_dict(state_dict)

    class_correct = [0.] * 10
    class_total = [0.] * 10
    correct = 0.0
    total = 0.0

    # test
    classification_model.eval()
    start_time = time.time()
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            outputs = classification_model(test_images)
            predict = torch.max(outputs, dim=1)[1]

            result = (predict == test_labels).squeeze()

            for i, label in enumerate(test_labels):
                class_correct[label] += result[i].item()
                class_total[label] += 1
                correct += result[i].item()
                total += 1
    end_time = time.time()
    total_time = int(end_time - start_time)
    print('Total test time: {}s.'.format(total_time))

    for i in range(10):
        print("Acuracy of [{:25s}] is: {:2.2f}%".format(
            classes[str(i)], 100 * class_correct[i] / class_total[i]
        ))

    print("\nTotal accuracy is: {:2.2f}%".format(100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dim", type=int, default=512, help="embedding dim")
    parser.add_argument("--n_cls", type=int, help="number of class")
    parser.add_argument("--data_root", type=str, help="data root")
    parser.add_argument("--image_transform", type=bool, default=False, help="default without image transform")

    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    parser.add_argument("--batch_size", type=int, help="batch size")

    parser.add_argument("--weights_path", type=str, help="weights path")

    parser.add_argument("--test_set", type=str, help="test set")

    args = parser.parse_args()

    main(args)
