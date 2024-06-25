import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.trainer import Trainer
from testmodels import CNN, MVCNN, Model, Classifier
import torch.optim as optim
from losses import CELossTotal, CELossShift, CELoss

BACKBONE_NAME = 'DenseNet121'  # ResNeSt50 / ResNet50 / DenseNet121

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default=r'../mimic_cxr/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default=r'../mimic_cxr/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--root_dir', type=str, default=r'../mimic_cxr/', help='the path to the directory containing the data.')

    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')
    parser.add_argument('--test_batch_size', type=int, default=192, help='the number of samples for a batch')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', default='results/iu_xray/model_best.pth', type=str, help='whether to reload the checkpoints.')  # None  'results/iu_xray/model_best.pth'

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    # with open('target.txt', 'w') as file:
    #     for entry in test_dataloader.dataset.examples:
    #         name = entry.get('report', '')
    #         file.write(name + '\n' + '\n')

    # build model architecture
    backbone = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)  # github
    FC_FEATURES = 1024

    NUM_LABELS = 114
    NUM_CLASSES = 2

    LR = 1e-4  # Fastest LR
    WD = 5e-5  # Avoid overfitting with L2 regularization
    DROPOUT = 0.1  # Avoid overfitting
    NUM_EMBEDS = 256
    FWD_DIM = 256

    NUM_HEADS = 8
    NUM_LAYERS = 6

    edge_index = []
    with open('edges.txt', 'r') as f:
        # 逐行读取文件内容，并将每行的数据拆分成字符串列表
        for line in f:
            # 使用 split() 函数将字符串拆分为单个数字
            row_data = line.strip().split()
            # 将字符串转换为整数，并添加到 read_data 列表中
            edge_index.append([int(num) for num in row_data])
    cnn = CNN(backbone, BACKBONE_NAME)
    cnn = MVCNN(cnn)
    cls_model = Classifier(num_topics=NUM_LABELS, num_states=NUM_CLASSES, embed_dim=NUM_EMBEDS,
                           num_heads=NUM_HEADS, dropout=DROPOUT)
    model = Model(cls_model, cnn=cnn, num_layers=NUM_LAYERS, embed_dim=NUM_EMBEDS, fc_features=FC_FEATURES,
                  num_heads=NUM_HEADS, fwd_dim=FWD_DIM, dropout=DROPOUT, edge_index=edge_index,
                  max_len=args.max_seq_length, tokenizer=tokenizer)

    # get function handles of loss and metrics

    criterion = CELossShift(ignore_index=3)
    # criterion = CELossTotal(ignore_index=3)
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)  #
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    MILESTONES = [25, 50, 75]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
