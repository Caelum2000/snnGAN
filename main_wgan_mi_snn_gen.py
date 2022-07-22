import torch
from torch import nn
import global_v as glv
from datasets import load_datasets
from model_snn import GAN_snn
from model_snn import DGAN_snn
from model_snn import WGAN_snn
from network_parser import Parse
import torchvision
from torchvision.utils import save_image
import argparse
import os
import logging
from metrics import clean_fid
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'n_reset'):
            m.n_reset()


def update_D(X, Z, net_D, net_G, trainer_D):
    # X.shape = (batch_size, 784)
    batch_size = X.shape[0]
    trainer_D.zero_grad()
    real_Y = net_D(X, is_imgs=True)  # real_Y.shape = (n_steps, batch_size, 1)
    # print(real_Y.shape)
    if not glv.network_config['is_mem']:
        real_Y = torch.sum(real_Y, dim=0) / glv.network_config[
            'n_steps']  # real_Y.shape = (batch_size,1)
    real_Y = real_Y.mean()
    real_Y.backward()
    reset_net(net_D)
    # print(real_Y.shape)
    fake_X, _ = net_G(Z)  # fake_X.shape = (n_steps, batch_size, 784)
    n_step = fake_X.shape[0]
    if glv.network_config['net_D_direct_input']:
        fake_X = fake_X.reshape((n_step, batch_size, 1, 28, 28))
    else:
        fake_X = fake_X.reshape((n_step, batch_size, 784))
    fake_Y = net_D(fake_X.detach())  # fake_Y.shape = (n_steps, batch_size, 1)
    if not glv.network_config['is_mem']:
        fake_Y = torch.sum(fake_Y, dim=0) / glv.network_config[
            'n_steps']  # fake_Y.shape = (batch_size,1)
    fake_Y = fake_Y.mean()
    (-fake_Y).backward()
    trainer_D.step()
    # print(fake_Y.data, real_Y.data)
    return fake_Y.data, real_Y.data


def update_G(Z, net_D, net_G, trainer_G, theta=0.5):
    batch_size = Z.shape[0]
    trainer_G.zero_grad()
    fake_X, infonce_loss = net_G(
        Z)  # fake_X.shape = (n_steps, batch_size, 784)
    n_step = fake_X.shape[0]
    if glv.network_config['net_D_direct_input']:
        fake_X = fake_X.reshape((n_step, batch_size, 1, 28, 28))
    else:
        fake_X = fake_X.reshape((n_step, batch_size, 784))
    fake_Y = net_D(fake_X)  # shape = (n_steps, batch_size, 1)
    if not glv.network_config['is_mem']:
        fake_Y = torch.sum(
            fake_Y,
            dim=0) / glv.network_config['n_steps']  # shape = (batch_size, 1)\
    fake_Y = fake_Y.mean()
    (fake_Y + theta * infonce_loss).backward()
    trainer_G.step()
    # print(f'Y_before:{fake_Y.detach()}')
    '''with torch.no_grad():
        Y_after = net_D(fake_X.detach())
        if not glv.network_config['is_mem']:
            Y_after = torch.sum(Y_after, dim=0) / glv.network_config['n_steps']  # fake_Y.shape = (batch_size,1)
        Y_after = Y_after.mean()
        print(f'Y_after:{Y_after}')
        print(" ")'''
    return fake_Y.data


if __name__ == "__main__":
    # print("This is Moba")
    data_path = "./data"
    config = "./NetworkConfigs/mi_gen.yaml"
    params = Parse(config)
    glv.init(params['Network'])
    device = glv.network_config['device']
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, dest='name', type=str)
    parser.add_argument("--exp_index",
                        required=True,
                        dest='exp_index',
                        type=str)
    args = parser.parse_args()
    os.makedirs(f'./exps/{args.name}', exist_ok=True)
    os.makedirs(f'./exps/{args.name}/{args.exp_index}', exist_ok=True)
    logging.basicConfig(
        filename=
        f'./exps/{args.name}/{args.exp_index}/{args.name}_{args.exp_index}.log',
        level=logging.INFO)

    print("loading dataset")
    # trainloader, _ = load_datasets.load_mnist(data_path)
    if glv.network_config['dataset'] == 'MNIST':
        trainloader, _ = load_datasets.load_mnist(
            data_path, is_normlized=glv.network_config['is_data_normlized'])
    elif glv.network_config['dataset'] == 'FashionMNIST':
        trainloader, _ = load_datasets.load_fmnist(
            data_path, is_normlized=glv.network_config['is_data_normlized'])

    net_G, net_D = None, None
    if glv.network_config['net_G'] == 'Generator':
        net_G = GAN_snn.Generator()  # simple dense layer
    elif glv.network_config['net_G'] == 'Generator_2_d':
        net_G = DGAN_snn.Generator_2(
            input_dim=glv.network_config['latent_dim'])
    elif glv.network_config['net_G'] == 'Generator_2_MP_d':
        net_G = DGAN_snn.Generator_2_MP(
            input_dim=glv.network_config['latent_dim'])
    elif glv.network_config['net_G'] == 'Generator_2_MP_scoring_d':
        net_G = DGAN_snn.Generator_3_MP_Scoring(
            input_dim=glv.network_config['latent_dim'])
    elif glv.network_config['net_G'] == 'Generator_2_MP_scoring_d_2':
        net_G = DGAN_snn.Generator_3_MP_Scoring_2(
            input_dim=glv.network_config['latent_dim'])
    elif glv.network_config['net_G'] == 'Generator_2_MP_scoring_d_2_MI':
        net_G = DGAN_snn.Generator_3_MP_Scoring_2_MI(
            input_dim=glv.network_config['latent_dim'])
    else:
        print("net_G model does not exist!")

    if glv.network_config['net_D'] == "Discriminator":
        net_D = GAN_snn.Discriminator()  # simple dense layer
    elif glv.network_config['net_D'] == 'Discriminator_MP':
        net_D = GAN_snn.Discriminator_MP()
    elif glv.network_config['net_D'] == "Discriminator_2_d":
        net_D = DGAN_snn.Discriminator_2()  # using conv layer
    elif glv.network_config['net_D'] == "Discriminator_3_d":
        net_D = DGAN_snn.Discriminator_3()
    elif glv.network_config['net_D'] == "Discriminator_3_MP_d":
        net_D = DGAN_snn.Discriminator_3_MP()
    elif glv.network_config['net_D'] == "Discriminator_4_MP_d":
        net_D = DGAN_snn.Discriminator_4_MP()
    elif glv.network_config['net_D'] == "Discriminator_3_MP_d_w":
        net_D = WGAN_snn.Discriminator_3_MP()
    elif glv.network_config['net_D'] == "Discriminator_3_MP_d_w_MI":
        net_D = WGAN_snn.Discriminator_3_MP_MI()
    else:
        print("net_D model does not exist!")
    # net_G = GAN_snn.Generator().to(device)
    # net_D = GAN_snn.Discriminator().to(device)
    # vloss = nn.BCELoss(reduction='sum')
    optimizer_G = torch.optim.RMSprop(net_G.parameters(),
                                      lr=glv.network_config['lr_G'])
    optimizer_D = torch.optim.RMSprop(net_D.parameters(),
                                      lr=glv.network_config['lr_D'])

    net_G = net_G.to(device)
    net_D = net_D.to(device)

    init_epoch = 0
    if glv.network_config['from_checkpoint']:
        print("loading checkpoint")
        checkpoint = torch.load(glv.network_config['checkpoint_path'])
        init_epoch = checkpoint['epoch']
        net_D.load_state_dict(checkpoint['model_state_dict_D'])
        net_G.load_state_dict(checkpoint['model_state_dict_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_state_dict_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_state_dict_G'])

    scheduler_G = None
    scheduler_D = None
    if glv.network_config["is_scheduler"]:
        if glv.network_config["scheduler_mode"] == "cos":
            scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_D, T_max=20)
            scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_G, T_max=20)

    logging.info(glv.network_config)

    print("start training")
    best_fid_score = 9999
    # batch_size = glv.network_config['batch_size']
    for epoch in range(init_epoch, glv.network_config['epochs']):
        fake_mean = 0
        real_mean = 0
        g_mean = 0
        batch_count = 0
        g_count = 0
        for X, _ in tqdm(trainloader, colour='blue'):
            batch_count += 1
            batch_size = X.shape[0]
            if not glv.network_config['net_D_direct_input']:
                X = X.reshape((batch_size, -1))
            X = X.to(device)
            Z = torch.randn((batch_size, glv.network_config['latent_dim']),
                            device=device)
            mean_increment_fake, mean_increment_real = update_D(
                X, Z, net_D, net_G, optimizer_D)
            # for parm in net_D.parameters():
            # parm.data.clamp_((-1)*glv.network_config['clamp_num'],glv.network_config['clamp_num'])
            fake_mean += mean_increment_fake
            real_mean += mean_increment_real
            reset_net(net_D)
            reset_net(net_G)
            if batch_count % glv.network_config['n_critic'] == 0:
                mean_increment_g = update_G(Z, net_D, net_G, optimizer_G)
                g_count += 1
                g_mean += mean_increment_g
            reset_net(net_D)
            reset_net(net_G)

        eta_G, eta_D = -1, -1
        if glv.network_config['is_scheduler']:
            eta_D = scheduler_D.get_last_lr()
            eta_G = scheduler_G.get_last_lr()
            scheduler_D.step()
            scheduler_G.step()

        with torch.no_grad():
            Z = torch.randn((21, glv.network_config['latent_dim']),
                            device=device)
            fake_X, _ = net_G(Z)  # fake_X.shape = (n_steps, batch_size, 784)
            reset_net(net_G)
            fake_X = torch.sum(
                fake_X,
                dim=0) / glv.network_config['n_steps']  # (batch_size, 784) 0~1
            fake_X = fake_X.reshape((21, 1, 28, 28))
            imgs = torch.cat([
                torch.cat([fake_X[i * 7 + j, :, :, :] for j in range(7)],
                          dim=2) for i in range(3)
            ],
                             dim=1)
            # imgs = imgs * 255.0
            save_image(
                imgs, f'./exps/{args.name}/{args.exp_index}/Epoch{epoch}.png')
            fid_score = -1
            better_model = False
            if glv.network_config['compute_fid']:
                fid_score = clean_fid.get_clean_fid_score(
                    lambda x: net_G(x)[0],
                    glv.network_config['dataset'],
                    device,
                    num_gen=5000)
                if fid_score < best_fid_score:
                    best_fid_score = fid_score
                    better_model = True
                reset_net(net_G)
            if better_model:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict_D': net_D.state_dict(),
                        'model_state_dict_G': net_G.state_dict(),
                        'optimizer_state_dict_D': optimizer_D.state_dict(),
                        'optimizer_state_dict_G': optimizer_G.state_dict()
                    }, f'./exps/{args.name}/{args.exp_index}/checkpoint.pth')
        logging.info(
            f'Epoch: {epoch}'
            f'fake_credit:{fake_mean / batch_count},'
            f'real_credit:{real_mean / batch_count}, g_credit:{g_mean / g_count},'
            f'Fid_sore: {fid_score}, best_fid_score: {best_fid_score}, eta_D: {eta_D}, eta_G: {eta_G}'
        )
        print(
            f'Epoch: {epoch}'
            f'fake_credit:{fake_mean / batch_count},'
            f'real_credit:{real_mean / batch_count}, g_credit:{g_mean / g_count}'
            f'Fid_sore: {fid_score}, best_fid_score: {best_fid_score}, eta_D: {eta_D}, eta_G: {eta_G}'
        )
