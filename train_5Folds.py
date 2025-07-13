import os
import time
import numpy as np
import torch
from PIL import Image

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util import html
from util.visualizer import Visualizer, save_images
from sklearn.model_selection import KFold
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def tensor_to_gray_image(tensor):
    """
    Convert a PyTorch tensor (3-channel) to a grayscale image in uint8 format.
    """
    image = (
        (tensor * 255)
        .cpu()
        .numpy()
        .astype("uint8")
        .squeeze(0)
        .transpose(1, 2, 0)  # HWC format
    )
    # Convert to grayscale using PIL
    gray_image = Image.fromarray(image).convert("L")
    return np.array(gray_image).astype(np.uint8)


if __name__ == '__main__':
    datasets_list = ["kidney", "liver"]
    # datasets_list = ["thyroid"]
    for dataset in datasets_list:
        print("Training on {}".format(dataset))
        opt = TrainOptions().parse()  # get training options
        opt.dataroot += "train_datasets/"+dataset
        opt.name = "vit-base-branch_USFM_US-enhangce-" + dataset + "_cyclegan"
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        # Perform k-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

        fold = 1
        for train_idx, val_idx in kfold.split(np.arange(dataset_size)):  # iterate through each fold
            print(f"\nTraining fold {fold}/5...")

            # Create a new model for each fold
            model = create_model(opt)  # create a model given opt.model and other options
            print(model)
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

            # Split dataset into training and validation sets
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)

            # Create data loaders for both training and validation sets
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

            total_iters = 0  # the total number of training iterations

            for epoch in range(opt.epoch_count,
                               opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs
                epoch_start_time = time.time()  # timer for entire epoch
                iter_data_time = time.time()  # timer for data loading per iteration
                epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
                visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
                model.update_learning_rate()  # update learning rates in the beginning of every epoch

                # Training phase
                model.isTrain = True
                for i, data in enumerate(train_loader):  # inner loop within one epoch
                    iter_start_time = time.time()  # timer for computation per iteration
                    if total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)  # unpack data from dataset and apply preprocessing
                    model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                    if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                        save_result = total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / len(train_loader), losses)

                # Save model and log results after each epoch
                if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                    print(f"Saving model at the end of epoch {epoch}, iters {total_iters}")
                    model.save_networks('latest')
                    # model.save_networks(epoch)

                print(
                    f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec")

            # Validation phase
            model.isTrain = False
            val_losses = []
            # ssim_values_A = []  # 用于存储A域每张图像的 SSIM
            # psnr_values_A = []  # 用于存储A域每张图像的 PSNR
            # ssim_values_B = []  # 用于存储B域每张图像的 SSIM
            # psnr_values_B = []  # 用于存储B域每张图像的 PSNR

            ssim_values = []
            psnr_values = []

            # create a website
            web_dir = os.path.join(opt.results_dir, opt.name,
                                   'fold_{}_{}_{}'.format(fold, opt.phase, opt.epoch))  # define the website directory
            if opt.load_iter > 0:  # load_iter is 0 by default
                web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
            print('creating web directory', web_dir)
            webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

            for i, data in enumerate(val_loader):  # validate on validation set
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.test()  # forward pass (no gradient computation)

                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths

                # generated_image = model.get_current_visuals()['fake_B']  # 获取生成图像
                # real_image = model.get_current_visuals()['real_A']  # 获取真实图像

                # real_A = model.get_current_visuals()['real_A']
                fake_A = model.get_current_visuals()['fake_A']
                real_B = model.get_current_visuals()['real_B']
                # fake_B = model.get_current_visuals()['fake_B']
                # rec_A = model.get_current_visuals()['rec_A']
                # rec_B = model.get_current_visuals()['rec_B']

                val_losses.append(model.get_current_losses())

                # 将图像从 [0, 1] 转换为 [0, 255]，并转换为 numpy 数组
                # generated_image = (generated_image * 255).cpu().numpy().astype("uint8").squeeze(0).transpose(1, 2, 0)
                # real_image = (real_image * 255).cpu().numpy().astype("uint8").squeeze(0).transpose(1, 2, 0)

                # real_A = tensor_to_gray_image(real_A)
                # fake_A = tensor_to_gray_image(fake_A)
                # real_B = tensor_to_gray_image(real_B)
                # fake_B = tensor_to_gray_image(fake_B)
                # rec_A = tensor_to_gray_image(rec_A)
                # rec_B = tensor_to_gray_image(rec_B)

                real_B = tensor_to_gray_image(real_B)
                fake_A = tensor_to_gray_image(fake_A)


                # 计算 SSIM 和 PSNR
                # ssim_value = structural_similarity(generated_image, real_image, channel_axis=2)
                # psnr_value = peak_signal_noise_ratio(real_image, generated_image)

                # ssim_value_A = structural_similarity(real_A, fake_A)
                # ssim_value_B = structural_similarity(real_B, fake_B)
                # psnr_value_A = peak_signal_noise_ratio(real_A, fake_A)
                # psnr_value_B = peak_signal_noise_ratio(real_B, fake_B)
                ssim_value = structural_similarity(real_B, fake_A)
                psnr_value = peak_signal_noise_ratio(real_B, fake_A)

                # ssim_values_A.append(ssim_value_A)
                # psnr_values_A.append(psnr_value_A)
                # ssim_values_B.append(ssim_value_B)
                # psnr_values_B.append(psnr_value_B)

                ssim_values.append(ssim_value)
                psnr_values.append(psnr_value)

                save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                            use_wandb=opt.use_wandb)
            webpage.save()  # save the HTML

            # Log validation losses after the fold
            val_GA_losses = []
            val_GB_losses = []
            val_DA_losses = []
            val_DB_losses = []
            # val_cycleA_losses = []
            # val_cycleB_losses = []
            for each in val_losses:
                val_GA_losses.append(each['G_A'])
                val_GB_losses.append(each['G_B'])
                val_DA_losses.append(each['D_A'])
                val_DB_losses.append(each['D_B'])
                # val_cycleA_losses.append(each['cycle_A'])
                # val_cycleB_losses.append(each['cycle_B'])
            avg_val_GA_loss = np.mean(val_GA_losses)
            avg_val_GB_loss = np.mean(val_GB_losses)
            avg_val_DA_loss = np.mean(val_DA_losses)
            avg_val_DB_loss = np.mean(val_DB_losses)
            # avg_val_cycleA_loss = np.mean(val_cycleA_losses)
            # avg_val_cycleB_loss = np.mean(val_cycleB_losses)

            # 计算验证集的平均 SSIM 和 PSNR
            # avg_ssim_A = np.mean(ssim_values_A)
            # avg_psnr_A = np.mean(psnr_values_A)
            # avg_ssim_B = np.mean(ssim_values_B)
            # avg_psnr_B = np.mean(psnr_values_B)
            avg_ssim = np.mean(ssim_values)
            avg_psnr = np.mean(psnr_values)

            avg_val_dicts = {"avg_ssim": avg_ssim, "avg_psnr": avg_psnr,
                             "avg_G_A": avg_val_GA_loss, "avg_G_B": avg_val_GB_loss,
                             "avg_D_A": avg_val_DA_loss, "avg_D_B": avg_val_DB_loss}

            visualizer.print_current_fold_avg_val_losses(fold=fold, losses=avg_val_dicts)

            model.save_networks(f"fold_{fold}_final")
            fold += 1

        print("Cross-validation training completed!")
