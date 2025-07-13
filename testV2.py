import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, Visualizer
from util import html
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image
import numpy as np

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


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
    datasets_list = ["thyroid"]
    for dataset in datasets_list:
        opt = TestOptions().parse()  # get test options
        opt.results_dir += 'selected/usfm/'+dataset
        opt.dataroot += "/10 testset/" + dataset
        # opt.name += "-" + dataset + "_cyclegan"
        # hard-code some parameters for test
        opt.num_threads = 0  # test code only supports num_threads = 0
        opt.batch_size = 1  # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)  # create a model given opt.model and other options
        print(model)
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)

        # initialize logger
        if opt.use_wandb:
            wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                                   config=opt) if not wandb.run else wandb.run
            wandb_run._label(repo='CycleGAN-and-pix2pix')

        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name,
                               '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()

        visualizer.reset()
        ssim_values = []
        psnr_values = []
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths

            fake_A = model.get_current_visuals()['fake_A']
            real_B = model.get_current_visuals()['real_B']

            # 转换为灰度图
            real_B = tensor_to_gray_image(real_B)
            fake_A = tensor_to_gray_image(fake_A)

            # 计算 SSIM 和 PSNR
            ssim_values.append(structural_similarity(real_B, fake_A))
            psnr_values.append(peak_signal_noise_ratio(real_B, fake_A))

            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                        use_wandb=opt.use_wandb)
        webpage.save()  # save the HTML

        avg_ssim = np.mean(ssim_values)
        avg_psnr = np.mean(psnr_values)
        avg_val_dicts = {"avg_ssim": avg_ssim, "avg_psnr": avg_psnr}
        visualizer.print_test_result(avg_val_dicts)
