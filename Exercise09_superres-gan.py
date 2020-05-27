## Pretrained GAN

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *

path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'crappy'

# from crappify import *

il = ImageList.from_folder(path_hr)
# 함수 이름을 병렬로 전달함.
parallel(crappifier(path_lr, path_hr), il.items)

bs, size = 32, 128
# bs,size = 24,160
# bs,size = 8,256
arch = models.resnet34

## Pre-train generator

arch = models.resnet34
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)

# pre-trained model을 사용하므로 imagenet_stats를 가져옴.
def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr / x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data

data_gen = get_data(bs, size)

data_gen.show_batch(4)

wd = 1e-3

y_range = (-3., 3.)

# 이미지를 긴 벡터로 평면화한다.
loss_gen = MSELossFlat()

# 데이터를 가지고 Unet을 만듦.
def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=y_range, loss_func=loss_gen)

# generator.
learn_gen = create_gen_learner()

learn_gen.fit_one_cycle(2, pct_start=0.8)

learn_gen.unfreeze()

learn_gen.fit_one_cycle(3, slice(1e-6, 1e-3))

learn_gen.show_results(rows=4)

learn_gen.save('gen-pre2')

## Save generated images

learn_gen.load('gen-pre2');

name_gen = 'image_gen'
path_gen = path / name_gen

# shutil.rmtree(path_gen)

path_gen.mkdir(exist_ok=True)

def save_preds(dl):
    i = 0
    names = dl.dataset.items

    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1

save_preds(data_gen.fix_dl)

PIL.Image.open(path_gen.ls()[0])

## Train critic

learn_gen = None
gc.collect()

def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
            .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data

data_crit = get_crit_data([name_gen, 'images'], bs=bs, size=size)

data_crit.show_batch(rows=3, ds_type=DatasetType.Train, imgsize=3)

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())

def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)

learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)

learn_critic.fit_one_cycle(6, 1e-3)

learn_critic.save('critic-pre2')

## GAN

learn_crit = None
learn_gen = None
gc.collect()

data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size)

learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')

learn_gen = create_gen_learner().load('gen-pre2')

switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1., 50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0., 0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4

learn.fit(40, lr)

learn.save('gan-1c')

learn.data = get_data(16, 192)

learn.fit(10, lr / 2)

learn.show_results(rows=16)

learn.save('gan-1c')
