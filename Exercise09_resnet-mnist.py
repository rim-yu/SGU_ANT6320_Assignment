# fastai.vision의 모든 모듈을 import한다.
from fastai import *
from fastai.vision import *

# MNIST data의 path를 가져옴.
path = untar_data(URLs.MNIST)

# path 안에 무엇이 있는지 보여줌.
path.ls()

# convert_mode = 'L' -> gray scale.
il = ImageList.from_folder(path, convert_mode='L')

# 가져온 파일 목록 중 하나를 보여준다.
il.items[0]

# binary color map을 사용함.
defaults.cmap = 'binary'

# 70,000 items. (1, 28, 28)
il

il[0].show()
plt.show()

# 폴더를 기준으로 train/valid 데이터셋으로 나눈다.
sd = il.split_by_folder(train='training', valid='testing')

# split data. 60,000개의 train 데이터셋과 10,000개의 valid 데이터셋으로 나누어짐.
sd

(path/'training').ls()

# label list.
ll = sd.label_from_folder()

ll

x, y = ll.train[0]

x.show()
plt.show()
print(y, x.shape)

# transforms. MNIST는 숫자 이미지 데이터라서 회전 등 일반적인 transform을 할 경우 의미가 바뀌어버린다.
# [] : valid 데이터셋에 대해 transform을 하지 않음.
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])

ll = ll.transform(tfms)

bs = 128

# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize()

x, y = data.train_ds[0]

x.show()
plt.show()
print(y)

# 임의의 패딩을 했으므로 약간씩 다른 위치에 있는 숫자를 가져올 수 있음.
def _plot(i, j, ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8, 8))

xb, yb = data.one_batch()
xb.shape, yb.shape

data.show_batch(rows=3, figsize=(5, 5))

# 간단한 CNN 만들기.
def conv(ni, nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)

model = nn.Sequential(
    # gray scale이라서 채널 하나.
    # (1, 28, 28) -> (8, 14, 14)
    conv(1, 8),  # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16),  # 7
    # 채널 수를 점차 늘려감.
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32),  # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16),  # 2
    # 채널 수 점차 줄여감.
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10),  # 1
    nn.BatchNorm2d(10),
    # [128, 10, 1, 1] -> [128, 10]
    Flatten()  # remove (1,1) grid
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.summary()

xb = xb.cuda()

# 128*10을 돌려줌.
model(xb).shape

learn.lr_find(end_lr=100)

learn.recorder.plot()

learn.fit_one_cycle(3, max_lr=0.1)

# Refactor
def conv2(ni, nf): return conv_layer(ni, nf, stride=2)

model = nn.Sequential(
    conv2(1, 8),  # 14
    conv2(8, 16),  # 7
    conv2(16, 32),  # 4
    conv2(32, 16),  # 2
    conv2(16, 10),  # 1
    Flatten()  # remove (1,1) grid
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, max_lr=0.1)

# Resnet-ish
# x + conv2(conv1(x))
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))

# help(res_block)

model = nn.Sequential(
    conv2(1, 8),
    # ResBlock 추가됨.
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)

def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), res_block(nf))

# ResNet-ish Architecture.
model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.lr_find(end_lr=100)
learn.recorder.plot()

learn.fit_one_cycle(12, max_lr=0.05)

print(learn.summary())