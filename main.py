import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import cyclegan
import numpy as np 
np.random.seed(19)


####################################################################################
### wei_w=276,h=368_no-focus
# name = "wei_w=276,h=368_no-focus"
# phase = "train"

### train完後test
# phase = "test"

# epoch = 400
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = True



####################################################################################
### wei_focus_w=300,h=400
# name = "wei_focus_w=300,h=400"
# phase = "train"
### train完後test
# phase = "test"

# epoch = 200
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# image_size_width = 300
# image_size_height = 400


####################################################################################
### wei_focus_w=330,h=440
# name = "wei_focus_w=330,h=440"
# phase = "train"
### train完後test
# phase = "test"

# epoch = 1000
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# image_size_width = 332
# image_size_height = 440

####################################################################################
### wei_focus_w=336,h=448
# name = "wei_focus_w=336,h=448"
# phase = "train"
### train完後test
# phase = "test"

# epoch = 200
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# image_size_width = 336
# image_size_height = 448


# ####################################################################################
# ### wei_focus_w=408,h=356 have shuffle
# name = "wei_focus_w=408,h=356"
# # phase = "train"
# ## train完後test
# phase = "test"

# epoch = 1000
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 408
# image_size_height = 356

####################################################################################
### wei_focus_w=408,h=356 no-shuffle
# name = "wei_focus_w=408,h=356_no-shuffle"
# # phase = "train"
# ### train完後test
# phase = "test"

# epoch = 500
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 408
# image_size_height = 356

####################################################################################
### 01 改96x96, B有全白, Mix random crop, 4個 random crop, 沒有shuffle
### wei_focus_just_text_96x96_01_x4
# name = "wei_focus_just_text_96x96_01_x4"
# # phase = "train"
# ## train完後test
# phase = "test"

# epoch = 1000
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
### 02 改random crop 10個
### wei_focus_just_text_96x96_02_x10
# name = "wei_focus_just_text_96x96_02_x10"
# # phase = "train"
# ## train完後test
# phase = "test"

# epoch = 1000
# dataset_dir = name
# save_freq = 1000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96


####################################################################################
### 03 改 left top
### wei_focus_just_text_96x96_03_left-top
# name = "wei_focus_just_text_96x96_03_left-top"
# # phase = "train"
# ## train完後test
# phase = "test"

# epoch = 800

# save_freq = 10000
# print_freq = 10
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96


####################################################################################
### 04 把 Domain B 的白色部分拿掉，同時也拿掉對應的 Domain A影像
### wei_focus_just_text_96x96_04_AB-no-white
# name = "wei_focus_just_text_96x96_04_AB-no-white"
# phase = "train"
# ## train完後test
# phase = "test"

# epoch = 800

# save_freq = 10000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
# ### 05 拿掉 utils.load_train_data 的 fliplr
# ### wei_focus_just_text_96x96_05_no-fliplr
# name = "wei_focus_just_text_96x96_05_no-fliplr"
# # phase = "train"
# # ## train完後test
# phase = "test"

# epoch = 800

# save_freq = 10000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
### 06 改成shuffle試試看 
### wei_focus_just_text_96x96_06_shuffle
# name = "wei_focus_just_text_96x96_06_shuffle"
# # phase = "train"
# # ## train完後test
# phase = "test"

# epoch = 800

# save_freq = 10000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96


####################################################################################
### 07 A的training有和B同步好了 
### wei_focus_just_text_96x96_07_A-no-white-ok
# name = "wei_focus_just_text_96x96_07_A-no-white-ok"
# # phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 #30000
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
### 07 B的training有和B同步好了 
### wei_focus_just_text_96x96_07b_A-no-white-ok_shuffle
# name = "wei_focus_just_text_96x96_07b_A-no-white-ok_shuffle"
# phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 ### 最多好像存5次
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
# ### 08a 加入 10張格線影像 且 no-shuffle
# ## wei_focus_just_text_96x96_08a_add-line-img_x10_no-shuffle
# name = "wei_focus_just_text_96x96_08a_add-line-img_x10_no-shuffle"
# phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 ### 最多好像存5次
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
### 08b 加入 10張格線影像 且 have-shuffle
### wei_focus_just_text_96x96_08b_add-line-img_x10_have-shuffle
# name = "wei_focus_just_text_96x96_08b_add-line-img_x10_have-shuffle"
# phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 ### 最多好像存5次
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 96
# image_size_height = 96

####################################################################################
### 09 試試看no-cycle
### wei-crop-accurate_w=304,h=472_mix
# name = "wei-crop-accurate_w=304,h=472_mix"
# phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 ### 最多好像存5次
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 304
# image_size_height = 472

# ####################################################################################
# ### wei-crop-accurate_w=304,h=472_mix_x328
# name = "wei-crop-accurate_w=304,h=472_mix_x328"
# phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 ### 最多好像存5次
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 304
# image_size_height = 472

####################################################################################
### wei-crop-accurate_w=304,h=472_left-top_x82
# name = "wei-crop-accurate_w=304,h=472_left-top_x82"
# phase = "train"
# # ## train完後test
# # phase = "test"

# epoch = 527 #800

# save_freq = 20000 ### 最多好像存5次
# print_freq = 100
# continue_train = False
# # continue_train = True
# image_size_width = 304
# image_size_height = 472

####################################################################################
### wei-crop-accurate_w=304,h=472_mix_x328_new_model
name = "wei-crop-accurate_w=304,h=472_mix_x328_new_model"
phase = "train"
# ## train完後test
# phase = "test"

epoch = 527 #800

save_freq = 20000 ### 最多好像存5次
print_freq = 100
continue_train = False
lambda_kong = 3
# continue_train = True
image_size_width = 304
image_size_height = 472

####################################################################################
dataset_dir = name
checkpoint_dir = name + "/checkpoint"
sample_dir = name + "/sample"
test_dir = name + "/test"
log_dir = name + "/logs"

####################################################################################

class Kong_args():
    def __init__(self):
        #self.dataset_dir = "horse2zebra" ### datasets下面的資料集名稱
        self.dataset_dir = dataset_dir # "wei_w=276,h=368" ### datasets下面的資料集名稱
        self.epoch= epoch # 200        ### epoch數
        self.epoch_step =100  ### 更新幾次下降一次learning rate吧~~~
        self.batch_size =1    ### batch_size
        self.train_size =1e8  ### images used to train，training_data的數量
        self.load_size =286   ### scale images to this size
        self.fine_size =256   ### then crop to this size
        self.ngf =64          ### of gen filters in first conv layer
        self.ndf =64          ### of discri filters in first conv layer
        self.input_nc =3      ### 輸入影像的channel數
        self.output_nc =3     ### 輸出影像的channel數
        self.lr =0.0002       ### Adam最佳化器的初始learning rate
        self.beta1 =0.5       ### momentum term of adam
        self.which_direction='AtoB' ### 'AtoB or BtoA'
        self.phase = phase # 'train' ### 'train 或 test'
        self.save_freq  = save_freq   # 1000 ### 更新幾次後 儲存一次model
        self.print_freq = print_freq  # 100 ### 更新幾次後 sample一下目前的Model生成的影像
        self.continue_train = continue_train # True ### if continue training, load the latest model: 1: true, 0: false')
        self.checkpoint_dir = checkpoint_dir #'./checkpoint' ### models are saved here
        self.sample_dir = sample_dir #'./sample' ### sample are saved here
        self.test_dir = test_dir #'./test' ###'test sample are saved here')
        self.L1_lambda = lambda_kong # 10.0 ###'weight on L1 term in objective')
        self.use_resnet =True ###'generation network using reidule block')
        self.use_lsgan =True ###gan loss defined in lsgan')
        self.max_size =50 ###max size of image pool, 0 means do not use image pool')
        
        self.image_size_width  = image_size_width
        self.image_size_height = image_size_height
        self.log_dir = log_dir 
        self.name = name

args = Kong_args()
#args = parser.parse_args()

# print("args",args)
# print("type(args)",type(args))
def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
        # os.makedirs(args.sample_dir +"/A")
        # os.makedirs(args.sample_dir +"/B")

        ### 因為寫在function裡會一直被呼叫到，所以我才拉出來main寫喔！
        # os.makedirs(args.sample_dir +"/to_curved/big")
        # os.makedirs(args.sample_dir +"/to_curved/big-left-top")
        # os.makedirs(args.sample_dir +"/to_curved/small-seen")
        # os.makedirs(args.sample_dir +"/to_curved/small-unseen")
        # os.makedirs(args.sample_dir +"/to_straight/big")
        # os.makedirs(args.sample_dir +"/to_straight/big-left-top")
        # os.makedirs(args.sample_dir +"/to_straight/small-seen")
        # os.makedirs(args.sample_dir +"/to_straight/small-unseen")

        ### 因為寫在function裡會一直被呼叫到，所以我才拉出來main寫喔！
        os.makedirs(args.sample_dir +"/to_straight/crop-accurate")

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        #model.train(args) if args.phase == 'train' \
        model.train_kong(args) if args.phase == 'train' \
             else model.test(args)

if __name__ == '__main__':
    # print(tf.__version__)
    tf.app.run()
