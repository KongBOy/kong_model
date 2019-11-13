class Kong_args():
    def __init__(self):
        self.dataset_dir = "horse2zebra" ### datasets下面的資料集名稱
        self.epoch=200        ### epoch數
        self.epoch_step =100  ### 更新幾次下降一次learning rate吧~~~
        self.batch_size =1    ### batch_size
        self.train_size =1e8  ### images used to train
        self.load_size =286   ### scale images to this size
        self.fine_size =256   ### then crop to this size
        self.ngf =64          ### of gen filters in first conv layer
        self.ndf =64          ### of discri filters in first conv layer
        self.input_nc =3      ### 輸入影像的channel數
        self.output_nc =3     ### 輸出影像的channel數
        self.lr =0.0002       ### Adam最佳化器的初始learning rate
        self.beta1 =0.5       ### momentum term of adam
        self.which_direction='AtoB' ### 'AtoB or BtoA'
        self.phase ='train' ### 'train 或 test'
        self.save_freq =1000 ### 更新幾次後 儲存一次model
        self.print_freq =100 ### 更新幾次後 sample一下目前的Model生成的影像
        self.continue_train =False ### if continue training, load the latest model: 1: true, 0: false')
        self.checkpoint_dir ='./checkpoint' ### models are saved here
        self.sample_dir ='./sample' ### sample are saved here
        self.test_dir ='./test' ###'test sample are saved here')
        self.L1_lambda =10.0 ###'weight on L1 term in objective')
        self.use_resnet =True ###'generation network using reidule block')
        self.use_lsgan =True ###gan loss defined in lsgan')
        self.max_size =50 ###max size of image pool, 0 means do not use image pool')
        
        self.image_size_height = 128
        self.image_size_width = 800