from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        # self.image_size = args.fine_size
        self.image_size_height = args.image_size_height
        self.image_size_width  = args.image_size_width
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        # OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
        #                       gf_dim df_dim output_c_dim is_training')
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size_height image_size_width\
                        gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, self.image_size_height, self.image_size_width,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))
        
        build_model_start_time = time.time()
        self._build_model()
        print("_build_model_cost_time:",time.time()-build_model_start_time)
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

        self.Kong_sample_dir = args.sample_dir
        self.Kong_test_dataA = None
        self.Kong_test_dataB = None
        self.Kong_test_dataPairs = None
        self.Kong_load_test_dataset()

    


    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        #[None, self.image_size_height, self.image_size_width,
                                        [None, None, None,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        ########################################################################################################
        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        ########################################################################################################
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size_height, self.image_size_width,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size_height, self.image_size_width,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss
        ########################################################################################################
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )
        ########################################################################################################
        self.test_A = tf.placeholder(tf.float32,
                                     #[None, self.image_size_height, self.image_size_width,
                                     [None, None, None,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     #[None, self.image_size_height, self.image_size_width,
                                     [None, None, None,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        print("train start count time")
        train_start_time = time.time()

        """Train cyclegan"""
        self.Kong_copy_current_py(args.name) ### 把目前的設定存一份起來！

        ### 這裡也花時間，大概30秒左右 建立完 lr, d_optim, g_optim
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        print("finish optimizer cost_time:",time.time() - train_start_time)
        
        ### 這兩步還好，只花了2秒鐘
        init_op = tf.global_variables_initializer() 
        self.sess.run(init_op)
        print("finish global_variable_initializer cost_time:",time.time() - train_start_time) 
        
        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph) ### 這個最花時間 大概70幾秒
        print("finish FileWriter cost_time:",time.time() - train_start_time)
        
        counter = 1
        start_time = time.time()
        import cv2
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        print("train before epoch cost time:",time.time()-train_start_time)

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            #np.random.shuffle(dataA)
            #np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size : (idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size : (idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, self.image_size_height, self.image_size_width) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                # print("type(batch_images)",type(batch_images))
                # print("batch_images.shape",batch_images.shape)
                # print("batch_images[0]",batch_images[0,:,:,:3])
                # cv2.imshow("batch_images[0]",batch_images[0,:,:,:3])

                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])
                # print("type(fake_A)",type(fake_A))
                # print("fake_A.shape",fake_A.shape)
                # print("fake_A",fake_A[0])
                # cv2.imshow("fake_A",fake_A[0])
                # cv2.imshow("fake_B",fake_B[0])
                # cv2.waitKey(0)

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                # counter += 1 原始寫這邊，我把它調到下面去囉
                cost_time = time.time() - start_time
                hour = cost_time//3600 ; minute = cost_time%3600//60 ; second = cost_time%3600%60
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f, %2d:%02d:%02d counter:%d" % (
                    epoch, idx, batch_idxs, time.time() - start_time,hour, minute, second, counter)))

                if np.mod(counter, args.print_freq) == 1:#1:
                    #self.sample_model(args.sample_dir, epoch, idx, args,  counter) ### sample目的地資料夾、存圖時紀錄epoch
                    # self.Kong_sample_patch_version(args.sample_dir, counter)  ### sample目的地資料夾、存圖時紀錄counter
                    self.Kong_sample_crop_accurate(args.sample_dir, counter)  ### sample目的地資料夾、存圖時紀錄counter

                if np.mod(counter, args.save_freq) == 1:#2:
                    self.save(args.checkpoint_dir, counter)

                counter += 1  ### 調到這裡

    def save(self, checkpoint_dir, step):
        #model_dir = "%s_%s" % (self.dataset_dir, self.image_size_width) ### 原本的寫法
        model_dir = "%s" % (self.dataset_dir) ### self.image_size_width 資訊已經融入db名字了，所以我省了 
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_name = "cyclegan.model"
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        #model_dir = "%s_%s" % (self.dataset_dir, self.image_size_width) ### 原本的寫法
        model_dir = "%s" % (self.dataset_dir) ### self.image_size_width 資訊已經融入db名字了，所以我省了 
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def Kong_copy_current_py(self,dst_dir):
        import shutil
        shutil.copy("main.py",   dst_dir+"/main.py")
        shutil.copy("model.py",  dst_dir+"/model.py")
        shutil.copy("module.py", dst_dir+"/module.py")
        shutil.copy("ops.py",    dst_dir+"/ops.py")
        shutil.copy("utils.py",  dst_dir+"/utils.py")


    def Kong_load_test_dataset(self):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        dataA.sort()
        dataB.sort()
        self.Kong_test_dataA = dataA
        self.Kong_test_dataB = dataB
        self.Kong_test_dataPairs = list(zip(self.Kong_test_dataA[:], self.Kong_test_dataB[:]))
        # for dataPair in self.Kong_test_dataPairs:
        #     print(dataPair)


    def Kong_sample_seperately(self, name, batch_files, img_num , counter):
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        fake_A, fake_B = self.sess.run( [self.fake_A, self.fake_B], feed_dict={self.real_data: sample_images})
        save_images(fake_A, [ 1, img_num ], './{}/to_curved/{}/{:02d}.jpg'.format(self.Kong_sample_dir, name, counter))
        save_images(fake_B, [ 1, img_num ], './{}/to_straight/{}/{:02d}.jpg'.format(self.Kong_sample_dir, name, counter))


    ### 在修改sample的時候，記得要去main裡 新增資料夾喔！
    ### 在修改sample的時候，記得要去main裡 新增資料夾喔！
    ### 在修改sample的時候，記得要去main裡 新增資料夾喔！很重要說三次！
    def Kong_sample_patch_version(self,sample_dir,counter):
        ### 注意，好像不能再 training 時 用test_A或test_B 會失敗這樣子，所以只好用 sample的方式囉！就成功了～
        self.Kong_sample_seperately("big", self.Kong_test_dataPairs[5:10], 5, counter)
        self.Kong_sample_seperately("big-left-top", self.Kong_test_dataPairs[  : 5], 5, counter)
        self.Kong_sample_seperately("small-seen"  , self.Kong_test_dataPairs[10:15], 5, counter)
        self.Kong_sample_seperately("small-unseen", self.Kong_test_dataPairs[15:20], 5, counter)


    ### 在修改sample的時候，記得要去main裡 新增資料夾喔！
    ### 在修改sample的時候，記得要去main裡 新增資料夾喔！
    ### 在修改sample的時候，記得要去main裡 新增資料夾喔！很重要說三次！
    def Kong_sample_crop_accurate(self,sample_dir,counter,sample_amount = 1):
        ### 注意，好像不能再 training 時 用test_A或test_B 會失敗這樣子，所以只好用 sample的方式囉！就成功了～
        self.Kong_sample_seperately("crop-accurate", self.Kong_test_dataPairs[ :sample_amount], sample_amount, counter)


    def sample_model(self, sample_dir, epoch, idx,args, counter): ### counter自己加的
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        # np.random.shuffle(dataA)
        # np.random.shuffle(dataB)

        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size])) ### 這裡因為batch_size設1，只會讀到一張而已喔
        
        # sample_images = [load_train_data(batch_file, args.load_size, self.image_size_height, self.image_size_width, is_testing=True) for batch_file in batch_files]
        sample_images = [load_train_data(batch_file, args.load_size, self.image_size_height, self.image_size_width, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        print("sample_images.shape",sample_images.shape)
        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )

        ### 原本的，用epoch 和 idx，但我想看的是 更新幾次配合img，還要自己轉很麻煩
        # save_images(fake_A, [self.batch_size, 1],
        #             './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        # save_images(fake_B, [self.batch_size, 1],
        #             './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

        ### 所以改成直接放counter囉！
        print("fake_A.shape",fake_A.shape)
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}.jpg'.format(sample_dir, counter))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}.jpg'.format(sample_dir, counter))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')
        
        #print("args.checkpoint_dir",args.checkpoint_dir)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, self.image_size_height, self.image_size_width)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '../..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '../..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()