from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv')) ### df_dim 意思是 D 的第一層 feature map 的channel數
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def generator_unet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks ### try12試試看
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
        r10 = residule_block(r9, options.gf_dim*4, name='g_r10')
        r11 = residule_block(r10, options.gf_dim*4, name='g_r11')
        r12 = residule_block(r11, options.gf_dim*4, name='g_r12')
        r13 = residule_block(r12, options.gf_dim*4, name='g_r13')
        r14 = residule_block(r13, options.gf_dim*4, name='g_r14')
        r15 = residule_block(r14, options.gf_dim*4, name='g_r15')
        r16 = residule_block(r15, options.gf_dim*4, name='g_r16')
        r17 = residule_block(r16, options.gf_dim*4, name='g_r17')
        r18 = residule_block(r17, options.gf_dim*4, name='g_r18')
        r19 = residule_block(r18, options.gf_dim*4, name='g_r19')
        r20 = residule_block(r19, options.gf_dim*4, name='g_r20')
        r21 = residule_block(r20, options.gf_dim*4, name='g_r21')
        r22 = residule_block(r21, options.gf_dim*4, name='g_r22')
        r23 = residule_block(r22, options.gf_dim*4, name='g_r23')
        r24 = residule_block(r23, options.gf_dim*4, name='g_r24')
        r25 = residule_block(r24, options.gf_dim*4, name='g_r25')
        r26 = residule_block(r25, options.gf_dim*4, name='g_r26')
        r27 = residule_block(r26, options.gf_dim*4, name='g_r27')
        r28 = residule_block(r27, options.gf_dim*4, name='g_r28')
        r29 = residule_block(r28, options.gf_dim*4, name='g_r29')
        r30 = residule_block(r29, options.gf_dim*4, name='g_r30')
        r31 = residule_block(r30, options.gf_dim*4, name='g_r31')
        r32 = residule_block(r31, options.gf_dim*4, name='g_r32')
        r33 = residule_block(r32, options.gf_dim*4, name='g_r33')
        r34 = residule_block(r33, options.gf_dim*4, name='g_r34')
        r35 = residule_block(r34, options.gf_dim*4, name='g_r35')
        r36 = residule_block(r35, options.gf_dim*4, name='g_r36')
        r37 = residule_block(r36, options.gf_dim*4, name='g_r37')
        r38 = residule_block(r37, options.gf_dim*4, name='g_r38')
        r39 = residule_block(r38, options.gf_dim*4, name='g_r39')
        r40 = residule_block(r39, options.gf_dim*4, name='g_r40')
        r41 = residule_block(r40, options.gf_dim*4, name='g_r41')
        r42 = residule_block(r41, options.gf_dim*4, name='g_r42')
        r43 = residule_block(r42, options.gf_dim*4, name='g_r43')
        r44 = residule_block(r43, options.gf_dim*4, name='g_r44')
        r45 = residule_block(r44, options.gf_dim*4, name='g_r45')
        r46 = residule_block(r45, options.gf_dim*4, name='g_r46')
        r47 = residule_block(r46, options.gf_dim*4, name='g_r47')
        r48 = residule_block(r47, options.gf_dim*4, name='g_r48')
        r49 = residule_block(r48, options.gf_dim*4, name='g_r49')
        r50 = residule_block(r49, options.gf_dim*4, name='g_r50')
        r51 = residule_block(r50, options.gf_dim*4, name='g_r51')
        r52 = residule_block(r51, options.gf_dim*4, name='g_r52')
        r53 = residule_block(r52, options.gf_dim*4, name='g_r53')
        r54 = residule_block(r53, options.gf_dim*4, name='g_r54')
        r55 = residule_block(r54, options.gf_dim*4, name='g_r55')
        r56 = residule_block(r55, options.gf_dim*4, name='g_r56')
        r57 = residule_block(r56, options.gf_dim*4, name='g_r57')
        r58 = residule_block(r57, options.gf_dim*4, name='g_r58')
        r59 = residule_block(r58, options.gf_dim*4, name='g_r59')
        r60 = residule_block(r59, options.gf_dim*4, name='g_r60')
        r61 = residule_block(r60, options.gf_dim*4, name='g_r61')
        r62 = residule_block(r61, options.gf_dim*4, name='g_r62')
        r63 = residule_block(r62, options.gf_dim*4, name='g_r63')
        r64 = residule_block(r63, options.gf_dim*4, name='g_r64')
        r65 = residule_block(r64, options.gf_dim*4, name='g_r65')
        r66 = residule_block(r65, options.gf_dim*4, name='g_r66')
        r67 = residule_block(r66, options.gf_dim*4, name='g_r67')
        r68 = residule_block(r67, options.gf_dim*4, name='g_r68')
        r69 = residule_block(r68, options.gf_dim*4, name='g_r69')
        r70 = residule_block(r69, options.gf_dim*4, name='g_r70')
        r71 = residule_block(r70, options.gf_dim*4, name='g_r71')
        r72 = residule_block(r71, options.gf_dim*4, name='g_r72')
        r73 = residule_block(r72, options.gf_dim*4, name='g_r73')
        r74 = residule_block(r73, options.gf_dim*4, name='g_r74')
        r75 = residule_block(r74, options.gf_dim*4, name='g_r75')
        r76 = residule_block(r75, options.gf_dim*4, name='g_r76')
        r77 = residule_block(r76, options.gf_dim*4, name='g_r77')
        r78 = residule_block(r77, options.gf_dim*4, name='g_r78')
        r79 = residule_block(r78, options.gf_dim*4, name='g_r79')
        r80 = residule_block(r79, options.gf_dim*4, name='g_r80')
        r81 = residule_block(r80, options.gf_dim*4, name='g_r81')
        r82 = residule_block(r81, options.gf_dim*4, name='g_r82')
        r83 = residule_block(r82, options.gf_dim*4, name='g_r83')
        r84 = residule_block(r83, options.gf_dim*4, name='g_r84')
        r85 = residule_block(r84, options.gf_dim*4, name='g_r85')
        r86 = residule_block(r85, options.gf_dim*4, name='g_r86')
        r87 = residule_block(r86, options.gf_dim*4, name='g_r87')
        r88 = residule_block(r87, options.gf_dim*4, name='g_r88')
        r89 = residule_block(r88, options.gf_dim*4, name='g_r89')
        r90 = residule_block(r89, options.gf_dim*4, name='g_r90')
        r91 = residule_block(r90, options.gf_dim*4, name='g_r91')
        r92 = residule_block(r91, options.gf_dim*4, name='g_r92')
        r93 = residule_block(r92, options.gf_dim*4, name='g_r93')
        r94 = residule_block(r93, options.gf_dim*4, name='g_r94')
        r95 = residule_block(r94, options.gf_dim*4, name='g_r95')
        r96 = residule_block(r95, options.gf_dim*4, name='g_r96')
        r97 = residule_block(r96, options.gf_dim*4, name='g_r97')
        r98 = residule_block(r97, options.gf_dim*4, name='g_r98')
        r99 = residule_block(r98, options.gf_dim*4, name='g_r99')
        r100 = residule_block(r99, options.gf_dim*4, name='g_r100')
        r101 = residule_block(r100, options.gf_dim*4, name='g_r101')
        r102 = residule_block(r101, options.gf_dim*4, name='g_r102')
        r103 = residule_block(r102, options.gf_dim*4, name='g_r103')
        r104 = residule_block(r103, options.gf_dim*4, name='g_r104')
        r105 = residule_block(r104, options.gf_dim*4, name='g_r105')
        r106 = residule_block(r105, options.gf_dim*4, name='g_r106')
        r107 = residule_block(r106, options.gf_dim*4, name='g_r107')
        r108 = residule_block(r107, options.gf_dim*4, name='g_r108')
        r109 = residule_block(r108, options.gf_dim*4, name='g_r109')
        r110 = residule_block(r109, options.gf_dim*4, name='g_r110')
        r111 = residule_block(r110, options.gf_dim*4, name='g_r111')
        r112 = residule_block(r111, options.gf_dim*4, name='g_r112')
        r113 = residule_block(r112, options.gf_dim*4, name='g_r113')
        r114 = residule_block(r113, options.gf_dim*4, name='g_r114')
        r115 = residule_block(r114, options.gf_dim*4, name='g_r115')
        r116 = residule_block(r115, options.gf_dim*4, name='g_r116')
        r117 = residule_block(r116, options.gf_dim*4, name='g_r117')
        r118 = residule_block(r117, options.gf_dim*4, name='g_r118')
        r119 = residule_block(r118, options.gf_dim*4, name='g_r119')
        r120 = residule_block(r119, options.gf_dim*4, name='g_r120')
        r121 = residule_block(r120, options.gf_dim*4, name='g_r121')
        r122 = residule_block(r121, options.gf_dim*4, name='g_r122')
        r123 = residule_block(r122, options.gf_dim*4, name='g_r123')
        r124 = residule_block(r123, options.gf_dim*4, name='g_r124')
        r125 = residule_block(r124, options.gf_dim*4, name='g_r125')
        r126 = residule_block(r125, options.gf_dim*4, name='g_r126')
        r127 = residule_block(r126, options.gf_dim*4, name='g_r127')
        r128 = residule_block(r127, options.gf_dim*4, name='g_r128')
        r129 = residule_block(r128, options.gf_dim*4, name='g_r129')
        r130 = residule_block(r129, options.gf_dim*4, name='g_r130')
        r131 = residule_block(r130, options.gf_dim*4, name='g_r131')
        r132 = residule_block(r131, options.gf_dim*4, name='g_r132')
        r133 = residule_block(r132, options.gf_dim*4, name='g_r133')
        r134 = residule_block(r133, options.gf_dim*4, name='g_r134')
        r135 = residule_block(r134, options.gf_dim*4, name='g_r135')
        r136 = residule_block(r135, options.gf_dim*4, name='g_r136')
        r137 = residule_block(r136, options.gf_dim*4, name='g_r137')
        r138 = residule_block(r137, options.gf_dim*4, name='g_r138')
        r139 = residule_block(r138, options.gf_dim*4, name='g_r139')
        r140 = residule_block(r139, options.gf_dim*4, name='g_r140')
        r141 = residule_block(r140, options.gf_dim*4, name='g_r141')
        r142 = residule_block(r141, options.gf_dim*4, name='g_r142')
        r143 = residule_block(r142, options.gf_dim*4, name='g_r143')
        r144 = residule_block(r143, options.gf_dim*4, name='g_r144')
        r145 = residule_block(r144, options.gf_dim*4, name='g_r145')
        r146 = residule_block(r145, options.gf_dim*4, name='g_r146')
        r147 = residule_block(r146, options.gf_dim*4, name='g_r147')
        r148 = residule_block(r147, options.gf_dim*4, name='g_r148')
        r149 = residule_block(r148, options.gf_dim*4, name='g_r149')
        r150 = residule_block(r149, options.gf_dim*4, name='g_r150')
        
        d1 = deconv2d(r21, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def abs_sum(in_, target):
    return tf.reduce_sum(tf.abs(in_ - target))

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
