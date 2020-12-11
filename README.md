# 本文禁止转载！

之前微博上掀起了一股装逼文体的新热潮

该文体先抑后扬

将装逼隐藏在浮于表面的抱怨之中

装逼者总是在不经意间流露出贵族式的烦恼

看似抱怨

实则炫耀

这样的写作手法被称作——

凡尔赛文学

![](https://img-blog.csdnimg.cn/img_convert/07dd9079d8251f2433754a16524a0aae.png)

![](https://img-blog.csdnimg.cn/img_convert/12dabed7705a4b3393c9a469b25fdfcb.png)


本项目运用Paddlehub实现了根据关键词的凡尔赛文学自动生成器。

![](https://img-blog.csdnimg.cn/img_convert/77f0d98f1239010cb7efb9408579b678.png)


## 1. 安装相关库


```python
! pip install xlrd
! pip install --upgrade paddlehub
! pip install paddle-ernie
```

按照数据集要求对其进行整理，格式为“序号\t输入文本\t标签”

数据集来自知乎并进行了手动标注：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201211091436169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)


```python
import pandas as pd

df = pd.read_excel("data.xlsx")
Keys = df["关键词"].values
Txts = df["文案"].values

with open("format_data.txt", "w") as f:
    for i, k in enumerate(Keys):
        t = Txts[i]
        # t = "凡尔赛"
        f.write("{}\t{}\t{}\n".format(i, k, t))
```

## 2. 调用Paddlehub模型进行预训练

模型我们选用ERNIE-GEN模型

论文地址：[https://arxiv.org/abs/2001.11314](https://arxiv.org/abs/2001.11314)

![](https://img-blog.csdnimg.cn/img_convert/23c56c22d013a26a0164ac0c57288c6c.png)



```python
import paddlehub as hub

module = hub.Module(name="ernie_gen")

result = module.finetune(
    train_path='format_data.txt',
    save_dir="Versailles_param",
    max_steps=1200,
    noise_prob=0.1,
    save_interval=400,
    max_encode_len=60,
    max_decode_len=60
)

# 将训练参数打包为hub model
module.export(params_path=result['last_save_path'], module_name="Versailles", author="lyp")
!hub install Versailles
```

    [32m[2020-12-10 22:26:30,355] [    INFO] - Installing ernie_gen module[0m


    Downloading ernie_gen
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmp4ioerv1o/ernie_gen
    [==================================================] 100.00%


    [32m[2020-12-10 22:26:34,769] [    INFO] - Successfully installed ernie_gen-1.0.2[0m
    2020-12-10 22:26:34,813-INFO: get pretrain dir from https://ernie-github.cdn.bcebos.com/model-ernie1.0.1.tar.gz
    downloading https://ernie-github.cdn.bcebos.com/model-ernie1.0.1.tar.gz: 788478KB [00:14, 53950.68KB/s]                            
    [32m[INFO] 2020-12-10 22:26:59,170 [feature_column.py:  349]:	reading raw files from format_data.txt[0m
    2020-12-10 22:26:59,174-INFO: get pretrain dir from https://ernie-github.cdn.bcebos.com/model-ernie1.0.1.tar.gz
    2020-12-10 22:27:02,458-INFO: loading pretrained model from /tmp/466eabcffd6d6a83ae9cb97dd1a167bd
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:718: UserWarning: Varibale [ pooler.weight pooler.bias ] are not used, because not included in layers state_dict
      format(" ".join(unused_para_list)))
    [32m[2020-12-10 22:27:09,748] [    INFO] - [step 0 / 1200]train loss 6.70396, ppl 815.63086, elr 0.000e+00[0m
    [32m[2020-12-10 22:27:31,429] [    INFO] - [step 100 / 1200]train loss 4.11473, ppl 61.23570, elr 4.167e-05[0m
    [32m[2020-12-10 22:27:53,068] [    INFO] - [step 200 / 1200]train loss 2.74297, ppl 15.53309, elr 4.167e-05[0m
    [32m[2020-12-10 22:28:14,802] [    INFO] - [step 300 / 1200]train loss 2.20646, ppl 9.08347, elr 3.750e-05[0m
    [32m[2020-12-10 22:28:36,874] [    INFO] - [step 400 / 1200]train loss 1.73751, ppl 5.68315, elr 3.333e-05[0m
    [32m[2020-12-10 22:28:36,875] [    INFO] - save the model in Versailles_param/step_400_ppl_5.68315[0m
    [32m[2020-12-10 22:29:01,927] [    INFO] - [step 500 / 1200]train loss 2.28282, ppl 9.80433, elr 2.917e-05[0m
    [32m[2020-12-10 22:29:21,656] [    INFO] - [step 600 / 1200]train loss 1.66694, ppl 5.29591, elr 2.500e-05[0m
    [32m[2020-12-10 22:29:41,022] [    INFO] - [step 700 / 1200]train loss 1.54020, ppl 4.66554, elr 2.083e-05[0m
    [32m[2020-12-10 22:30:00,048] [    INFO] - [step 800 / 1200]train loss 0.99123, ppl 2.69455, elr 1.667e-05[0m
    [32m[2020-12-10 22:30:00,049] [    INFO] - save the model in Versailles_param/step_800_ppl_2.69455[0m
    [32m[2020-12-10 22:30:24,166] [    INFO] - [step 900 / 1200]train loss 1.20773, ppl 3.34589, elr 1.250e-05[0m
    [32m[2020-12-10 22:30:43,950] [    INFO] - [step 1000 / 1200]train loss 0.74044, ppl 2.09686, elr 8.333e-06[0m
    [32m[2020-12-10 22:31:02,997] [    INFO] - [step 1100 / 1200]train loss 0.92916, ppl 2.53238, elr 4.167e-06[0m
    [32m[2020-12-10 22:31:22,075] [    INFO] - [step 1200 / 1200]train loss 1.05983, ppl 2.88587, elr 0.000e+00[0m
    [32m[2020-12-10 22:31:22,076] [    INFO] - save the model in Versailles_param/step_1200_ppl_2.88587[0m
    [32m[2020-12-10 22:31:27,124] [    INFO] - [final step 1201]train loss 0.78724, ppl 2.19731, elr 0.000e+00[0m
    [32m[2020-12-10 22:31:27,126] [    INFO] - save the model in Versailles_param/step_1201_ppl_2.19731[0m
    [32m[2020-12-10 22:31:31,939] [    INFO] - Begin export the model save in Versailles_param/step_1201_ppl_2.19731.pdparams ...[0m
    [32m[2020-12-10 22:31:32,682] [    INFO] - The module has exported to /home/aistudio/Versailles[0m


    2020-12-10 22:31:38,676-INFO: font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
    2020-12-10 22:31:39,012-INFO: generated new fontManager
    Successfully installed Versailles


## 3. 运行预测


```python
import paddlehub as hub

module = hub.Module(directory="/home/aistudio/Versailles/")
```


```python
test_texts = ["凡尔赛"]
results = module.generate(texts=test_texts, use_gpu=False, beam_width=10)
for result in results[0]:
    print(result)
```

    今天又是努力搬砖的一天！根本没时间摸鱼，中午只能休息两个小时，虽然五点就能准时下班，但早上十点就得到
    今天又是努力搬砖的一天！根本没时间摸鱼中午只能休息两个小时虽然五点就能准时下班但早上十点就得到公司还
    今天看中了一栋别墅，我真的好喜欢位置也好，跑到楼顶就能看见天安门。但是真的太贵啦，买下它要花掉我一个
    今天看中了一栋别墅,我真的好喜欢位置也好,跑到楼顶就能看见天安门。但是真的太贵啦,买下它要花掉我一个
    真羡慕那些可以随随便便离家出走的孩子，我都出来一个月了，还没走出我家草坪。
    真羡慕年轻人啊，活力满满见个冠军奖杯那么激动，不像我，拿好几次都没啥感觉了，上次还差点手滑给摔了。
    今天看中了一栋别墅，我真的好喜欢位置也好，跑到楼顶就能看见天安门，但是真的太贵啦，买下它要花掉我一个
    我在沙发上背单词，突然牛奶热好了我去拿，回来的时候发现没加书签。我问他：“我背到哪了？”他不慌不忙地
    今天又是努力搬砖的一天！根本没时间摸鱼中午只能休息两个小时虽然五点就能准时下班但早上十点就得到公司，
    真羡慕那些可以随随便便离家出走的孩子，我都出来一个月了，还没走出我家草坪



```python
test_texts = ["房"]
results = module.generate(texts=test_texts, use_gpu=False, beam_width=5)
for result in results[0]:
    print(result)
```

    最近去试完衣服顺路回家都买几捧玫瑰，老公突然就说要买一个带院子的房子，种满玫瑰叫园丁专门打理，但是玫
    难受了，我竟然错过了悉尼歌剧院的演出！因为他非得拉着我去挑什么房子，一个千佛山脚下的普通别墅而已，至
    最近去试完衣服顺路回家都买几捧玫瑰，老公突然就说要买一个带院子的房子，种满玫瑰叫园丁专门打理。
    “最近有个追求者给我在陆家嘴买了套房但是我不想要我已经在汤臣一品有三层了我觉得他是在用钱侮辱我呜呜呜
    难受了，我竟然错过了悉尼歌剧院的演出！因为他非得拉着我去挑什么房子。



```python
test_texts = ["老公"]
results = module.generate(texts=test_texts, use_gpu=False, beam_width=5)
for result in results[0]:
am_width=5)
for result in results[0]:
    print(result)
```

    今天皮带忘带了去gucci随便买了个结果打孔的时候店员说腰太细了她从来没打过这么细的我觉得女生圆润一点好的呀
    “今天皮带忘带了去gucci随便买了个结果打孔的时候店员说腰太细了她从来没打过这么细的我觉得女生圆润一点好的
    今天皮带忘带了去gucci随便买了个结果打孔的时候店员说腰太细了她从来没打过这么近的我觉得女生圆润一点好的呀
    “今天皮带忘带了去gucci随便买了个结果打孔的时候店员说腰太细了她从来没打过这么近的我觉得女生圆润一点好的
    最近老公发了个朋友圈，说特朗普落选那天全朋友圈消费由他买单，虽然我知道他是开玩笑的，但是他那帮朋友真


## 4. 改进方向：

可以看到数据集太少，出现了明显的过拟合现象，可以多搜集一些。

## 5. 交流群：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201120115403928.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70#pic_center)
