import torch
import collections.abc as container_abcs
import collections

"""
参考博客：https://blog.csdn.net/acm_durante/article/details/122648066
torch.cat(tensors,dim)对序列(指list or tuple)数据内部的张量进行在指定维度（dim）上拼接
要求：1.数据的元素的维度一致（如都是4维），2. 要合并的维度上shape可以不一致，其余的shape必须一致（如[11, 1, 160, 160] at entry 0 and [10, 1, 160, 160] at entry 1）
意义：把多张图片合并成一个batch


参考博客：https://www.w3cschool.cn/article/4441770.html
torch.stack(tensors,dim)对序列(指list or tuple)数据内部的张量进行扩维(由dim来决定)拼接（可看出torch.cat）
要求：tensors序列的元素形状一样
意义：数据都是二维矩阵(平面)，它可以把这些一个个平面(矩阵)按第三维(例如：时间序列)压成一个三维的立方体，而立方体的长度就是时间序列长度
"""




def labelToMultiChs(img,label_nc=4):
    """将单通道label对应到多通道上"""
    batch_n,chs,x,y=img.size()    
    imgMultiChs=[]
    for i in range(label_nc):
        label_chs=(img==i).to(torch.uint8)
        imgMultiChs+=[label_chs]
    imgMultiChs=torch.cat(imgMultiChs,1)
    return imgMultiChs
 

#输入chs,x,y(tensor)
#return:1,x,y(tensor)
def probabilityToImg(img):
    """此方法会将结果归一化到[0,1]"""
    if len(img.size())==3:
        chs,x,y=img.size()
        imgNp=torch.zeros(x,y)
        for i in range(chs):
            imgNp+=((img[i,:,:]==1).float())*i
        imgNp=imgNp.unsqueeze(0)/3
        return imgNp
    elif len(img.size())==4:
        batch_n,chs,x,y=img.size()
        result=[]
        for k in range(batch_n):
            imgNp=torch.zeros(x,y)
            for i in range(chs):
                imgNp+=((img[k,i,:,:]==1).float())*i
            imgNp=(imgNp.unsqueeze(0)/(chs-1)).unsqueeze(0)
            result+=[imgNp]
        result=torch.cat(result,0)
        return result

def probabilityToimg_gpu(img):
    """此方法将以概率形式表达的图片转化成一维图片，并却图片取值为[0,1]。主要用于转化分割图片"""
    assert img.ndim == 4, "image should have 4 dimensions, but only have {}".format(img.ndim)
    device = img.device
    batch_n, chs, x, y = img.size()
    result = []
    for k in range(batch_n):
        imgNp = torch.zeros(x, y).to(device)
        for i in range(chs):
            imgNp += ((img[k,i,:,:]==1).float())*i
        imgNp=(imgNp.unsqueeze(0)/(chs-1)).unsqueeze(0)
        result += [imgNp]
    result = torch.cat(result, 0)
    return result

def normalize(img):
    """将值归一化到[0,1]中"""
    batch_n,chs,x_dim,y_dim=img.size()
    img=img.reshape(batch_n,chs*x_dim*y_dim)
    max_pixel=img.max(1)[0].unsqueeze(1)
    min_pixel=img.min(1)[0].unsqueeze(1)
    img=((img-min_pixel)/(max_pixel-min_pixel)).reshape(batch_n,chs,x_dim,y_dim)
    return img

default_collate_err_msg_format = ("default_collate: batch must contain tensors, numbers; but found {}:{}")

def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)

    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int): # int_classes
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type, batch))