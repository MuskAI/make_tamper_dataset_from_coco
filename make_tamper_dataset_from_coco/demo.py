import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms as t
import numpy as np
from PIL import Image
a = np.ones((320,320,3), dtype='uint8')
b = np.ones((320,320,3), dtype='uint8')

a = Image.fromarray(a)
b = Image.fromarray(b)
trans = t.Compose([
    t.Resize((256,256)),
    t.RandomCrop((224,224)),
    t.ToTensor()
])


class Mydata(Dataset):
    def __init__(self):
        a = np.ones((320, 320, 3), dtype='uint8')
        b = np.ones((320, 320, 3), dtype='uint8')
        a_data_list = []
        b_data_list = []
        for i in range(100):
            a_data_list.append(a)
            b_data_list.append(b)
        self.a = a_data_list
        self.b = b_data_list

    def __getitem__(self, item):
        src = self.a[item]
        gt  = self.b[item]
        src = Image.fromarray(src)
        gt = Image.fromarray(gt)
        src = trans(src)
        gt = trans(gt)
        return {'src':src,'gt':gt}
    def __len__(self):

        return len(self.a)

    def __check(self):
        pass
    def hahah(self):
        pass
mydata = Mydata()
trainloader =torch.utils.data.DataLoader(dataset=mydata,batch_size=2,shuffle=True,num_workers=1)

if __name__ == '__main__':
    for idx,item in enumerate(trainloader):
        print(idx,item['src'].shape)
    #
    # for i in trainloader:
    #     print(i)