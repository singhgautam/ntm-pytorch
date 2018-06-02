import torch
import torchvision
from externalmemory import ExternalMemory

#initialize device
print 'torch.version',torch.__version__
print 'torch.cuda.is_available()',torch.cuda.is_available()
device = torch.device("cpu") # other "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True}

#set hyper-params
mem_size = 100
mem_width = 20

#make memory
externalmemory = ExternalMemory(
    mem_size = mem_size,
    mem_width = mem_width,
    device = device
)
torchvision.utils.save_image(externalmemory.mem, '../imgsaves/test-mem.png')

#test read
w = torch.zeros(mem_size)
w[5] = 1.0
mem_out = externalmemory.read(w)
mem_out = torch.stack((mem_out, mem_out))
torchvision.utils.save_image(mem_out, '../imgsaves/test-mem_out.png')

#test erase
e = torch.ones(mem_width)
externalmemory.erase(w,e)
torchvision.utils.save_image(externalmemory.mem, '../imgsaves/test-mem_after_erase.png')

#test add
a = torch.zeros(mem_width)
a[10] = 1.0
externalmemory.add(w,a)
torchvision.utils.save_image(externalmemory.mem, '../imgsaves/test-mem_after_erase_and_add.png')

#test-address by content
w_adr_content = externalmemory.address_by_content(a,10.0)
torchvision.utils.save_image(torch.stack((w_adr_content,w_adr_content)), '../imgsaves/test-w_adr_by_content.png')

#test-address by location
shift = torch.zeros(3)
shift[0] = 1.0
w_adr_shift = externalmemory.address_by_location(w_adr_content,shift)
torchvision.utils.save_image(torch.stack((w_adr_shift,w_adr_shift)), '../imgsaves/test-w_adr_by_location.png')

#test-address full
_w = torch.zeros(mem_size)
_w[10] = 1.0
w_adr = externalmemory.address(a,10.0,10.0,1.0,shift,_w)
torchvision.utils.save_image(torch.stack((w_adr,w_adr)), '../imgsaves/test-w_adr_full.png')
