from torchinfo import summary
from ChangeFormer import ChangeFormer

model = ChangeFormer(3, 2)

print(summary(model, input_size=((1, 3, 256, 256), (1, 3, 256, 256))))
      


