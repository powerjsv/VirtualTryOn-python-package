# from vtons import IDMVTON
from models import IDMVTON


model = IDMVTON()


human_img_dest ='/home/moonsoo/Gits/VirtualTryOn/taylor-.jpg' # 모델 사진 경로(서버 상 업로드 및 경로 복사)
garm_img_dest = '/home/moonsoo/Gits/VirtualTryOn/09263_00.jpg' # 옷 사진 경로(서버 상 업로드 및 경로 복사)
garment_des = 'blue and gold silky blouse' # 옷에 대한 설명+


result = model.start_tryon(human_img_dest, garm_img_dest, garment_des)

print(result)


