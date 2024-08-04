from vtons import IDMVTON
import torch

# CUDA 캐시 메모리 비우기
torch.cuda.empty_cache()


model = IDMVTON()

human_img_dest ='/home/moonsoo/Gits/jsv_package_test/moipg/taylor-.jpg' # 모델 사진 경로(서버 상 업로드 및 경로 복사)
garm_img_dest = '/home/moonsoo/Gits/jsv_package_test/moipg/09263_00.jpg' # 옷 사진 경로(서버 상 업로드 및 경로 복사)
garment_des = 'blue and gold silky blouse' # 옷에 대한 설명+


model.run_start_tryon(human_img_dest, garm_img_dest, garment_des)
