# NEW/app.py

import subprocess
import os

def run_start_tryon(human_img_dest, garm_img_dest, garment_des):
    # IDM_VTON 폴더 경로와 apple.py 파일 경로 설정
    idm_vton_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'IDM-VTON','gradio_demo'))
    apple_script_path = os.path.join(idm_vton_path, 'apple.py')
    
    # apple.py 파일 실행 및 인자 전달
    subprocess.run([
        'python', apple_script_path,
        '--human_img_path', human_img_dest,
        '--garm_img_path', garm_img_dest,
        '--garment_des', garment_des
    ])

if __name__ == "__main__":
    # 예시 인자
    human_img_dest ='/home/moonsoo/Gits/VTON/taylor-.jpg' # 모델 사진 경로(서버 상 업로드 및 경로 복사)
    garm_img_dest = '/home/moonsoo/Gits/VTON/09263_00.jpg' # 옷 사진 경로(서버 상 업로드 및 경로 복사)
    garment_des = 'blue and gold silky blouse' # 옷에 대한 설명+
    
    run_start_tryon(human_img_dest, garm_img_dest, garment_des)
