import subprocess
import os


class idmvton():
    def __init__(self):
        folder_path = '../IDM-VTON/ckpt/densepose'

        if os.path.exists(folder_path) and not os.listdir(folder_path):
            repo_id = "yisol/IDM-VTON"
            save_path = "../IDM-VTON/ckpt"

            # Download checkpoints to run the vton model
            hf_hub_download(repo_id=repo_id, filename="densepose/model_final_162be9.pkl", local_dir="../IDM-VTON/ckpt")
            hf_hub_download(repo_id=repo_id, filename="humanparsing/parsing_atr.onnx", local_dir="../IDM-VTON/ckpt")
            hf_hub_download(repo_id=repo_id, filename="humanparsing/parsing_lip.onnx", local_dir="../IDM-VTON/ckpt")
            hf_hub_download(repo_id=repo_id, filename="openpose/ckpts/body_pose_model.pth", local_dir="../IDM-VTON/ckpt")
        else:
            print("Checkpoints already exist.")
            print()


    def run_start_tryon(self, human_img_dest, garm_img_dest, garment_des):
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



"""
Directory Strucuture by __init__:

ckpt
|-- densepose
    |-- model_final_162be9.pkl
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx

|-- openpose
    |-- ckpts
        |-- body_pose_model.pth
    
"""