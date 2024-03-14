from fastapi import FastAPI
import comp_other as comp_other_router
import comp_people as comp_people_router
import comp_celeb as comp_celeb_router
from fastapi.middleware.cors import CORSMiddleware

import os

# 텐서플로우 버전 확인
import tensorflow as tf
# print(tf.__version__)

# # dlib 버전 확인
# import dlib
# print(dlib.__version__)

# oneDNN CPU performance optimizations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '-1'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

checkGpu = tf.config.list_physical_devices('GPU')

print("checkGpu is ",checkGpu)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  
    allow_headers=["*"],
)

app.include_router(comp_other_router.router)
app.include_router(comp_people_router.router)
app.include_router(comp_celeb_router.router)