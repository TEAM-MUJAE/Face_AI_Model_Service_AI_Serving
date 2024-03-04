from fastapi import FastAPI
import comp_other as comp_other_router
import comp_people as comp_people_router
import comp_celeb as comp_celeb_router
import os

# 텐서플로우 버전 확인
# import tensorflow as tf
# print(tf.__version__)

# oneDNN CPU performance optimizations
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

app.include_router(comp_other_router.router)
app.include_router(comp_people_router.router)
app.include_router(comp_celeb_router.router)
