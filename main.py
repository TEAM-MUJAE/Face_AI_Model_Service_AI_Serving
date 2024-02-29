from fastapi import FastAPI
import comp_other as comp_other_router
import comp_people as comp_people_router

app = FastAPI()

app.include_router(comp_other_router.router)
app.include_router(comp_people_router.router)
