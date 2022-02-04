import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.logger import logger
from mtcnn import MTCNN
from PIL import Image
from utils.detect_face import calculate_distance

app = FastAPI(
    title="AI Model Face Detection",
    version="0.0.1"
)
device = "cpu"
mtcnn = MTCNN(
        image_size=256, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
)

logger.info("Loaded models successfully")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/face_detect")
def predict(file: UploadFile = File(...),  threshold: float = 0.1):
    """ Detect face from image

           Parameters:
           ----------------

           file: upload file image

               ALLOWED_EXTENSION = {'jpg', 'jpeg'}

           threshold: float

               threshold for distances between faces and input image

           Returns:
           ----------------
           distances: list

               list of distances between faces and input image

           detect_close_face: interger

               number of close faces detected

           detect_face: interger

               number of faces detected

    """
    threshold = threshold
    contents = file.file
    image = Image.open(contents)
    boxes, probs = mtcnn.detect(image, landmarks=False)
    distances = calculate_distance(image, boxes.tolist(), threshold)
    return {
        "distances": distances,
        "detect_close_face": len(distances),
        "detect_face": len(probs.tolist())
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, debug=True)