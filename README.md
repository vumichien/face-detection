# Face-detection-api

## Language: 
Python 3.8

## Environment: 
```
pip intsll -r requirements.txt
```

## Run: 
```
python main.py
```

### Request
```
curl -X 'POST' \
      'http://127.0.0.1:8000/face_detect?threshold=0.1' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@test.jpg;type=image/jpeg'
```

### Response
```
{
  "distances": [
    0.1214
  ],
  "detect_close_face": 1,
  "detect_face": 7
}
```