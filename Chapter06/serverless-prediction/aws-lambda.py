from __future__ import print_function

import json
import urllib
import boto3
import base64
import io

print('Loading function')

s3 = boto3.client('s3')
rekognition = boto3.client("rekognition", <AWS Region>)

bucket=<bucket Name>
key_path='raw-image/'

def lambda_handler(event, context):
    
    output={}
    try:
        if event['operation']=='label-detect':
            print('Detecting label')
            fileName= event['fileName']
            bucket_key=key_path + fileName
            data = base64.b64decode(event['base64Image'])
            image=io.BytesIO(data)
            s3.upload_fileobj(image, bucket, bucket_key)
            rekog_response = rekognition.detect_labels(Image={"S3Object": {"Bucket": bucket,"Name": bucket_key,}},MaxLabels=5,MinConfidence=90,)
            for label in rekog_response['Labels']:
                output[label['Name']]=label['Confidence']
        else:
            print('Detecting faces')
            FEATURES_BLACKLIST = ("Landmarks", "Emotions", "Pose", "Quality", "BoundingBox", "Confidence")
            fileName= event['fileName']
            bucket_key=key_path + fileName
            data = base64.b64decode(event['base64Image'])
            image=io.BytesIO(data)
            s3.upload_fileobj(image, bucket, bucket_key)
            face_response = rekognition.detect_faces(Image={"S3Object": {"Bucket": bucket,	"Name": bucket_key, }}, Attributes=['ALL'],)
            for face in face_response['FaceDetails']:
                output['Face']=face['Confidence']
                for emotion in face['Emotions']:
                    output[emotion['Type']]=emotion['Confidence']
                for feature, data in face.iteritems():
                    if feature not in FEATURES_BLACKLIST:
                        output[feature]=data
    except Exception as e:
        print(e)
        raise e		
			
    return output			
			
			
			