#!/bin/bash

echo "Prediction for 1st Image:"
echo "--------------------------------"
(echo -n '{"data": "'; base64 test-1.jpg; echo '"}') | curl -X POST -H "Content-Type: application/json" -d @- http://127.0.0.1:5000

echo "Prediction for 2nd Image:"
echo "--------------------------------"
(echo -n '{"data": "'; base64 test-1.jpeg; echo '"}') | curl -X POST -H "Content-Type: application/json" -d @- http://127.0.0.1:5000
